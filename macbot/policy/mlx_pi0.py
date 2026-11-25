"""MLX inference for stock LeRobot PI0 PaliGemma checkpoints."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .mlx_chunk import MLXChunkIO, stage_call
from .mlx_pi05 import _EXPERT, GuidedMLXFlow, PaliGemmaNetwork
from .pali_tokenizer import local_pali_tokenizer


class PI0Network(PaliGemmaNetwork):
    adaptive_expert = False

    def flow_shapes(self, a, d):
        state_dim = self.w["state_proj.weight"].shape[1] if "state_proj.weight" in self.w else 0
        return (
            ("state_proj", (a, state_dim)), ("action_in_proj", (a, d)), ("action_out_proj", (d, a)),
            ("time_mlp_in", (a, 2 * a)), ("time_mlp_out", (a, a)),
        )

    def expert_inputs(self, valid, horizon):
        # The first suffix token is state. It may see the prefix and itself,
        # while the subsequent action block is bidirectional.
        positions = self.pos_embed(mx.sum(valid, axis=1, keepdims=True) + mx.arange(horizon + 1)[None])
        suffix = (mx.arange(horizon + 1)[:, None] != 0) | (mx.arange(horizon + 1)[None, :] == 0)
        prefix = mx.broadcast_to(valid[:, None, :], (valid.shape[0], horizon + 1, valid.shape[1]))
        suffix = mx.broadcast_to(suffix, (valid.shape[0], horizon + 1, horizon + 1))
        return positions, mx.concatenate((prefix, suffix), axis=-1)[:, None]

    def velocity(self, actions, state_token, time_embedding, cache, positions, mask):
        x = self.linear(actions, "action_in_proj")
        t = mx.broadcast_to(time_embedding[:, None, :].astype(x.dtype), x.shape)
        x = self.linear(nn.silu(self.linear(mx.concatenate((x, t), axis=-1), "time_mlp_in")), "time_mlp_out")
        x = mx.concatenate((state_token, x), axis=1)
        for i, kv in enumerate(cache):
            x, _ = self.text_layer(x, f"{_EXPERT}layers.{i}.", positions, mask, previous=kv)
        return self.linear(self.norm(x[:, 1:], _EXPERT + "norm", None)[0], "action_out_proj")


class PI0MLX(GuidedMLXFlow, MLXChunkIO):
    kind = "pi0"

    def __init__(self, path, task, steps=None, *, dtype=mx.float16, tokenizer=None, **io):
        from importlib import import_module

        from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding

        import_module("lerobot.policies.pi0.processor_pi0")
        # Only a local tokenizer replaces the one named by the saved processor.
        super().__init__(path, task, tokenizer=tokenizer or local_pali_tokenizer(path), **io)
        config = self.config
        if config.paligemma_variant != "gemma_2b" or tuple(config.image_resolution) != (224, 224):
            raise ValueError("MLX pi0 requires the stock 224px Gemma 2B prefix")
        if config.use_relative_actions:
            raise ValueError("MLX pi0 requires an absolute-action checkpoint")
        self.steps = config.num_inference_steps if steps is None else steps
        if self.steps < 1:
            raise ValueError("At least one denoising step is required")
        self.rtc_beta, self.rtc_schedule = None, "linear"
        self.network = PI0Network(mx.load(str(path / "model.safetensors")), dtype=dtype,
                                  fused_rope=True, pack_linears=True)
        self.vision, self.prefix, self.velocity = self.network.vision, self.network.prefix, self.network.velocity
        self.times = [1.0 + i * (-1.0 / self.steps) for i in range(self.steps)]
        embed = create_sinusoidal_pos_embedding(
            self.torch.tensor(self.times, dtype=self.torch.float32),
            self.network.w["action_in_proj.weight"].shape[0], config.min_period, config.max_period,
            device=self.torch.device("cpu"),
        )
        self.time_embeddings = mx.array(embed.float().numpy())

    def enable_compile(self):
        self.vision, self.prefix, self.velocity = map(mx.compile, (self.vision, self.prefix, self.velocity))

    def prepare(self, imgs, q):
        from lerobot.policies.common.vla_utils import pad_vector, resize_with_pad_torch
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        batch = self.prepare_batch(imgs, q)
        res = tuple(self.config.image_resolution)
        images = []
        for key in self.config.image_features:
            if key not in self.cameras:
                continue
            image = batch[key].float().permute(0, 2, 3, 1)
            if tuple(image.shape[1:3]) != res:
                image = resize_with_pad_torch(image, *res)
            images.append((image * 2 - 1).numpy())
        # Only trailing padding can be removed with the contiguous fused RoPE path.
        mask = batch[OBS_LANGUAGE_ATTENTION_MASK].numpy()
        n = int(mask.sum())
        if not mask[0, :n].all() or mask[0, n:].any():
            raise ValueError("MLX pi0 requires right-padded text")
        tokens = batch[OBS_LANGUAGE_TOKENS].numpy()[:, :n]
        state = pad_vector(batch["observation.state"].float(), self.config.max_state_dim)
        return (mx.array(np.concatenate(images)), mx.array(tokens, dtype=mx.int32),
                mx.array(mask[:, :n]), mx.array(state.numpy()))

    def __call__(self, imgs, q, leftover=None, delay=0, horizon=None, return_raw=False, *, noise=None, stage=None):
        with self.torch.inference_mode():
            images, tokens, mask, state = stage_call(stage, "prepare", self.prepare, imgs, q)
            vision = stage_call(stage, "vision", self.vision, images)
            valid, cache = stage_call(stage, "prefix_and_kv", self.prefix, vision, tokens, mask, False)
            x = self.starting_noise(noise)
            positions, keep = self.network.expert_inputs(valid, x.shape[1])
            state_token = self.network.linear(state, "state_proj")[:, None]
            guidance = (self.rtc_target(leftover, delay, horizon, x.shape)
                        if leftover is not None and len(leftover) > delay > 0 else None)
            for i, time in enumerate(self.times):
                v = stage_call(stage, f"denoise_{i:02d}", self.velocity, x, state_token,
                               self.time_embeddings[i:i + 1], cache, positions, keep).astype(mx.float32)
                if guidance is not None:
                    target, weights = guidance
                    strength = (self.rtc_beta if i == 0 else
                                min(self.rtc_beta, (time * time + (1 - time) ** 2) / (time * (1 - time))))
                    v = v - strength * ((target - (x - time * v)) * weights)
                x = x + (-1.0 / self.steps) * v
            return stage_call(stage, "readback_unnormalise", self.finish, x, return_raw)
