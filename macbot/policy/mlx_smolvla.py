"""MLX sampling for the stock SmolVLA/SmolVLM2-500M architecture."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .mlx_chunk import MLXChunkIO, stage_call

_MODEL = "vlm_with_expert.vlm.model."
_TEXT = _MODEL + "text_model."
_VISION = _MODEL + "vision_model."
_EXPERT = "vlm_with_expert.lm_expert."


class SmolNetwork:
    """Batch the RGB views and project cross-attention K/V once per chunk."""

    def __init__(self, weights, dtype):
        self.dtype = dtype
        self.w = {k.removeprefix("model."): v.astype(dtype) for k, v in weights.items()}
        self.w.pop("vlm_with_expert.vlm.lm_head.weight", None)
        patch = _VISION + "embeddings.patch_embedding.weight"
        self.w[patch] = mx.contiguous(self.w[patch].transpose(0, 2, 3, 1))
        self.packed = {}
        for stem, depth in ((_VISION + "encoder.", 12), (_TEXT, 16), (_EXPERT, 16)):
            for i in range(depth):
                root = f"{stem}layers.{i}."
                if stem != _EXPERT or i % 2 == 0:
                    self._pack(root + "self_attn.", "q_proj", "k_proj", "v_proj")
                if stem != _VISION + "encoder.":
                    self._pack(root + "mlp.", "gate_proj", "up_proj")
        mx.eval(self.w, self.packed)

    def _pack(self, root, *names):
        weights = [self.w.pop(root + n + ".weight") for n in names]
        biases = [self.w.pop(root + n + ".bias", None) for n in names]
        cuts = tuple(np.cumsum([x.shape[0] for x in weights[:-1]]).tolist())
        self.packed[root] = (mx.concatenate(weights), None if biases[0] is None else mx.concatenate(biases), cuts)

    def project(self, x, root):
        w, b, cuts = self.packed[root]
        x = x.astype(w.dtype) @ w.T
        return mx.split(x if b is None else x + b, cuts, axis=-1)

    def linear(self, x, name):
        w = self.w[name + ".weight"]
        y = x.astype(w.dtype) @ w.T
        b = self.w.get(name + ".bias")
        return y if b is None else y + b

    @staticmethod
    def heads(x):
        return x.reshape(*x.shape[:2], -1, 64).transpose(0, 2, 1, 3)

    @staticmethod
    def attention(q, k, v, mask=None):
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=mask).transpose(0, 2, 1, 3).flatten(2)

    @staticmethod
    def rope(x, offset):
        # LeRobot's SmolVLA attention uses 10000 independently of the HF backbone.
        return mx.fast.rope(x, 64, traditional=False, base=10000, scale=1, offset=offset)

    def vision(self, images):
        emb = _VISION + "embeddings."
        x = mx.conv2d(images.astype(self.dtype), self.w[emb + "patch_embedding.weight"], stride=16)
        x = x.flatten(1, 2) + self.w[emb + "patch_embedding.bias"] + self.w[emb + "position_embedding.weight"]

        def norm(z, name):
            return mx.fast.layer_norm(z, self.w[name + ".weight"], self.w[name + ".bias"], 1e-6)

        for i in range(12):
            root = f"{_VISION}encoder.layers.{i}."
            q, k, v = map(self.heads, self.project(norm(x, root + "layer_norm1"), root + "self_attn."))
            x = x + self.linear(self.attention(q, k, v), root + "self_attn.out_proj")
            h = self.linear(norm(x, root + "layer_norm2"), root + "mlp.fc1")
            x = x + self.linear(nn.gelu_approx(h), root + "mlp.fc2")
        x = norm(x, _VISION + "post_layernorm")
        # Group each 4x4 patch neighborhood in the original connector's row/column/channel order.
        x = x.reshape(x.shape[0], 8, 4, 8, 4, x.shape[-1]).transpose(0, 1, 3, 2, 4, 5).flatten(3).flatten(1, 2)
        return self.linear(x, _MODEL + "connector.modality_projection.proj")

    def rms(self, x, name):
        return mx.fast.rms_norm(x, self.w[name + ".weight"], 1e-5)

    def residual(self, x, attn, root):
        x = x + self.linear(attn, root + "self_attn.o_proj")
        h = self.rms(x, root + "post_attention_layernorm")
        gate, up = self.project(h, root + "mlp.")
        return x + self.linear(nn.silu(gate) * up, root + "mlp.down_proj")

    def prefix(self, vision, tokens, state):
        text = self.w[_TEXT + "embed_tokens.weight"][tokens]
        scale = text.shape[-1] ** 0.5
        x = mx.concatenate((vision.flatten(0, 1)[None] * scale, text * scale,
                            self.linear(state, "state_proj")[:, None]), axis=1)
        length = x.shape[1]
        indices = mx.arange(length)
        mask = ((indices[:, None] == length - 1) | (indices[None, :] < length - 1))[None, None]
        cache = []
        for i in range(16):
            root = f"{_TEXT}layers.{i}."
            h = self.rms(x, root + "input_layernorm")
            q, k, v = map(self.heads, self.project(h, root + "self_attn."))
            k = self.rope(k, 0)
            if i % 2:
                cross = f"{_EXPERT}layers.{i}.self_attn."
                cache.append(tuple(self.heads(self.linear(z.transpose(0, 2, 1, 3).flatten(2), cross + name))
                                   for z, name in ((k, "k_proj"), (v, "v_proj"))))
            else:
                cache.append((k, v))
            if i < 15:  # The final prefix layer contributes only its K/V.
                x = self.residual(x, self.attention(self.rope(q, 0), k, v, mask), root)
        return tuple(cache)

    def velocity(self, actions, time, cache):
        x = self.linear(actions, "action_in_proj")
        t = mx.broadcast_to(time[:, None, :].astype(x.dtype), x.shape)
        h = self.linear(mx.concatenate((x, t), axis=-1), "action_time_mlp_in")
        x = self.linear(nn.silu(h), "action_time_mlp_out")
        after_prefix = cache[0][0].shape[-2]
        # The prefix is visible to every action; self-attention in the suffix is causal.
        iota = mx.arange(x.shape[1])
        suffix_mask = iota[None, :] <= iota[:, None]
        self_mask = mx.concatenate((mx.ones((x.shape[1], after_prefix), dtype=mx.bool_), suffix_mask), axis=-1)
        for i, (prefix_k, prefix_v) in enumerate(cache):
            root = f"{_EXPERT}layers.{i}."
            h = self.rms(x, root + "input_layernorm")
            if i % 2:
                q = self.rope(self.heads(self.linear(h, root + "self_attn.q_proj")), 0)
                k, v = prefix_k, prefix_v
                mask = None
            else:
                q, k, v = map(self.heads, self.project(h, root + "self_attn."))
                q, k = self.rope(q, after_prefix), self.rope(k, after_prefix)
                k, v = mx.concatenate((prefix_k, k), axis=2), mx.concatenate((prefix_v, v), axis=2)
                mask = self_mask
            x = self.residual(x, self.attention(q, k, v, mask), root)
        return self.linear(self.rms(x, _EXPERT + "norm"), "action_out_proj").astype(mx.float32)


class SmolVLAMLX(MLXChunkIO):
    kind = "smolvla"

    def __init__(self, path, task, *, dtype=mx.float16, **io):
        from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding

        super().__init__(path, task, **io)
        conf = self.config
        if (conf.vlm_model_name != "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
                or tuple(conf.resize_imgs_with_padding or ()) != (512, 512)
                or conf.num_vlm_layers != 16 or conf.num_expert_layers not in (-1, 0, 16)
                or conf.self_attn_every_n_layers != 2 or conf.attention_mode != "cross_attn"
                or conf.expert_width_multiplier != 0.75 or not conf.use_cache or conf.empty_cameras
                or conf.add_image_special_tokens or conf.prefix_length > 0 or conf.adapt_to_pi_aloha):
            raise ValueError("MLX SmolVLA requires the stock SmolVLM2-500M configuration")
        self.steps = conf.num_steps
        if self.steps < 1:
            raise ValueError("At least one denoising step is required")
        try:
            net = SmolNetwork(mx.load(str(path / "model.safetensors")), dtype)
        except KeyError as exc:
            raise ValueError(f"Missing SmolVLA weight: {exc}") from exc
        self.vision, self.prefix, self.velocity = net.vision, net.prefix, net.velocity
        times = self.torch.tensor([1 + i * (-1.0 / self.steps) for i in range(self.steps)])
        width = net.w["action_in_proj.weight"].shape[0]
        emb = create_sinusoidal_pos_embedding(times, width, conf.min_period, conf.max_period, device=times.device)
        self.time_embeddings = mx.array(emb.float().numpy())

    def enable_compile(self):
        self.vision, self.prefix, self.velocity = map(mx.compile, (self.vision, self.prefix, self.velocity))

    def prepare(self, imgs, q):
        from lerobot.policies.common.vla_utils import pad_vector, resize_with_pad
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        batch = self.prepare_batch(imgs, q)
        images = self.torch.cat([batch[key] for key in self.config.image_features], dim=0)
        images = resize_with_pad(images, 512, 512, pad_value=0) * 2 - 1
        tokens = batch[OBS_LANGUAGE_TOKENS].numpy()
        mask = batch[OBS_LANGUAGE_ATTENTION_MASK].numpy()
        state = pad_vector(batch["observation.state"], self.config.max_state_dim)
        return mx.array(images.permute(0, 2, 3, 1).numpy()), mx.array(tokens[mask][None]), mx.array(state.numpy())

    def __call__(self, imgs, q, *, noise=None, stage=None):
        with self.torch.inference_mode():
            images, tokens, state = stage_call(stage, "prepare", self.prepare, imgs, q)
            vision = stage_call(stage, "vision", self.vision, images)
            cache = stage_call(stage, "prefix_and_kv", self.prefix, vision, tokens, state)
            x = self.starting_noise(noise)
            for i in range(self.steps):
                v = stage_call(stage, f"denoise_{i:02d}", self.velocity, x, self.time_embeddings[i:i + 1], cache)
                x = x + (-1.0 / self.steps) * v
            return stage_call(stage, "readback_unnormalise", self.finish, x)
