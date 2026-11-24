"""MLX inference for LeRobot X-VLA with Florence-2 and padded joint actions."""
from __future__ import annotations

import math
from importlib import import_module

import mlx.core as mx
import mlx.nn as nn

from .mlx_chunk import DenseWeights, MLXChunkIO, attention, stage_call

_VISION = "vlm.vision_tower."
_PROJECTOR = "vlm.multi_modal_projector."
_TEXT = "vlm.language_model.encoder."
_HEAD = "transformer."


class XVLANetwork(DenseWeights):
    def __init__(self, weights, config, florence, dtype):
        super().__init__({k.removeprefix("model."): v for k, v in weights.items()}, dtype)
        self.config, self.visual, self.text = config, florence.vision_config, florence.text_config
        self.text_embedding = self.w.get("vlm.language_model.shared.weight")
        if self.text_embedding is None:
            self.text_embedding = self.w[_TEXT + "embed_tokens.weight"]
        for i in range(self.text.encoder_layers):
            root = f"{_TEXT}layers.{i}.self_attn."
            for param in ("weight", "bias"):
                self.w[root + f"qkv.{param}"] = mx.concatenate(
                    [self.w.pop(root + f"{p}_proj.{param}") for p in "qkv"]
                )
        mx.eval(self.w)

    def spatial_attention(self, x, root, heads):
        b, h, w, c = x.shape
        s = self.visual.window_size
        ph, pw = (-h) % s, (-w) % s
        windows = mx.pad(x, [(0, 0), (0, ph), (0, pw), (0, 0)])
        nh, nw = (h + ph) // s, (w + pw) // s
        windows = windows.reshape(b, nh, s, nw, s, c).transpose(0, 1, 3, 2, 4, 5).reshape(-1, s * s, c)
        q, k, v = mx.split(self.linear(windows, root + ".qkv"), 3, axis=-1)
        windows = self.linear(attention(q, k, v, heads), root + ".proj")
        return windows.reshape(b, nh, nw, s, s, c).transpose(0, 1, 3, 2, 4, 5).reshape(
            b, nh * s, nw * s, c
        )[:, :h, :w]

    def channel_attention(self, x, root, groups):
        b, h, w, c = x.shape
        qkv = self.linear(x, root + ".qkv").reshape(b, h * w, 3, groups, c // groups)
        q, k, v = (z.squeeze(0) for z in mx.split(qkv.transpose(2, 0, 3, 4, 1), 3, axis=0))
        # Here each channel attends over its group, with the spatial axis as the embedding.
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=(h * w)**-0.5)
        return self.linear(y.transpose(0, 3, 1, 2).reshape(b, h, w, c), root + ".proj")

    def vision(self, x):
        c = self.visual
        for i, depth in enumerate(c.depths):
            root = f"{_VISION}convs.{i}."
            if c.patch_prenorm[i]:
                x = self.norm(x, root + "norm")
            x = self.conv(x, root + "conv", stride=c.patch_stride[i], padding=c.patch_padding[i])
            if not c.patch_prenorm[i]:
                x = self.norm(x, root + "norm")
            for j in range(depth):
                for kind in ("spatial", "channel"):
                    root = f"{_VISION}blocks.{i}.{j}.{kind}_block."
                    x = x + self.conv(x, root + "conv1", padding=1, groups=x.shape[-1])
                    h = self.norm(x, root + "norm1")
                    x = x + (self.spatial_attention(h, root + "window_attn", c.num_heads[i]) if kind == "spatial"
                             else self.channel_attention(h, root + "channel_attn", c.num_groups[i]))
                    x = x + self.conv(x, root + "conv2", padding=1, groups=x.shape[-1])
                    h = self.linear(self.norm(x, root + "norm2"), root + "ffn.fc1")
                    x = x + self.linear(nn.gelu(h), root + "ffn.fc2")
        _, h, w, _ = x.shape
        pos = _PROJECTOR + "image_position_embed."
        cols = self.w[pos + "column_embeddings.weight"][:w]
        rows = self.w[pos + "row_embeddings.weight"][:h]
        x = x + mx.concatenate((mx.broadcast_to(cols, (h, w, cols.shape[-1])),
                                mx.broadcast_to(rows[:, None], (h, w, rows.shape[-1]))), axis=-1)
        x = x.flatten(1, 2) + self.w[_PROJECTOR + "visual_temporal_embed.pos_idx_to_embed"][0]
        x = mx.concatenate((x.mean(axis=1, keepdims=True), x), axis=1)
        return self.norm(self.linear(x, _PROJECTOR + "image_projection"), _PROJECTOR + "image_proj_norm")

    def encode_text(self, vision, tokens):
        x = mx.concatenate((vision[:1], self.text_embedding[tokens]), axis=1)
        positions = self.w[_TEXT + "embed_positions.weight"][2:2 + x.shape[1]]
        x = self.norm(x + positions, _TEXT + "layernorm_embedding")
        for i in range(self.text.encoder_layers):
            root = f"{_TEXT}layers.{i}."
            q, k, v = mx.split(self.linear(x, root + "self_attn.qkv"), 3, axis=-1)
            y = attention(q, k, v, self.text.encoder_attention_heads)
            x = self.norm(x + self.linear(y, root + "self_attn.out_proj"), root + "self_attn_layer_norm")
            y = self.linear(nn.gelu(self.linear(x, root + "fc1")), root + "fc2")
            x = self.norm(x + y, root + "final_layer_norm")
        return x

    def domain_linear(self, x, name, domain):
        w = self.w[name + ".fc.weight"][domain].reshape(x.shape[-1], -1)
        return mx.addmm(self.w[name + ".bias.weight"][domain], x, w)

    def condition(self, vision, tokens, domain):
        text = self.encode_text(vision, tokens)
        count, length, dim = vision.shape
        # Absent views are zero before projection, but still take part in action attention.
        aux = mx.pad(vision[1:], [(0, self.config.num_image_views - count), (0, 0), (0, 0)]).reshape(1, -1, dim)
        if self.config.use_hetero_proj:
            text = self.domain_linear(text, _HEAD + "vlm_proj", domain)
            aux = self.domain_linear(aux, _HEAD + "aux_visual_proj", domain)
        else:
            text = self.linear(text, _HEAD + "vlm_proj")
            aux = self.linear(aux, _HEAD + "aux_visual_proj")
        context = mx.concatenate((text, aux), axis=1)
        start = self.config.chunk_size
        end = start + context.shape[1]
        if end > self.w[_HEAD + "pos_emb"].shape[1]:
            raise ValueError("X-VLA input exceeds the checkpoint's positional embeddings")
        context = context + self.w[_HEAD + "pos_emb"][:, start:end]
        if self.config.len_soft_prompts:
            prompts = self.w[_HEAD + "soft_prompt_hub.weight"][domain].reshape(1, -1, self.config.hidden_size)
            context = mx.concatenate((context, prompts), axis=1)
        return context

    def predict(self, actions, state, time, context, domain):
        n = actions.shape[1]
        action_input = mx.concatenate((actions, mx.broadcast_to(state[:, None], (1, n, state.shape[-1])),
                                       mx.broadcast_to(time[None, None], (1, n, time.shape[-1]))), axis=-1)
        x = self.domain_linear(action_input, _HEAD + "action_encoder", domain)
        x = mx.concatenate((x + self.w[_HEAD + "pos_emb"][:, :n], context), axis=1)
        for i in range(self.config.depth):
            root = f"{_HEAD}blocks.{i}."
            q, k, v = mx.split(self.linear(self.norm(x, root + "norm1"), root + "attn.qkv"), 3, axis=-1)
            x = x + self.linear(attention(q, k, v, self.config.num_heads), root + "attn.proj")
            y = self.linear(self.norm(x, root + "norm2"), root + "mlp.fc1")
            x = x + self.linear(nn.gelu_approx(y), root + "mlp.fc2")
        return self.domain_linear(self.norm(x[:, :n], _HEAD + "norm"), _HEAD + "action_decoder", domain)


class XVLAMLX(MLXChunkIO):
    kind = "xvla"

    def __init__(self, path, task, *, dtype=mx.float16, **io):
        # Register X-VLA's saved processor steps before loading the pipelines.
        from lerobot.policies.xvla.modeling_xvla import (
            _is_vendored_florence_state_dict,
            _remap_vendored_florence_state_dict,
        )

        import_module("lerobot.policies.xvla.processor_xvla")
        super().__init__(path, task, **io)
        conf = self.config
        florence = conf.get_florence_config()
        if (conf.action_mode.lower() != "auto" or not conf.use_proprio or conf.max_action_dim < self.width
                or conf.num_denoising_steps < 1 or conf.dim_time < 2
                or florence.vision_config.activation_function != "gelu"
                or florence.text_config.activation_function != "gelu"):
            raise ValueError("MLX X-VLA requires Florence-2 with GELU and padded joint actions")
        weights = mx.load(str(path / "model.safetensors"))
        if _is_vendored_florence_state_dict(weights):
            projection = weights.pop("model.vlm.image_projection", None)
            weights = _remap_vendored_florence_state_dict(weights)
            if projection is not None:
                weights["model." + _PROJECTOR + "image_projection.weight"] = mx.contiguous(projection.T)
        try:
            net = XVLANetwork(weights, conf, florence, dtype)
        except KeyError as exc:
            raise ValueError(f"Missing X-VLA weight: {exc}") from exc
        self.vision, self.condition, self.predict = net.vision, net.condition, net.predict
        self.dtype = dtype
        self.steps = conf.num_denoising_steps
        torch = self.torch
        time = torch.arange(self.steps, 0, -1, dtype=torch.float32) / self.steps
        half = conf.dim_time // 2
        angles = time[:, None] * torch.exp(-math.log(100) * torch.arange(half, dtype=torch.float32) / half)
        embeddings = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if conf.dim_time % 2:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1))
        self.times = mx.array(time.numpy()).astype(dtype)
        self.time_embeddings = mx.array(embeddings.numpy()).astype(dtype)

    def enable_compile(self):
        self.vision, self.condition, self.predict = map(mx.compile, (self.vision, self.condition, self.predict))

    def prepare(self, imgs, q):
        from lerobot.policies.common.vla_utils import pad_vector, resize_with_pad
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_STATE

        batch = self.prepare_batch(imgs, q)
        images = [batch[k] for k in self.config.image_features if k in batch]
        if self.config.resize_imgs_with_padding is not None:
            images = [resize_with_pad(x, *self.config.resize_imgs_with_padding, pad_value=0.0) for x in images]
        images = self.torch.cat(images).permute(0, 2, 3, 1)
        domain = batch.get(self.config.domain_feature_key, batch.get("domain_id", 0))
        domain = int(self.torch.as_tensor(domain).reshape(-1)[0])
        if not 0 <= domain < self.config.num_domains:
            raise ValueError(f"X-VLA domain {domain} is outside the checkpoint's domain table")
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim, truncate=True)
        return (mx.array(images.float().numpy()).astype(self.dtype), mx.array(batch[OBS_LANGUAGE_TOKENS].numpy()),
                mx.array(state.float().numpy()).astype(self.dtype), domain)

    def __call__(self, imgs, q, *, noise=None, stage=None):
        with self.torch.inference_mode():
            images, tokens, state, domain = stage_call(stage, "prepare", self.prepare, imgs, q)
            vision = stage_call(stage, "vision", self.vision, images)
            context = stage_call(stage, "condition", self.condition, vision, tokens, domain)
            noise = self.starting_noise(noise).astype(self.dtype)
            x = mx.zeros_like(noise)
            for i in range(self.steps):
                t = self.times[i]
                x = stage_call(stage, f"denoise_{i:02d}", self.predict,
                               noise * t + x * (1 - t), state, self.time_embeddings[i], context, domain)
            return stage_call(stage, "readback_unnormalise", self.finish, x)
