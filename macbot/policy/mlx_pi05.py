"""MLX inference for stock LeRobot PI05 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .models import PI05, _to_obs

_VLM = "paligemma_with_expert.paligemma.model."
_EXPERT = "paligemma_with_expert.gemma_expert.model."
_VISION = _VLM + "vision_tower."
_PREFIX = _VLM + "language_model."


@dataclass(frozen=True)
class Geometry:
    vision_depth: int = 27
    vision_heads: int = 16
    patch: int = 14
    text_depth: int = 18
    heads: int = 8
    kv_heads: int = 1
    head_dim: int = 256
    rope_base: float = 10000.0
    vision_eps: float = 1e-6
    rms_eps: float = 1e-6


STOCK_GEOMETRY = Geometry()


class PI05Network:
    """Inference graph with immutable prefix K/V and FP32 Euler accumulation."""

    def __init__(self, weights, geometry=STOCK_GEOMETRY, *, dtype=mx.bfloat16, bits=None, quantize="prefix",
                 fused_attention=True, fused_rope=False, pack_linears=False, gelu_fp32=False):
        self.g = geometry
        self.dtype = dtype
        self.fused_attention = fused_attention
        self.fused_rope = fused_rope
        self.gelu_fp32 = gelu_fp32
        self.w = {k.removeprefix("model."): v for k, v in weights.items()}
        embed = _PREFIX + "embed_tokens.weight"
        head = "paligemma_with_expert.paligemma.lm_head.weight"
        if head in self.w:  # LeRobot maps the tied LM head onto the input embedding on load.
            self.w[embed] = self.w.pop(head)
        self.w.pop("paligemma_with_expert.gemma_expert.lm_head.weight", None)
        for direction in ("in", "out"):
            for suffix in ("weight", "bias"):
                old = f"action_time_mlp_{direction}.{suffix}"
                if old in self.w:
                    self.w[f"time_mlp_{direction}.{suffix}"] = self.w.pop(old)
        self.validate_weights()
        self.w = {k: v.astype(dtype) for k, v in self.w.items()}
        patch = _VISION + "embeddings.patch_embedding.weight"
        self.w[patch] = mx.contiguous(self.w[patch].transpose(0, 2, 3, 1))
        self.qw = {}
        if bits is not None:
            if bits not in (4, 8) or quantize not in ("prefix", "expert", "both"):
                raise ValueError("Expected 4 or 8 bits and prefix, expert or both")
            for key in list(self.w):
                selected = (
                    (quantize in ("prefix", "both") and key.startswith(_PREFIX + "layers."))
                    or (quantize in ("expert", "both") and key.startswith(_EXPERT + "layers."))
                )
                if selected and key.endswith(".weight") and (".self_attn." in key or ".mlp." in key):
                    weight = self.w.pop(key)
                    self.qw[key] = (*mx.quantize(weight, group_size=64, bits=bits), bits)
        self.packed = {}
        if pack_linears:
            for i in range(self.g.vision_depth):
                root = f"{_VISION}encoder.layers.{i}.self_attn."
                self.pack(tuple(root + name for name in ("q_proj", "k_proj", "v_proj")))
            for stem in (_PREFIX, _EXPERT):
                for i in range(self.g.text_depth):
                    if stem == _PREFIX and i == self.g.text_depth - 1:
                        continue
                    root = f"{stem}layers.{i}."
                    self.pack(tuple(root + "self_attn." + name for name in ("q_proj", "k_proj", "v_proj")))
                    self.pack(tuple(root + "mlp." + name for name in ("gate_proj", "up_proj")))
        mx.eval(self.w, self.qw, self.packed)

    def pack(self, names):
        if any(name + ".weight" not in self.w for name in names):
            return
        weights = [self.w.pop(name + ".weight") for name in names]
        biases = [self.w.pop(name + ".bias", None) for name in names]
        cuts = tuple(np.cumsum([w.shape[0] for w in weights[:-1]]).tolist())
        self.packed[names] = (mx.concatenate(weights, axis=0), mx.concatenate(biases) if biases[0] is not None else None, cuts)

    def project(self, x, names):
        kernels = self.packed.get(names)
        if kernels is None:
            return tuple(self.linear(x, name) for name in names)
        weight, bias, cuts = kernels
        y = x.astype(self.dtype) @ weight.T
        return tuple(mx.split(y if bias is None else y + bias, cuts, axis=-1))

    def gelu(self, x):
        return nn.gelu_approx(x.astype(mx.float32)).astype(x.dtype) if self.gelu_fp32 else nn.gelu_approx(x)

    def validate_weights(self):
        g = self.g
        shapes = {}

        def add(name, shape, bias=False):
            shapes[name + ".weight"] = shape
            if bias:
                shapes[name + ".bias"] = (shape[0],)

        def width(key, axis=0):
            if key not in self.w or self.w[key].ndim <= axis:
                raise ValueError(f"Missing or invalid PI05 weight: {key}")
            return self.w[key].shape[axis]

        v = width(_VISION + "embeddings.patch_embedding.weight")
        p = width(_PREFIX + "embed_tokens.weight", 1)
        a = width("action_in_proj.weight")
        d = width("action_in_proj.weight", 1)
        add(_VISION + "embeddings.patch_embedding", (v, 3, g.patch, g.patch), True)
        add(_VISION + "embeddings.position_embedding", (width(_VISION + "embeddings.position_embedding.weight"), v))
        add(_VISION + "post_layernorm", (v,), True)
        add(_VLM + "multi_modal_projector.linear", (p, v), True)
        add(_PREFIX + "embed_tokens", (width(_PREFIX + "embed_tokens.weight"), p))
        for i in range(g.vision_depth):
            root = f"{_VISION}encoder.layers.{i}."
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                add(root + "self_attn." + name, (v, v), True)
            for name in ("layer_norm1", "layer_norm2"):
                add(root + name, (v,), True)
            h = width(root + "mlp.fc1.weight")
            add(root + "mlp.fc1", (h, v), True)
            add(root + "mlp.fc2", (v, h), True)
        for stem, dim, adaptive in ((_PREFIX, p, False), (_EXPERT, a, True)):
            norms = [stem + "norm"]
            for i in range(g.text_depth):
                root = f"{stem}layers.{i}."
                norms.extend(root + name for name in ("input_layernorm", "post_attention_layernorm"))
                for name, shape in (
                    ("q_proj", (g.heads * g.head_dim, dim)), ("k_proj", (g.kv_heads * g.head_dim, dim)),
                    ("v_proj", (g.kv_heads * g.head_dim, dim)), ("o_proj", (dim, g.heads * g.head_dim)),
                ):
                    add(root + "self_attn." + name, shape)
                h = width(root + "mlp.up_proj.weight")
                add(root + "mlp.up_proj", (h, dim))
                add(root + "mlp.gate_proj", (h, dim))
                add(root + "mlp.down_proj", (dim, h))
            for name in norms:
                if adaptive:
                    add(name + ".dense", (3 * dim, dim), True)
                else:
                    add(name, (dim,))
        for name, shape in (
            ("action_in_proj", (a, d)), ("action_out_proj", (d, a)),
            ("time_mlp_in", (a, a)), ("time_mlp_out", (a, a)),
        ):
            add(name, shape, True)
        missing = shapes.keys() - self.w.keys()
        unexpected = self.w.keys() - shapes.keys()
        malformed = [k for k in shapes.keys() & self.w.keys() if tuple(self.w[k].shape) != shapes[k]]
        if missing or unexpected or malformed:
            raise ValueError(f"Unsupported PI05 weights: missing={sorted(missing)[:4]}, "
                             f"unexpected={sorted(unexpected)[:4]}, shape mismatch={sorted(malformed)[:4]}")

    def linear(self, x, name):
        key = name + ".weight"
        x = x.astype(self.dtype)
        if key in self.qw:
            w, scales, offsets, bits = self.qw[key]
            y = mx.quantized_matmul(x, w, scales, offsets, transpose=True, group_size=64, bits=bits)
        else:
            y = x @ self.w[key].T
        bias = self.w.get(name + ".bias")
        return y if bias is None else y + bias

    @staticmethod
    def heads(x, num_heads):
        return x.reshape(*x.shape[:2], num_heads, -1).transpose(0, 2, 1, 3)

    def attention(self, q, k, v, mask=None):
        scale = q.shape[-1] ** -0.5
        if self.fused_attention:
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        rep = q.shape[1] // k.shape[1]
        if rep > 1:
            k, v = mx.repeat(k, rep, axis=1), mx.repeat(v, rep, axis=1)
        logits = (q @ k.swapaxes(-1, -2)) * scale
        if mask is not None:
            logits = mx.where(mask, logits.astype(mx.float32), -2.3819763e38)
        return mx.softmax(logits.astype(mx.float32), axis=-1).astype(q.dtype) @ v

    def vision(self, images):
        """NHWC image batch, one row per camera."""
        g, w = self.g, self.w
        emb = _VISION + "embeddings."
        x = mx.conv2d(images.astype(self.dtype), w[emb + "patch_embedding.weight"], stride=g.patch)
        x = x.flatten(1, 2) + w[emb + "patch_embedding.bias"] + w[emb + "position_embedding.weight"]

        def norm(z, name):
            return mx.fast.layer_norm(z, w[name + ".weight"], w[name + ".bias"], g.vision_eps)

        for i in range(g.vision_depth):
            root = f"{_VISION}encoder.layers.{i}."
            h = norm(x, root + "layer_norm1")
            names = tuple(root + "self_attn." + name for name in ("q_proj", "k_proj", "v_proj"))
            q, k, v = (self.heads(z, g.vision_heads) for z in self.project(h, names))
            y = self.attention(q, k, v).transpose(0, 2, 1, 3).flatten(2)
            x = x + self.linear(y, root + "self_attn.out_proj")
            h = norm(x, root + "layer_norm2")
            x = x + self.linear(self.gelu(self.linear(h, root + "mlp.fc1")), root + "mlp.fc2")
        return self.linear(norm(x, _VISION + "post_layernorm"), _VLM + "multi_modal_projector.linear")

    def pos_embed(self, positions):
        if self.fused_rope:
            # Only valid tokens contribute. Padding positions are masked in every layer.
            return positions[:, 0]
        half = self.g.head_dim // 2
        # macbot's reference casts the model's RoPE buffer to BF16 when placing the
        # checkpoint on MPS, then converts it back to FP32 for the angle computation.
        inv = (1.0 / (self.g.rope_base ** (mx.arange(half, dtype=mx.float32) / half)))
        if self.dtype != mx.float32:
            inv = inv.astype(mx.bfloat16).astype(mx.float32)
        angles = positions[..., None].astype(mx.float32) * inv
        angles = mx.concatenate((angles, angles), axis=-1)[:, None]
        return mx.cos(angles).astype(self.dtype), mx.sin(angles).astype(self.dtype)

    def rotary(self, x, pos):
        if self.fused_rope:
            return mx.fast.rope(x, self.g.head_dim, traditional=False, base=self.g.rope_base, scale=1.0, offset=pos)
        lo, hi = mx.split(x, 2, axis=-1)
        cos, sin = pos
        return x * cos + mx.concatenate((-hi, lo), axis=-1) * sin

    def norm(self, x, name, cond, modulation=None):
        w = self.w
        if cond is None:
            weight = 1.0 + w[name + ".weight"].astype(mx.float32)
            return mx.fast.rms_norm(x.astype(mx.float32), weight, self.g.rms_eps).astype(x.dtype), None
        y = mx.fast.rms_norm(x.astype(mx.float32), None, self.g.rms_eps)
        if modulation is None:
            modulation = self.linear(cond, name + ".dense")
        scale, shift, gate = mx.split(modulation[:, None, :], 3, axis=-1)
        y = y * (1 + scale.astype(mx.float32)) + shift.astype(mx.float32)
        return y.astype(x.dtype), gate.astype(x.dtype)

    def text_layer(self, x, root, positions, mask, previous=None, cond=None, *, kv_only=False, modulation=(None, None)):
        g = self.g
        y, gate = self.norm(x, root + "input_layernorm", cond, modulation[0])
        names = tuple(root + "self_attn." + name for name in ("q_proj", "k_proj", "v_proj"))
        if kv_only:
            k, v = self.project(y, names[1:])
            return x, (self.rotary(self.heads(k, g.kv_heads), positions), self.heads(v, g.kv_heads))
        q, k, v = self.project(y, names)
        q = self.rotary(self.heads(q, g.heads), positions)
        k = self.rotary(self.heads(k, g.kv_heads), positions)
        v = self.heads(v, g.kv_heads)
        if previous is not None:
            k, v = (mx.concatenate((old, new), axis=2) for old, new in zip(previous, (k, v), strict=True))
        z = self.attention(q, k, v, mask).transpose(0, 2, 1, 3).flatten(2)
        z = self.linear(z, root + "self_attn.o_proj")
        x = x + (z if gate is None else z * gate)
        y, gate = self.norm(x, root + "post_attention_layernorm", cond, modulation[1])
        g_out, up = self.project(y, (root + "mlp.gate_proj", root + "mlp.up_proj"))
        z = self.gelu(g_out) * up
        z = self.linear(z, root + "mlp.down_proj")
        return x + (z if gate is None else z * gate), (k, v)

    def embed_text(self, tokens):
        table = self.w[_PREFIX + "embed_tokens.weight"]
        return table[tokens] * mx.array(table.shape[1] ** 0.5, dtype=self.dtype)

    def prefix(self, vision, tokens, token_mask, full_mask=True):
        text = self.embed_text(tokens)
        x = mx.concatenate((vision.flatten(0, 1)[None], text), axis=1)
        valid = mx.concatenate((mx.ones((1, vision.shape[0] * vision.shape[1]), dtype=mx.bool_), token_mask), axis=1)
        positions = self.pos_embed(mx.cumsum(valid, axis=1) - 1)
        attn = valid[:, None, None, :] & valid[:, None, :, None] if full_mask else None
        cache = []
        for i in range(self.g.text_depth):
            x, kv = self.text_layer(x, f"{_PREFIX}layers.{i}.", positions, attn, kv_only=i == self.g.text_depth - 1)
            cache.append(kv)
        return valid, tuple(cache)

    def expert_inputs(self, valid, horizon, masked=True):
        positions = self.pos_embed(mx.sum(valid, axis=1, keepdims=True) + mx.arange(horizon)[None])
        if not masked:
            return positions, None
        keep = mx.concatenate((valid, mx.ones((valid.shape[0], horizon), dtype=mx.bool_)), axis=1)
        return positions, keep[:, None, None, :]

    def time_condition(self, time, min_period, max_period):
        width = self.w["time_mlp_in.weight"].shape[1]
        period = min_period * (max_period / min_period) ** mx.linspace(0.0, 1.0, width // 2)
        phase = time[..., None] * (2 * np.pi / period)
        embedding = mx.concatenate((mx.sin(phase), mx.cos(phase)), axis=-1)
        return nn.silu(self.linear(nn.silu(self.linear(embedding, "time_mlp_in")), "time_mlp_out"))

    def adarms(self, cond):
        names = ["input_layernorm", "post_attention_layernorm"]
        layers = [self.linear(cond, f"{_EXPERT}layers.{i}.{name}.dense")
                  for i in range(self.g.text_depth) for name in names]
        return (*layers, self.linear(cond, _EXPERT + "norm.dense"))

    def velocity(self, actions, cond, cache, positions, mask, modulation=None):
        x = self.linear(actions, "action_in_proj")
        for i, kv in enumerate(cache):
            layer_mod = (None, None) if modulation is None else modulation[2 * i:2 * i + 2]
            x, _ = self.text_layer(x, f"{_EXPERT}layers.{i}.", positions, mask, previous=kv, cond=cond,
                                   modulation=layer_mod)
        final_mod = None if modulation is None else modulation[-1]
        return self.linear(self.norm(x, _EXPERT + "norm", cond, final_mod)[0], "action_out_proj")


class PI05MLX:
    kind = "pi05"
    supports_rtc = False

    def __init__(self, path, task, steps=None, *, bits=None, quantize="prefix", trim=False, fused_attention=True,
                 fused_rope=False, dtype=mx.bfloat16, precompute_adarms=False, pack_linears=False, gelu_fp32=False,
                 pool_images=False, time_grid="uniform"):
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config

        config = PreTrainedConfig.from_pretrained(path)
        config.device = "cpu"
        if not isinstance(config, PI05Config) or config.type != "pi05":
            raise ValueError("The MLX backend only supports stock LeRobot PI05 checkpoints")
        if config.use_visual_memory or config.use_proprioceptive_memory or getattr(config.rtc_config, "enabled", False):
            raise ValueError("The MLX backend does not support memory or checkpoint-configured RTC")
        config.validate_features()
        self.config = config
        self.torch, self.task = torch, task
        self.rename, self.shapes = PI05.RENAME, None
        self.H = config.n_action_steps
        self.steps = steps if steps is not None else config.num_inference_steps
        if self.steps < 1:
            raise ValueError("At least one denoising step is required")
        self.trim = trim
        self.pool_images = pool_images
        self.pre, self.post = make_pre_post_processors(
            config, pretrained_path=str(path), preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.network = PI05Network(mx.load(str(path / "model.safetensors")), dtype=dtype, bits=bits, quantize=quantize,
                                   fused_attention=fused_attention, fused_rope=fused_rope, pack_linears=pack_linears,
                                   gelu_fp32=gelu_fp32)
        self.vision, self.prefix, self.velocity = self.network.vision, self.network.prefix, self.network.velocity
        if time_grid not in ("uniform", "quadratic"):
            raise ValueError("Expected a uniform or quadratic MLX time grid")
        times = [1.0 - i / self.steps for i in range(self.steps)] + [0.0]
        self.deltas = None
        if time_grid == "quadratic":
            times = [t ** 2 for t in times]
            self.deltas = [end - t for t, end in zip(times[:-1], times[1:], strict=True)]
        schedule = mx.array(times[:-1], dtype=mx.float32)
        self.cond = self.network.time_condition(schedule, config.min_period, config.max_period)
        self.modulation = self.network.adarms(self.cond) if precompute_adarms else None
        mx.eval(self.cond, self.modulation)

    def enable_compile(self):
        self.vision, self.prefix, self.velocity = map(mx.compile, (self.vision, self.prefix, self.velocity))

    def velocity_args(self, i):
        return self.cond[i:i + 1], None if self.modulation is None else tuple(x[i:i + 1] for x in self.modulation)

    def pool(self, vision):
        if not self.pool_images:
            return vision
        views, tokens, width = vision.shape
        side = int(tokens ** 0.5)
        if side * side != tokens or side % 2:
            raise ValueError("Image pooling requires an even square grid")
        return mx.mean(vision.reshape(views, side // 2, 2, side // 2, 2, width), axis=(2, 4)).flatten(1, 2)

    def prepare(self, imgs, q):
        from lerobot.policies.common.vla_utils import resize_with_pad_torch
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        torch = self.torch
        obs = self.pre(_to_obs(torch, imgs, q, self.task, self.rename))
        res = tuple(self.config.image_resolution)
        live = [obs[k] for k in self.config.image_features if k in obs]
        if not live:
            raise ValueError("No matching camera keys for PI05")
        images = []
        for image in live:
            image = image.permute(0, 2, 3, 1)
            if tuple(image.shape[1:3]) != res:
                image = resize_with_pad_torch(image, *res)
            images.append((image * 2 - 1).numpy())
        tokens, mask = obs[OBS_LANGUAGE_TOKENS].numpy(), obs[OBS_LANGUAGE_ATTENTION_MASK].numpy()
        if self.trim:
            length = int(mask.sum())
            if mask[0, :length].all() and not mask[0, length:].any():
                tokens, mask = tokens[:, :length], mask[:, :length]
        return mx.array(np.concatenate(images, axis=0)), mx.array(tokens, dtype=mx.int32), mx.array(mask), bool(mask.all())

    def stages(self, images, tokens, mask, all_valid, noise, *, stage=None):
        def run(name, fn, *args):
            return fn(*args) if stage is None else stage(name, fn, *args)

        features = run("vision", self.vision, images)
        valid, cache = run("prefix_and_kv", self.prefix, self.pool(features), tokens, mask, not all_valid)
        positions, keep = self.network.expert_inputs(valid, noise.shape[1], masked=not all_valid)
        x = noise
        for i in range(self.steps):
            cond, modulation = self.velocity_args(i)
            velocity = run(f"denoise_{i:02d}", self.velocity, x, cond, cache, positions, keep, modulation)
            v = velocity.astype(mx.float32)
            x = x - v / self.steps if self.deltas is None else x + self.deltas[i] * v
        return x

    def __call__(self, imgs, q, leftover=None, delay=0, horizon=None, return_raw=False, *, noise=None, stage=None):
        if leftover is not None and len(leftover) > delay > 0:
            raise ValueError("Guided RTC requires the PyTorch backend")
        with self.torch.inference_mode():
            obs = self.prepare(imgs, q) if stage is None else stage("prepare", self.prepare, imgs, q)
            if noise is None:
                noise = mx.random.normal((1, self.config.chunk_size, self.config.max_action_dim))
            else:
                noise = mx.array(noise, dtype=mx.float32)
            x = self.stages(*obs, noise, stage=stage)
            dim = self.config.output_features["action"].shape[0]

            def finish():
                raw = np.array(x[..., :dim], dtype=np.float32)
                action = self.post(self.torch.from_numpy(raw)).numpy()
                if not np.isfinite(action).all():
                    raise RuntimeError("MLX PI05 produced nonfinite actions")
                return (action[0], raw[0]) if return_raw else action[0]

            return finish() if stage is None else stage("readback_unnormalise", finish)
