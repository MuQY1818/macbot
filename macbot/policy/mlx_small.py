"""MLX inference for the small LeRobot policies: Diffusion, VQ-BeT and TD-MPC."""
from __future__ import annotations

import json
import math
from collections import deque

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .generic import STATE, _features, _processor_overrides
from .mlx_chunk import DenseWeights


def _mish(x):
    return nn.mish(x)


def _as_mlx(value):
    if hasattr(value, "detach"):
        return mx.array(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return mx.array(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_as_mlx(item) for item in value)
    return value


def _elu(x):
    return nn.elu(x)


def _group_norm(x, weight, bias, groups, eps=1e-5):
    batch, length, channels = x.shape
    if channels % groups:
        raise ValueError(f"GroupNorm channels {channels} are not divisible by {groups}")
    grouped = x.reshape(batch, length, groups, channels // groups)
    mean = grouped.mean(axis=(1, 3), keepdims=True)
    var = ((grouped - mean) ** 2).mean(axis=(1, 3), keepdims=True)
    y = ((grouped - mean) * mx.rsqrt(var + eps)).reshape(batch, length, channels)
    return y * weight.reshape(1, 1, -1) + bias.reshape(1, 1, -1)


def _group_norm2d(x, weight, bias, groups, eps=1e-5):
    batch, height, width, channels = x.shape
    if channels % groups:
        raise ValueError(f"GroupNorm channels {channels} are not divisible by {groups}")
    grouped = x.reshape(batch, height, width, groups, channels // groups)
    mean = grouped.mean(axis=(1, 2, 4), keepdims=True)
    var = ((grouped - mean) ** 2).mean(axis=(1, 2, 4), keepdims=True)
    y = ((grouped - mean) * mx.rsqrt(var + eps)).reshape(batch, height, width, channels)
    return y * weight.reshape(1, 1, 1, -1) + bias.reshape(1, 1, 1, -1)


def _linear(weights, x, name):
    return weights.linear(x, name)


def _conv1d(weights, x, name, *, stride=1, padding=0):
    weight = weights.w[name + ".weight"].transpose(0, 2, 1)
    bias = weights.w.get(name + ".bias")
    y = mx.conv1d(x, weight, stride=stride, padding=padding)
    return y if bias is None else y + bias


def _conv_transpose1d(weights, x, name, *, stride=1, padding=0):
    weight = weights.w[name + ".weight"].transpose(1, 2, 0)
    bias = weights.w.get(name + ".bias")
    y = mx.conv_transpose1d(x, weight, stride=stride, padding=padding)
    return y if bias is None else y + bias


def _fold_bn(weights, conv, bn):
    scale = weights.pop(bn + ".weight").astype(mx.float32)
    running_mean = weights.pop(bn + ".running_mean").astype(mx.float32)
    running_var = weights.pop(bn + ".running_var").astype(mx.float32)
    bias = weights.pop(bn + ".bias").astype(mx.float32)
    scale = scale * mx.rsqrt(running_var + 1e-5)
    weights[conv + ".weight"] = weights[conv + ".weight"].astype(mx.float32) * scale[:, None, None, None]
    weights[conv + ".bias"] = bias - running_mean * scale


def _fold_resnet_bn(weights, root):
    if root + "backbone.1.running_mean" in weights:
        _fold_bn(weights, root + "backbone.0", root + "backbone.1")
    for layer in range(1, 5):
        for block in range(2):
            base = f"{root}backbone.{layer + 3}.{block}."
            if base + "bn1.running_mean" in weights:
                _fold_bn(weights, base + "conv1", base + "bn1")
                _fold_bn(weights, base + "conv2", base + "bn2")
            if layer > 1 and block == 0:
                if base + "downsample.1.running_mean" in weights:
                    _fold_bn(weights, base + "downsample.0", base + "downsample.1")


def _resnet18(weights, x, prefix):
    """Run the torchvision ResNet-18 trunk with folded eval-mode BatchNorm."""
    store = weights.w if hasattr(weights, "w") else weights
    group_norm = prefix + "backbone.0.bias" not in store
    x = mx.conv2d(x, store[prefix + "backbone.0.weight"], stride=2, padding=3)
    if prefix + "backbone.0.bias" in store:
        x = x + store[prefix + "backbone.0.bias"]
    if group_norm:
        x = _group_norm2d(x, store[prefix + "backbone.1.weight"], store[prefix + "backbone.1.bias"], 4)
    x = nn.relu(x)
    x = nn.MaxPool2d(3, stride=2, padding=1)(x)
    for layer in range(1, 5):
        for block in range(2):
            base = f"{prefix}backbone.{layer + 3}.{block}."
            stride = 2 if layer > 1 and block == 0 else 1
            skip = x
            if stride == 2:
                skip = mx.conv2d(x, store[base + "downsample.0.weight"], stride=2)
                if base + "downsample.0.bias" in store:
                    skip = skip + store[base + "downsample.0.bias"]
                if group_norm:
                    channels = skip.shape[-1]
                    _groups = max(1, channels // 16)
                    skip = _group_norm2d(skip, store[base + "downsample.1.weight"],
                                         store[base + "downsample.1.bias"], _groups)
            y = mx.conv2d(x, store[base + "conv1.weight"], stride=stride, padding=1)
            if base + "conv1.bias" in store:
                y = y + store[base + "conv1.bias"]
            if group_norm:
                channels = y.shape[-1]
                y = _group_norm2d(y, store[base + "bn1.weight"], store[base + "bn1.bias"], max(1, channels // 16))
            y = nn.relu(y)
            y = mx.conv2d(y, store[base + "conv2.weight"], padding=1)
            if base + "conv2.bias" in store:
                y = y + store[base + "conv2.bias"]
            if group_norm:
                channels = y.shape[-1]
                y = _group_norm2d(y, store[base + "bn2.weight"], store[base + "bn2.bias"], max(1, channels // 16))
            x = nn.relu(y + skip)
    return x


def _spatial_softmax(weights, x, prefix):
    store = weights.w if hasattr(weights, "w") else weights
    pool_prefix = prefix + "pool.nets"
    if pool_prefix + ".weight" in store:
        x = mx.conv2d(x, store[pool_prefix + ".weight"], padding=0)
        x = x + store[pool_prefix + ".bias"]
    batch, height, width, channels = x.shape
    logits = x.transpose(0, 3, 1, 2).reshape(batch, channels, height * width)
    probs = mx.softmax(logits, axis=-1)
    pos = store[prefix + "pool.pos_grid"]
    return (probs @ pos).reshape(batch, -1)


def _image_features(weights, images, prefix, crop_shape=None):
    if crop_shape is not None:
        height, width = images.shape[1:3]
        crop_h, crop_w = crop_shape
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        images = images[:, top:top + crop_h, left:left + crop_w]
    x = _resnet18(weights, images, prefix)
    x = _spatial_softmax(weights, x, prefix)
    x = _linear(weights, x, prefix + "out")
    return nn.relu(x)


def _mlp(weights, x, prefix, *, activations=()):
    index = 0
    activation_index = 0
    while prefix + f".{index}.weight" in weights.w:
        x = _linear(weights, x, prefix + f".{index}")
        next_index = index + 2
        if prefix + f".{next_index}.weight" in weights.w and activation_index < len(activations):
            x = activations[activation_index](x)
            activation_index += 1
        index = next_index
    return x


class MLXSmallIO:
    """Saved processors and observation history for non-VLA MLX policies."""

    shapes = None
    supports_rtc = False
    requires_task = False
    stepwise = False

    def __init__(self, path, task, *, cameras, camera_map=None, joint_count=None):
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors

        self.torch = torch
        self.task = task
        metadata = json.loads((path / "config.json").read_text())
        self.width, self.has_state, self.cameras = _features(path, metadata, cameras, camera_map or {}, joint_count)
        config = PreTrainedConfig.from_pretrained(path, cli_overrides=["--device", "cpu"])
        if config.type != self.kind or not self.has_state or not self.cameras:
            raise ValueError(f"MLX {self.kind} requires joint state and RGB cameras")
        config.validate_features()
        if self.requires_task and not task.strip():
            raise ValueError(f"MLX {self.kind} requires task text")
        self.config = config
        self.n_obs_steps = int(getattr(config, "n_obs_steps", 1))
        if self.kind == "diffusion":
            self.H = int(config.n_action_steps)
        elif self.kind == "vqbet":
            self.H = int(config.action_chunk_size)
        else:
            self.H = max(1, int(config.n_action_steps))
        pre, steps = _processor_overrides(path, "pre", "cpu")
        post, _ = _processor_overrides(path, "post", "cpu")
        self.task_keys = {
            step.get("config", {}).get("task_key", "task")
            for step in steps
            if step.get("registry_name") in {"tokenizer_processor", "multi_task_dit_task_tokenizer"}
            or step.get("class", "").endswith(".TokenizerProcessorStep")
        }
        self.pre, self.post = make_pre_post_processors(
            config, pretrained_path=path, preprocessor_overrides=pre, postprocessor_overrides=post,
        )
        self._history = deque(maxlen=self.n_obs_steps)

    def reset(self):
        self.pre.reset()
        self.post.reset()
        self._history.clear()

    def prepare_current(self, imgs, q):
        obs = {key: [self.task] for key in self.task_keys | {"task"}}
        obs[STATE] = self.torch.tensor(q, dtype=self.torch.float32).unsqueeze(0)
        for key, name in self.cameras.items():
            obs[key] = self.torch.from_numpy(imgs[name]).permute(2, 0, 1).float().div_(255).unsqueeze(0)
        return self.pre(obs)

    def prepare_history(self, imgs, q):
        current = self.prepare_current(imgs, q)
        self._history.append(current)
        while len(self._history) < self.n_obs_steps:
            self._history.appendleft(current)
        keys = set(current)
        return {key: self.torch.cat([item[key].unsqueeze(1) for item in self._history], dim=1)
                for key in keys if hasattr(self._history[0].get(key), "shape")}

    def finish(self, x):
        raw = np.asarray(x, dtype=np.float32)
        if raw.ndim == 2:
            raw = raw[None]
        raw = raw[..., :self.width]
        action = self.post(self.torch.from_numpy(raw)).float().cpu().numpy()
        action = action[0] if action.ndim == 3 else action
        if action.shape != (self.H, self.width) or not np.isfinite(action).all():
            raise ValueError(f"MLX {self.kind} returned an invalid action chunk with shape {action.shape}")
        return action

    def enable_compile(self):
        self._compiled = mx.compile(self._call_impl)

    def _mlx_batch(self, batch):
        result = {}
        for key, value in batch.items():
            if key in getattr(self.config, "image_features", {}):
                if hasattr(value, "detach"):
                    array = np.ascontiguousarray(value.detach().cpu().numpy().transpose(0, 1, 3, 4, 2))
                    result[key] = mx.array(array)
                else:
                    result[key] = value if value.shape[-1] == 3 else value.transpose(0, 1, 3, 4, 2)
            else:
                result[key] = _as_mlx(value)
        return result

    def __call__(self, imgs, q, *, stage=None, **kwargs):
        batch = self.prepare_history(imgs, q)
        return self._call(batch, stage=stage, **kwargs)

    def _call(self, batch, *, stage=None, **kwargs):
        return self.finish(self._call_impl(batch, **kwargs))

    def _call_impl(self, batch, **kwargs):
        raise NotImplementedError


class DiffusionNetwork(DenseWeights):
    def __init__(self, weights, config):
        weights = {k.removeprefix("diffusion."): v for k, v in weights.items()
                   if not k.endswith("num_batches_tracked")}
        prefixes = sorted({k.split(".backbone", 1)[0] + "." for k in weights if ".backbone." in k})
        for prefix in prefixes:
            _fold_resnet_bn(weights, prefix)
        super().__init__(weights, mx.float32)
        self.config = config
        self.rgb_prefixes = prefixes
        mx.eval(self.w)

    def encode_images(self, images):
        features = []
        for index, prefix in enumerate(self.rgb_prefixes):
            features.append(_image_features(self, images[:, :, index], prefix))
        return mx.stack(features, axis=2)

    def condition(self, batch):
        state = _as_mlx(batch[STATE])
        image_keys = list(self.config.image_features)
        image_features = []
        batch_size, steps = state.shape[:2]
        if len(self.rgb_prefixes) == 1:
            prefixes = self.rgb_prefixes * len(image_keys)
        elif len(self.rgb_prefixes) == len(image_keys):
            prefixes = self.rgb_prefixes
        else:
            raise ValueError("Diffusion encoder count does not match the checkpoint image features")
        for key, prefix in zip(image_keys, prefixes, strict=True):
            value = batch[key]
            if hasattr(value, "detach"):
                source = mx.array(np.ascontiguousarray(value.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)))
            else:
                source = value if value.shape[-1] == 3 else value.transpose(0, 1, 3, 4, 2)
            view = source.reshape(batch_size * steps, *source.shape[2:])
            feature = _image_features(self, view, prefix, self.config.crop_shape)
            image_features.append(feature.reshape(batch_size, steps, -1))
        image_features = mx.stack(image_features, axis=2)
        return mx.concatenate((state, image_features.reshape(state.shape[0], state.shape[1], -1)), axis=-1).reshape(
            state.shape[0], -1
        )

    def _sinusoidal(self, timestep):
        dim = self.config.diffusion_step_embed_dim
        half = dim // 2
        scale = math.log(10000) / (half - 1)
        freqs = mx.exp(-scale * mx.arange(half, dtype=mx.float32))
        emb = timestep.astype(mx.float32)[:, None] * freqs[None]
        return mx.concatenate((mx.sin(emb), mx.cos(emb)), axis=-1)

    def _residual(self, x, cond, root, in_channels, out_channels):
        y = _conv1d(self, x, root + "conv1.block.0", padding=self.config.kernel_size // 2)
        y = _group_norm(y, self.w[root + "conv1.block.1.weight"], self.w[root + "conv1.block.1.bias"],
                        self.config.n_groups)
        y = _mish(y)
        film = _linear(self, _mish(cond), root + "cond_encoder.1")
        if self.config.use_film_scale_modulation:
            scale, bias = mx.split(film, 2, axis=-1)
            y = scale[:, None] * y + bias[:, None]
        else:
            y = y + film[:, None]
        y = _conv1d(self, y, root + "conv2.block.0", padding=self.config.kernel_size // 2)
        y = _group_norm(y, self.w[root + "conv2.block.1.weight"], self.w[root + "conv2.block.1.bias"],
                        self.config.n_groups)
        y = _mish(y)
        if in_channels != out_channels:
            skip = _conv1d(self, x, root + "residual_conv", padding=0)
        else:
            skip = x
        return y + skip

    def unet(self, x, timestep, global_cond):
        x = mx.array(x)
        timestep_embed = self._sinusoidal(timestep)
        timestep_embed = _mish(_linear(self, timestep_embed, "unet.diffusion_step_encoder.1"))
        timestep_embed = _linear(self, timestep_embed, "unet.diffusion_step_encoder.3")
        cond = mx.concatenate((timestep_embed, global_cond), axis=-1)
        dims = tuple(self.config.down_dims)
        skips = []
        for index, (dim_in, dim_out) in enumerate(zip((self.config.action_feature.shape[0],) + dims[:-1], dims, strict=True)):
            root = f"unet.down_modules.{index}."
            x = self._residual(x, cond, root + "0.", dim_in, dim_out)
            x = self._residual(x, cond, root + "1.", dim_out, dim_out)
            skips.append(x)
            if index < len(dims) - 1:
                x = _conv1d(self, x, root + "2", stride=2, padding=1)
        for index in range(2):
            root = f"unet.mid_modules.{index}."
            x = self._residual(x, cond, root, dims[-1], dims[-1])
        reversed_pairs = list(reversed(list(zip(dims[:-1], dims[1:], strict=True))))
        for index, (dim_out, dim_in) in enumerate(reversed_pairs):
            root = f"unet.up_modules.{index}."
            skip = skips.pop()
            x = mx.concatenate((x, skip), axis=-1)
            x = self._residual(x, cond, root + "0.", dim_in * 2, dim_out)
            x = self._residual(x, cond, root + "1.", dim_out, dim_out)
            x = _conv_transpose1d(self, x, root + "2", stride=2, padding=1)
        x = _conv1d(self, x, "unet.final_conv.0.block.0", padding=self.config.kernel_size // 2)
        x = _group_norm(x, self.w["unet.final_conv.0.block.1.weight"],
                        self.w["unet.final_conv.0.block.1.bias"], self.config.n_groups)
        x = _mish(x)
        return _conv1d(self, x, "unet.final_conv.1")


def _beta_schedule(config):
    if config.beta_schedule == "linear":
        return np.linspace(config.beta_start, config.beta_end, config.num_train_timesteps, dtype=np.float32)
    if config.beta_schedule == "scaled_linear":
        return np.linspace(config.beta_start**0.5, config.beta_end**0.5,
                           config.num_train_timesteps, dtype=np.float32) ** 2
    if config.beta_schedule != "squaredcos_cap_v2":
        raise ValueError(f"Unsupported diffusion beta schedule {config.beta_schedule!r}")
    values = []

    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    for index in range(config.num_train_timesteps):
        t1, t2 = index / config.num_train_timesteps, (index + 1) / config.num_train_timesteps
        values.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
    return np.asarray(values, dtype=np.float32)


class DiffusionMLX(MLXSmallIO):
    kind = "diffusion"

    def __init__(self, path, task, **io):
        super().__init__(path, task, **io)
        self.network = DiffusionNetwork(mx.load(str(path / "model.safetensors")), self.config)
        self.betas = mx.array(_beta_schedule(self.config))
        self.alphas_cumprod = mx.cumprod(1.0 - self.betas)
        self.steps = int(self.config.num_inference_steps or self.config.num_train_timesteps)
        self.timesteps = self._set_timesteps(self.steps)
        self._compiled = None

    def _set_timesteps(self, steps):
        if steps > self.config.num_train_timesteps:
            raise ValueError("num_inference_steps cannot exceed num_train_timesteps")
        if self.config.noise_scheduler_type == "DDIM":
            ratio = self.config.num_train_timesteps // steps
            return np.arange(0, steps, dtype=np.int32)[::-1] * ratio
        ratio = self.config.num_train_timesteps // steps
        return np.arange(0, steps, dtype=np.int32)[::-1] * ratio

    def _step(self, model_output, timestep, sample, noise):
        t = int(timestep)
        prev = t - self.config.num_train_timesteps // self.steps
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[prev] if prev >= 0 else mx.array(1.0)
        beta_t = 1 - alpha_t
        if self.config.prediction_type == "epsilon":
            original = (sample - mx.sqrt(beta_t) * model_output) / mx.sqrt(alpha_t)
            epsilon = model_output
        elif self.config.prediction_type == "sample":
            original = model_output
            epsilon = (sample - mx.sqrt(alpha_t) * original) / mx.sqrt(beta_t)
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type!r}")
        if self.config.clip_sample:
            original = mx.clip(original, -self.config.clip_sample_range, self.config.clip_sample_range)
        if self.config.noise_scheduler_type == "DDIM":
            variance = (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
            direction = mx.sqrt(mx.maximum(1 - alpha_prev, 0.0)) * epsilon
            result = mx.sqrt(alpha_prev) * original + direction
            return result
        current_alpha = alpha_t / alpha_prev
        current_beta = 1 - current_alpha
        coeff_original = mx.sqrt(alpha_prev) * current_beta / beta_t
        coeff_sample = mx.sqrt(current_alpha) * (1 - alpha_prev) / beta_t
        result = coeff_original * original + coeff_sample * sample
        if t > 0:
            variance = ((1 - alpha_prev) / (1 - alpha_t)) * current_beta
            result = result + mx.sqrt(mx.maximum(variance, 0.0)) * noise
        return result

    def _call_impl(self, batch, **kwargs):
        global_cond = self.network.condition(batch)
        shape = (1, self.config.horizon, self.config.action_feature.shape[0])
        noise = kwargs.get("noise")
        sample = mx.random.normal(shape) if noise is None else mx.array(noise, dtype=mx.float32)
        step_noise = kwargs.get("step_noise")
        for index, timestep in enumerate(self.timesteps):
            model = self.network.unet(sample, mx.full((1,), int(timestep)), global_cond)
            if step_noise is None:
                draw = mx.random.normal(shape)
            else:
                draw = mx.array(step_noise[index], dtype=mx.float32)
            sample = self._step(model, timestep, sample, draw)
        start = self.n_obs_steps - 1
        return sample[:, start:start + self.H]

    def __call__(self, imgs, q, *, noise=None, step_noise=None, stage=None, **kwargs):
        batch = self.prepare_history(imgs, q)
        runner = self._compiled or self._call_impl
        if self._compiled is not None:
            batch = self._mlx_batch(batch)
            noise = None if noise is None else _as_mlx(noise)
            step_noise = None if step_noise is None else _as_mlx(step_noise)
        return self.finish(runner(batch, noise=noise, step_noise=step_noise))


class VQBeTNetwork(DenseWeights):
    def __init__(self, weights, config):
        weights = {k.removeprefix("vqbet."): v for k, v in weights.items()
                   if not k.endswith(("num_batches_tracked", "discretized", "freeze_codebook", "cluster_size",
                                      "embed_avg", "initted"))}
        _fold_resnet_bn(weights, "rgb_encoder.")
        super().__init__(weights, mx.float32)
        self.config = config
        self.width = config.action_feature.shape[0]
        mx.eval(self.w)

    def gpt(self, x):
        x = _linear(self, x, "policy.transformer.wte")
        positions = mx.arange(x.shape[1], dtype=mx.int32)
        x = x + self.w["policy.transformer.wpe.weight"][positions][None]
        heads = self.config.gpt_n_head
        head_dim = self.config.gpt_hidden_dim // heads
        for layer in range(self.config.gpt_n_layer):
            root = f"policy.transformer.h.{layer}."
            h = mx.fast.layer_norm(x, self.w[root + "ln_1.weight"], self.w[root + "ln_1.bias"], 1e-5)
            qkv = _linear(self, h, root + "attn.c_attn")
            q, k, v = mx.split(qkv, 3, axis=-1)
            q = q.reshape(q.shape[0], q.shape[1], heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(k.shape[0], k.shape[1], heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(v.shape[0], v.shape[1], heads, head_dim).transpose(0, 2, 1, 3)
            mask = mx.triu(mx.full((x.shape[1], x.shape[1]), -1e9), 1)
            attn = mx.softmax((q @ k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim) + mask, axis=-1) @ v
            attn = attn.transpose(0, 2, 1, 3).reshape(x.shape)
            x = x + _linear(self, attn, root + "attn.c_proj")
            h = mx.fast.layer_norm(x, self.w[root + "ln_2.weight"], self.w[root + "ln_2.bias"], 1e-5)
            h = nn.gelu(_linear(self, h, root + "mlp.0"))
            x = x + _linear(self, h, root + "mlp.2")
        x = mx.fast.layer_norm(x, self.w["policy.transformer.ln_f.weight"],
                               self.w["policy.transformer.ln_f.bias"], 1e-5)
        return _linear(self, x, "policy.lm_head")

    def action_head(self, features, *, argmax=False):
        batch, tokens, _ = features.shape
        flat = features.reshape(batch * tokens, -1)
        offsets = _mlp(self, flat, "action_head.map_to_cbet_preds_offset")
        groups, choices = 2, self.config.vqvae_n_embed
        offsets = offsets.reshape(batch * tokens, groups, choices, self.config.action_chunk_size,
                                  self.config.action_feature.shape[0])
        if self.config.sequentially_select:
            primary = _mlp(self, flat, "action_head.map_to_cbet_preds_primary_bin")
            primary = primary.reshape(batch * tokens, choices)
            primary_centers = (mx.argmax(primary, axis=-1) if argmax else
                               mx.random.categorical(primary / self.config.bet_softmax_temperature, axis=-1))
            primary_one_hot = mx.equal(primary_centers[..., None], mx.arange(choices)).astype(mx.float32)
            secondary_input = mx.concatenate((flat, primary_one_hot), axis=-1)
            secondary = _mlp(self, secondary_input, "action_head.map_to_cbet_preds_secondary_bin")
            secondary = secondary.reshape(batch * tokens, choices)
            secondary_centers = (mx.argmax(secondary, axis=-1) if argmax else
                                 mx.random.categorical(secondary / self.config.bet_softmax_temperature, axis=-1))
            centers = mx.stack((primary_centers, secondary_centers), axis=1)
        else:
            logits = _mlp(self, flat, "action_head.map_to_cbet_preds_bin")
            logits = logits.reshape(batch * tokens, groups, choices)
            centers = (mx.argmax(logits, axis=-1) if argmax else
                       mx.random.categorical(logits / self.config.bet_softmax_temperature, axis=-1))
        one_hot = mx.equal(centers[..., None], mx.arange(choices)).astype(mx.float32)
        selected = (offsets * one_hot[..., None, None]).sum(axis=2).sum(axis=1)
        codebooks = mx.stack(
            [self.w[f"action_head.vqvae_model.vq_layer.layers.{i}._codebook.embed"][0] for i in range(groups)],
            axis=0,
        )
        latent = (one_hot[..., None] * codebooks[None]).sum(axis=(1, 2))
        decoded = _mlp(self, latent, "action_head.vqvae_model.decoder", activations=(nn.relu, nn.relu))
        decoded = decoded.reshape(batch * tokens, self.config.action_chunk_size, self.config.action_feature.shape[0])
        predicted = decoded + selected
        return predicted.reshape(batch, tokens, -1)

    def __call__(self, batch, *, argmax=False):
        image_key = next(iter(self.config.image_features))
        image_value = batch[image_key]
        images = (image_value if image_value.shape[-1] == 3
                  else _as_mlx(image_value).transpose(0, 1, 3, 4, 2))
        state = _as_mlx(batch[STATE])
        batch_size, steps = state.shape[:2]
        image_features = _image_features(self, images.reshape(batch_size * steps, *images.shape[2:]),
                                         "rgb_encoder.", self.config.crop_shape)
        image_features = image_features.reshape(batch_size, steps, -1)
        rgb = _mlp(self, image_features, "rgb_feature_projector")
        state_tokens = _mlp(self, state, "state_projector")
        action_token = mx.broadcast_to(self.w["action_token"], (batch_size, steps, self.config.gpt_input_dim))
        tokens = mx.stack((rgb, state_tokens, action_token), axis=2).reshape(batch_size, steps * 3, -1)
        extra = self.config.n_action_pred_token - 1
        if extra > 0:
            tokens = mx.concatenate((tokens, mx.broadcast_to(
                self.w["action_token"], (batch_size, extra, self.config.gpt_input_dim))), axis=1)
        features = self.gpt(tokens)
        historical = mx.array(np.arange(steps) * 3 + 2, dtype=mx.int32)
        if extra > 0:
            features = mx.concatenate((features[:, historical], features[:, -extra:]), axis=1)
        else:
            features = features[:, historical]
        return self.action_head(features, argmax=argmax)


class VQBeTMLX(MLXSmallIO):
    kind = "vqbet"

    def __init__(self, path, task, **io):
        super().__init__(path, task, **io)
        self.network = VQBeTNetwork(mx.load(str(path / "model.safetensors")), self.config)
        self._compiled = None

    def _call_impl(self, batch, *, argmax=False, **kwargs):
        return self.network(batch, argmax=argmax)[:, self.n_obs_steps - 1, :].reshape(
            1, self.config.action_chunk_size, self.width
        )

    def __call__(self, imgs, q, *, argmax=False, **kwargs):
        batch = self.prepare_history(imgs, q)
        runner = self._compiled or self._call_impl
        if self._compiled is not None:
            batch = self._mlx_batch(batch)
        return self.finish(runner(batch, argmax=argmax))


class TDMPCRuntime(DenseWeights):
    def __init__(self, weights, config):
        weights = {k.removeprefix("model."): v for k, v in weights.items()
                   if k.startswith("model.") and not k.startswith("model_target.")}
        super().__init__(weights, mx.float32)
        self.config = config
        self.width = config.action_feature.shape[0]
        mx.eval(self.w)

    def encoder(self, batch):
        features = []
        image_key = next(iter(self.config.image_features), None)
        if image_key:
            image_value = batch[image_key]
            image = (image_value if image_value.shape[-1] == 3
                     else _as_mlx(image_value).transpose(0, 1, 3, 4, 2))[:, 0]
            x = image
            for index, activation in ((0, nn.relu), (2, nn.relu), (4, nn.relu), (6, nn.relu)):
                x = mx.conv2d(x, self.w[f"_encoder.image_enc_layers.{index}.weight"], stride=2)
                x = x + self.w[f"_encoder.image_enc_layers.{index}.bias"]
                x = activation(x)
            x = x.reshape(x.shape[0], -1)
            x = _linear(self, x, "_encoder.image_enc_layers.9")
            x = mx.fast.layer_norm(x, self.w["_encoder.image_enc_layers.10.weight"],
                                   self.w["_encoder.image_enc_layers.10.bias"], 1e-5)
            features.append(mx.sigmoid(x))
        if self.config.robot_state_feature:
            state = _as_mlx(batch[STATE])[:, 0]
            x = _linear(self, state, "_encoder.state_enc_layers.0")
            x = _elu(x)
            x = _linear(self, x, "_encoder.state_enc_layers.2")
            x = mx.fast.layer_norm(x, self.w["_encoder.state_enc_layers.3.weight"],
                                   self.w["_encoder.state_enc_layers.3.bias"], 1e-5)
            features.append(mx.sigmoid(x))
        return mx.stack(features, axis=0).mean(axis=0)

    def dynamics(self, z, a):
        x = mx.concatenate((z, a), axis=-1)
        x = _linear(self, x, "_dynamics.0")
        x = mx.fast.layer_norm(x, self.w["_dynamics.1.weight"], self.w["_dynamics.1.bias"], 1e-5)
        x = _mish(x)
        x = _linear(self, x, "_dynamics.3")
        x = mx.fast.layer_norm(x, self.w["_dynamics.4.weight"], self.w["_dynamics.4.bias"], 1e-5)
        x = _mish(x)
        x = _linear(self, x, "_dynamics.6")
        x = mx.fast.layer_norm(x, self.w["_dynamics.7.weight"], self.w["_dynamics.7.bias"], 1e-5)
        return mx.sigmoid(x)

    def reward(self, z, a):
        x = mx.concatenate((z, a), axis=-1)
        x = _linear(self, x, "_reward.0")
        x = mx.fast.layer_norm(x, self.w["_reward.1.weight"], self.w["_reward.1.bias"], 1e-5)
        x = _mish(x)
        x = _linear(self, x, "_reward.3")
        x = mx.fast.layer_norm(x, self.w["_reward.4.weight"], self.w["_reward.4.bias"], 1e-5)
        x = _mish(x)
        return _linear(self, x, "_reward.6").squeeze(-1)

    def pi(self, z, std=0.0):
        x = _linear(self, z, "_pi.0")
        x = mx.fast.layer_norm(x, self.w["_pi.1.weight"], self.w["_pi.1.bias"], 1e-5)
        x = _mish(x)
        x = _linear(self, x, "_pi.3")
        x = mx.fast.layer_norm(x, self.w["_pi.4.weight"], self.w["_pi.4.bias"], 1e-5)
        x = _mish(x)
        x = mx.tanh(_linear(self, x, "_pi.6"))
        return x if std <= 0 else x + std * mx.random.normal(x.shape)

    def q_values(self, z, a):
        values = []
        x = mx.concatenate((z, a), axis=-1)
        for index in range(self.config.q_ensemble_size):
            root = f"_Qs.{index}."
            y = _linear(self, x, root + "0")
            y = mx.fast.layer_norm(y, self.w[root + "1.weight"], self.w[root + "1.bias"], 1e-5)
            y = mx.tanh(y)
            y = _linear(self, y, root + "3")
            y = nn.elu(y)
            values.append(_linear(self, y, root + "5").squeeze(-1))
        return mx.stack(values, axis=0)

    def plan(self, z):
        cfg = self.config
        batch = z.shape[0]
        pi_actions = mx.zeros((cfg.horizon, cfg.n_pi_samples, batch, self.width))
        if cfg.n_pi_samples:
            latent = mx.repeat(z[None], cfg.n_pi_samples, axis=0)
            for t in range(cfg.horizon):
                action = self.pi(latent, cfg.min_std)
                pi_actions[t] = action
                latent = self.dynamics(latent, action)
        all_count = cfg.n_gaussian_samples + cfg.n_pi_samples
        latent0 = mx.repeat(z[None], all_count, axis=0)
        mean = mx.zeros((cfg.horizon, batch, self.width))
        std = mx.full(mean.shape, cfg.max_std)
        elite_actions = None
        scores = None
        for _ in range(cfg.cem_iterations):
            noise = mx.random.normal((cfg.horizon, cfg.n_gaussian_samples, batch, self.width))
            gaussian = mx.clip(mean[:, None] + std[:, None] * noise, -1, 1)
            actions = mx.concatenate((gaussian, pi_actions), axis=1)
            latent = latent0
            total = mx.zeros((all_count, batch))
            discount = 1.0
            for t in range(cfg.horizon):
                q = self.q_values(latent, actions[t])
                reg = -cfg.uncertainty_regularizer_coeff * q.std(axis=0) if cfg.uncertainty_regularizer_coeff > 0 else 0
                reward = self.reward(latent, actions[t])
                latent = self.dynamics(latent, actions[t])
                total = total + discount * (reward + reg)
                discount *= cfg.discount
            order = mx.argsort(total, axis=0)
            elite = order[-cfg.n_elites:]
            elite_value = mx.take_along_axis(total, elite, axis=0)
            elite_actions = mx.take_along_axis(actions, elite[None, ..., None], axis=1)
            top = elite_value.max(axis=0, keepdims=True)
            scores = mx.exp(cfg.elite_weighting_temperature * (elite_value - top))
            scores = scores / scores.sum(axis=0, keepdims=True)
            weighted = scores[None, ..., None] * elite_actions
            new_mean = weighted.sum(axis=1)
            new_std = mx.sqrt((scores[None, ..., None] * (elite_actions - new_mean[:, None]) ** 2).sum(axis=1))
            mean = cfg.gaussian_mean_momentum * mean + (1 - cfg.gaussian_mean_momentum) * new_mean
            std = mx.clip(new_std, cfg.min_std, cfg.max_std)
        if elite_actions is None:
            return self.pi(z)[None]
        selected = mx.random.categorical(mx.log(scores.T), axis=-1)
        selected = selected.astype(mx.int32)
        return mx.take_along_axis(elite_actions, selected[None, :, None, None], axis=1)[:, 0]


class TDMPCMLX(MLXSmallIO):
    kind = "tdmpc"

    def __init__(self, path, task, **io):
        super().__init__(path, task, **io)
        self.network = TDMPCRuntime(mx.load(str(path / "model.safetensors")), self.config)
        self._prev_mean = None
        self._compiled = None

    def reset(self):
        super().reset()
        self._prev_mean = None

    def _call_impl(self, batch, **kwargs):
        z = self.network.encoder(batch)
        if self.config.use_mpc:
            actions = self.network.plan(z)
        else:
            actions = self.network.pi(z)[None]
        actions = mx.clip(actions, -1, 1)
        return actions[:self.H]

    def __call__(self, imgs, q, **kwargs):
        batch = self.prepare_history(imgs, q)
        runner = self._compiled or self._call_impl
        if self._compiled is not None:
            batch = self._mlx_batch(batch)
        return self.finish(runner(batch))
