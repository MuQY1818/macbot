"""Saved LeRobot processors for single-observation MLX chunk policies."""
from __future__ import annotations

import json

import mlx.core as mx
import numpy as np

from .generic import STATE, _features, _processor_overrides


class MLXChunkIO:
    shapes = None
    stepwise = False
    supports_rtc = False
    requires_task = True

    def __init__(self, path, task, *, cameras, camera_map=None, joint_count=None):
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors

        self.torch, self.task = torch, task
        metadata = json.loads((path / "config.json").read_text())
        self.width, self.has_state, self.cameras = _features(path, metadata, cameras, camera_map or {}, joint_count)
        config = PreTrainedConfig.from_pretrained(path, cli_overrides=["--device", "cpu"])
        if config.type != self.kind or not self.has_state or not self.cameras:
            raise ValueError(f"MLX {self.kind} requires joint state and RGB cameras")
        if self.requires_task and not task.strip():
            raise ValueError(f"MLX {self.kind} requires task text")
        if config.n_obs_steps != 1 or getattr(getattr(config, "rtc_config", None), "enabled", False):
            raise ValueError(f"MLX {self.kind} requires one observation and no checkpoint-configured RTC")
        if not 0 < config.n_action_steps <= config.chunk_size:
            raise ValueError("Invalid checkpoint action horizon")
        config.validate_features()
        self.config = config
        self.H = config.n_action_steps
        pre, steps = _processor_overrides(path, "pre", "cpu")
        post, _ = _processor_overrides(path, "post", "cpu")
        self.task_keys = {step.get("config", {}).get("task_key", "task") for step in steps
                          if step.get("registry_name") == "tokenizer_processor"
                          or step.get("class", "").endswith(".TokenizerProcessorStep")}
        self.pre, self.post = make_pre_post_processors(
            config, pretrained_path=path, preprocessor_overrides=pre, postprocessor_overrides=post,
        )

    def reset(self):
        self.pre.reset()
        self.post.reset()

    def prepare_batch(self, imgs, q):
        torch = self.torch
        obs = {key: [self.task] for key in self.task_keys | {"task"}}
        obs[STATE] = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
        for key, name in self.cameras.items():
            obs[key] = torch.from_numpy(imgs[name]).permute(2, 0, 1).float().div_(255).unsqueeze(0)
        return self.pre(obs)

    def starting_noise(self, noise):
        shape = (1, self.config.chunk_size, self.config.max_action_dim)
        if noise is None:
            return mx.random.normal(shape)
        noise = mx.array(noise, dtype=mx.float32)
        if noise.shape != shape:
            raise ValueError(f"Expected starting noise with shape {shape}, got {noise.shape}")
        return noise

    def finish(self, x, return_raw=False):
        raw = np.array(x[..., :self.width], dtype=np.float32)
        action = self.post(self.torch.from_numpy(raw)).float().cpu().numpy()
        if action.shape != raw.shape or not np.isfinite(action).all():
            raise ValueError(f"MLX {self.kind} returned an invalid joint chunk")
        return (action[0, :self.H], raw[0, :self.H]) if return_raw else action[0, :self.H]


def stage_call(stage, name, fn, *args):
    return fn(*args) if stage is None else stage(name, fn, *args)


class DenseWeights:
    """Torch checkpoint layout with NHWC convolutions and fused attention."""

    def __init__(self, weights, dtype):
        self.dtype = dtype
        self.w = {}
        for key, weight in weights.items():
            if weight.ndim == 4 and key.endswith(".weight"):
                weight = mx.contiguous(weight.transpose(0, 2, 3, 1))
            self.w[key] = weight.astype(dtype)

    def linear(self, x, name):
        w = self.w[name + ".weight"]
        bias = self.w.get(name + ".bias")
        return x @ w.T if bias is None else mx.addmm(bias, x, w.T)

    def norm(self, x, name):
        return mx.fast.layer_norm(x, self.w[name + ".weight"], self.w[name + ".bias"], 1e-5)

    def conv(self, x, name, *, stride=1, padding=0, groups=1):
        y = mx.conv2d(x, self.w[name + ".weight"], stride=stride, padding=padding, groups=groups)
        bias = self.w.get(name + ".bias")
        return y if bias is None else y + bias


def attention(q, k, v, heads):
    def split(x):
        return x.reshape(*x.shape[:2], heads, -1).transpose(0, 2, 1, 3)

    y = mx.fast.scaled_dot_product_attention(split(q), split(k), split(v), scale=(q.shape[-1] // heads)**-0.5)
    return y.transpose(0, 2, 1, 3).flatten(2)
