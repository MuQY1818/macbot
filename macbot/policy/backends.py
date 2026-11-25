"""Select a policy backend from published evidence, without timing inference."""
from __future__ import annotations

import json
import platform
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
from typing import NamedTuple


class Backend(NamedTuple):
    name: str
    modes: frozenset[str]


CHUNK = frozenset({"chunk", "ensemble", "async"})
MLX_BACKENDS = {
    "act": Backend("mlx", CHUNK),
    "pi0": Backend("mlx", CHUNK | {"rtc"}),
    "pi0_fast": Backend("mlx", CHUNK),
    "pi05": Backend("mlx", CHUNK | {"rtc"}),
    "smolvla": Backend("mlx", CHUNK),
    "xvla": Backend("mlx", CHUNK),
}


@dataclass(frozen=True)
class Qualification:
    """A measured implementation for a compatible checkpoint layout and mode."""

    backend: str
    evidence: str
    modes: frozenset[str]
    compiled: bool
    compatible: Callable[[Path, dict], bool]


def stock_smolvla(metadata):
    options = {
        "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "resize_imgs_with_padding": (512, 512),
        "num_vlm_layers": 16,
        "self_attn_every_n_layers": 2,
        "attention_mode": "cross_attn",
        "expert_width_multiplier": 0.75,
        "use_cache": True,
        "empty_cameras": 0,
        "add_image_special_tokens": False,
        "adapt_to_pi_aloha": False,
        "use_delta_joint_actions_aloha": False,
    }
    return (all(metadata.get(k, value) == value for k, value in options.items())
            and metadata.get("num_expert_layers", -1) in (-1, 0, 16)
            and metadata.get("prefix_length", -1) <= 0)


def compatible_smolvla(path, metadata):
    fields = {**metadata, "resize_imgs_with_padding": tuple(metadata.get("resize_imgs_with_padding", (512, 512)) or ())}
    if not stock_smolvla(fields) or metadata.get("n_obs_steps", 1) != 1 or metadata.get("num_steps", 10) != 10:
        return False
    expected = {
        "pre": ("rename_observations_processor", "to_batch_processor", "smolvla_new_line_processor",
                "tokenizer_processor", "device_processor", "normalizer_processor"),
        "post": ("unnormalizer_processor", "device_processor"),
    }
    for side, layout in expected.items():
        steps = json.loads((path / f"policy_{side}processor.json").read_text())["steps"]
        if tuple(step.get("registry_name") for step in steps) != layout:
            return False
    return (path / "model.safetensors").is_file() and not (path / "adapter_config.json").is_file()


def compatible_pi(path, metadata):
    stock = {
        "paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m", "n_obs_steps": 1,
        "image_resolution": [224, 224], "empty_cameras": 0, "use_relative_actions": False,
        "chunk_size": 50, "n_action_steps": 50, "max_state_dim": 32, "max_action_dim": 32,
        "num_inference_steps": 10, "min_period": 0.004, "max_period": 4.0,
    }
    if any(metadata.get(k, default) != default for k, default in stock.items()):
        return False
    pi05 = metadata["type"] == "pi05"
    common = ("rename_observations_processor", "to_batch_processor")
    expected = {
        "pre": common + (("normalizer_processor", "pi05_prepare_state_tokenizer_processor_step",
                          "tokenizer_processor", "device_processor") if pi05 else
                         ("pi0_new_line_processor", "tokenizer_processor", "device_processor", "normalizer_processor")),
        "post": ("unnormalizer_processor", "device_processor"),
    }
    normalization = "QUANTILES" if pi05 else "MEAN_STD"
    for side, layout in expected.items():
        steps = json.loads((path / f"policy_{side}processor.json").read_text())["steps"]
        active_steps = [s for s in steps
                        if not (s["registry_name"] in {"relative_actions_processor", "absolute_actions_processor"}
                                and s.get("config", {}).get("enabled") is False)]
        if tuple(s["registry_name"] for s in active_steps) != layout:
            return False
        active = {s["registry_name"]: s.get("config", {}) for s in active_steps}
        norm = active["normalizer_processor" if side == "pre" else "unnormalizer_processor"]
        if norm.get("norm_map") != {"VISUAL": "IDENTITY", "STATE": normalization, "ACTION": normalization}:
            return False
        if side == "pre":
            if active["tokenizer_processor"].get("padding_side") != "right":
                return False
            mapping = active["rename_observations_processor"].get("rename_map", {})
            if any(key in metadata.get("input_features", {}) and key != value for key, value in mapping.items()):
                return False
    return (path / "model.safetensors").is_file() and not (path / "adapter_config.json").is_file()


def compatible_xvla(path, metadata):
    options = {
        "n_obs_steps": 1, "chunk_size": 30, "n_action_steps": 30, "num_denoising_steps": 10,
        "dtype": "bfloat16", "action_mode": "auto", "use_proprio": True, "max_state_dim": 20,
        "max_action_dim": 20, "num_image_views": 4, "empty_cameras": 1, "resize_imgs_with_padding": [224, 224],
        "hidden_size": 1024, "depth": 24, "num_heads": 16, "mlp_ratio": 4.0, "num_domains": 30,
        "dim_time": 32, "len_soft_prompts": 32, "max_len_seq": 512, "use_hetero_proj": False,
    }
    if any(metadata.get(k) != v for k, v in options.items()):
        return False
    if (metadata.get("output_features", {}).get("action") or {}).get("shape") != [6]:
        return False
    visuals = [f["shape"] for k, f in metadata.get("input_features", {}).items()
               if f["type"] == "VISUAL" and k != "observation.images.empty_camera_0"]
    if visuals != [[3, 256, 256]] * 2:
        return False
    florence = metadata["florence_config"]
    vision, text = florence["vision_config"], florence["text_config"]
    if (vision.get("dim_embed", vision.get("embed_dim")) != [256, 512, 1024, 2048]
            or vision.get("depths") != [1, 1, 9, 1] or vision.get("window_size") != 12
            or text.get("encoder_layers") != 12 or text.get("d_model") != 1024):
        return False
    expected = {
        "pre": ("rename_observations_processor", "to_batch_processor", "tokenizer_processor",
                "xvla_add_domain_id", "xvla_image_to_float", "xvla_imagenet_normalize",
                "device_processor", "normalizer_processor"),
        "post": ("unnormalizer_processor", "device_processor"),
    }
    norm = {"STATE": "IDENTITY", "ACTION": "MEAN_STD", "VISUAL": "IDENTITY"}
    for side, layout in expected.items():
        steps = json.loads((path / f"policy_{side}processor.json").read_text())["steps"]
        if tuple(step["registry_name"] for step in steps) != layout:
            return False
        normalizer = steps[-1 if side == "pre" else 0]
        if normalizer["config"]["norm_map"] != norm or not (path / normalizer["state_file"]).is_file():
            return False
        if side == "pre":
            tokenizer = steps[2]["config"]
            if (tokenizer.get("tokenizer_name") != "facebook/bart-large" or tokenizer.get("max_length") != 50
                    or tokenizer.get("padding") != "max_length" or tokenizer.get("padding_side") != "right"
                    or steps[3]["config"].get("domain_id") != 0):
                return False
    return (path / "model.safetensors").is_file() and not (path / "adapter_config.json").is_file()


def compatible_act(path, metadata):
    stock = {
        "n_obs_steps": 1, "chunk_size": 100, "n_action_steps": 100, "vision_backbone": "resnet18",
        "replace_final_stride_with_dilation": False, "pre_norm": False, "dim_model": 512, "n_heads": 8,
        "dim_feedforward": 3200, "feedforward_activation": "relu", "n_encoder_layers": 4,
        "n_decoder_layers": 1, "temporal_ensemble_coeff": None,
    }
    if any(metadata.get(k, v) != v for k, v in stock.items()):
        return False
    features = metadata.get("input_features", {})
    # The public timing covers a single 640x480 view. The local two-camera
    # result is useful for explicit deployment but is not public evidence.
    if features != {"observation.state": {"type": "STATE", "shape": [14]},
                    "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]}}:
        return False
    if metadata.get("output_features") != {"action": {"type": "ACTION", "shape": [14]}}:
        return False
    layouts = {
        "pre": ("rename_observations_processor", "to_batch_processor", "device_processor", "normalizer_processor"),
        "post": ("unnormalizer_processor", "device_processor"),
    }
    for side, layout in layouts.items():
        steps = json.loads((path / f"policy_{side}processor.json").read_text())["steps"]
        if tuple(step["registry_name"] for step in steps) != layout:
            return False
        normalizer = steps[-1 if side == "pre" else 0]
        if normalizer["config"]["norm_map"] != {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}:
            return False
        if not (path / normalizer["state_file"]).is_file():
            return False
    return (path / "model.safetensors").is_file() and not (path / "adapter_config.json").is_file()


def pi05_camera_map(config, metadata):
    from .models import PI05

    prefix = "observation.images."
    expected = {name.removeprefix(prefix) for name, feature in metadata.get("input_features", {}).items()
                if name.startswith(prefix) and feature.get("type") == "VISUAL"}
    explicit = config.get("policy", {}).get("camera_map", {})
    cameras = config["cameras"]
    if explicit.keys() - cameras.keys():
        raise ValueError("policy.camera_map names unknown cameras")
    result = {}
    for name in cameras:
        if name in explicit:
            target = explicit[name].removeprefix(prefix)
            if target not in expected:
                raise ValueError(f"camera {name!r} maps to an unknown feature")
        elif name in expected:
            target = name
        else:
            target = PI05.RENAME.get(name, name)
        if target in expected:
            if target in result.values():
                raise ValueError(f"multiple cameras map to {target!r}")
            result[name] = target
    # The three-slot base layout leaves right_wrist_0_rgb empty on two-camera arms; the pi0.5 wrappers mask it,
    # as in explicit --backend mlx runs. Every other checkpoint view needs a camera.
    required = expected - {"right_wrist_0_rgb"} if {"base_0_rgb", "left_wrist_0_rgb"} <= expected else expected
    if not required or not required <= set(result.values()) or result.keys() != cameras.keys():
        raise ValueError("configured cameras do not cover the checkpoint views")
    return result


# Entries must link to a trained public checkpoint and matched fidelity and latency
# results in docs/acceleration.md. Results from private checkpoints do not transfer.
QUALIFIED: dict[str, tuple[Qualification, ...]] = {
    "act": (
        Qualification("mlx", "lerobot/act_aloha_sim_transfer_cube_human", frozenset({"chunk"}), True,
                      compatible_act),
    ),
    "pi05": (
        Qualification("mlx", "zz4321/so101_pi05", frozenset({"chunk", "rtc"}), True, compatible_pi),
    ),
    "pi0": (
        Qualification("mlx", "zz4321/so101_pi0_cube", frozenset({"chunk", "rtc"}), True, compatible_pi),
    ),
    "pi0_fast": (),
    "smolvla": (
        Qualification(
            "mlx", "lerobot/smolvla_base", frozenset({"chunk"}), True, compatible_smolvla,
        ),
    ),
    "xvla": (
        Qualification("mlx", "whosricky/xvla-so101-pick-red-cube-2cam", frozenset({"chunk"}), True, compatible_xvla),
    ),
}


@dataclass(frozen=True)
class Selection:
    requested: str
    backend: str
    reason: str
    camera_map: dict[str, str] | None = None

    def announce(self):
        prefix = "Backend auto selected" if self.requested == "auto" else "Backend"
        print(f"{prefix}: {self.backend} ({self.reason})", flush=True)


def available(kind):
    return ("torch", MLX_BACKENDS[kind].name) if kind in MLX_BACKENDS else ("torch",)


def supported_mlx_dependencies():
    from packaging.version import InvalidVersion, Version

    try:
        lerobot = Version(version("lerobot"))
        return Version("0.7") <= lerobot < Version("0.8") and Version(version("mlx")) >= Version("0.32.4")
    except (PackageNotFoundError, InvalidVersion):
        return False


def mlx_runtime_available():
    try:
        import mlx.core as mx

        return mx.metal.is_available()
    except (ImportError, OSError, RuntimeError):
        return False


def select_backend(requested, path, metadata, args, *, bench=False, config=None):
    if requested != "auto":
        return Selection(requested, requested, "configured")
    if (getattr(args, "mlx_bits", None) is not None or getattr(args, "mlx_time_grid", "uniform") != "uniform"):
        raise SystemExit("Lossy MLX options require explicit --backend mlx")
    steps = getattr(args, "steps", None)
    if steps is not None and steps != metadata.get("num_inference_steps", 10):
        raise SystemExit("Changing denoising steps requires an explicit backend")

    def fallback(reason):
        return Selection("auto", "torch", reason)

    if getattr(args, "policy_impl", "auto") == "generic":
        return fallback("generic implementation requires torch")
    if args.device != "mps":
        return fallback(f"no accelerated backend qualified on {args.device}")
    kind = metadata.get("type", "act")
    capability = MLX_BACKENDS.get(kind)
    if capability is None:
        return fallback(f"no alternative backend for {kind}")
    mode = getattr(args, "mode", "auto")
    if mode == "auto":
        mode = "rtc" if kind == "pi05" and not bench else "chunk"
    if mode not in capability.modes:
        return fallback(f"{capability.name} does not support {mode} for {kind}")
    if any(metadata.get(k) for k in ("use_visual_memory", "use_proprioceptive_memory")) or (
        metadata.get("rtc_config") or {}
    ).get("enabled"):
        return fallback("checkpoint uses memory or checkpoint-configured RTC")
    if not (rows := QUALIFIED.get(kind)):
        return fallback(f"no public checkpoint qualified for {kind}")
    if platform.system() != "Darwin" or platform.machine() != "arm64" or find_spec("mlx") is None:
        return fallback("optional Apple silicon MLX runtime unavailable")

    compiled = not getattr(args, "no_compile", False)
    if kind == "act" and not bench and getattr(args, "mode", "auto") == "auto":
        return fallback("ACT defaults to ensemble; select --mode chunk to use qualified MLX")
    applicable = [row for row in rows if row.backend == capability.name and mode in row.modes
                  and row.compiled == compiled]
    if not applicable:
        return fallback(f"no qualified {mode} result with these compiler settings")
    if not supported_mlx_dependencies():
        return fallback("requires LeRobot 0.7.x and MLX 0.32.4 or newer")
    try:
        for row in applicable:
            if row.compatible(path, metadata):
                if not mlx_runtime_available():
                    return fallback("MLX could not access the Apple GPU")
                camera_map = None
                if kind == "pi05" and config is not None:
                    try:
                        camera_map = pi05_camera_map(config, metadata)
                    except (AttributeError, KeyError, TypeError, ValueError) as exc:
                        return fallback(f"cannot confirm pi0.5 camera mapping: {exc}")
                if kind in {"act", "xvla"} and config is not None:
                    from .generic import _features

                    try:
                        _features(path, metadata, {name: v["save_size"] for name, v in config["cameras"].items()},
                                  config.get("policy", {}).get("camera_map", {}), None)
                    except (KeyError, TypeError, ValueError) as exc:
                        return fallback(f"cannot confirm {kind} camera mapping: {exc}")
                return Selection("auto", row.backend, f"compatible {kind} {mode}; measured on {row.evidence}",
                                 camera_map)
    except (OSError, ValueError, KeyError, TypeError):
        return fallback("checkpoint processor layout could not be confirmed")
    return fallback("checkpoint layout does not match the qualified implementation")
