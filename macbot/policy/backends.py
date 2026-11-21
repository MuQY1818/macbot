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
    "pi0": Backend("mlx", CHUNK | {"rtc"}),
    "pi05": Backend("mlx", CHUNK | {"rtc"}),
    "smolvla": Backend("mlx", CHUNK),
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


# Entries must link to a trained public checkpoint and matched fidelity and latency
# results in docs/acceleration.md. Results from private checkpoints do not transfer.
QUALIFIED: dict[str, tuple[Qualification, ...]] = {
    "smolvla": (
        Qualification(
            "mlx", "lerobot/smolvla_base", frozenset({"chunk"}), True, compatible_smolvla,
        ),
    ),
}


@dataclass(frozen=True)
class Selection:
    requested: str
    backend: str
    reason: str

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


def select_backend(requested, path, metadata, args, *, bench=False):
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
                return Selection("auto", row.backend, f"compatible {kind} {mode}; measured on {row.evidence}")
    except (OSError, ValueError, KeyError, TypeError):
        return fallback("checkpoint processor layout could not be confirmed")
    return fallback("checkpoint layout does not match the qualified implementation")
