"""Published policy coverage, independent of whether model extras are installed."""
from __future__ import annotations

from importlib import import_module

from .backends import available

BUILTINS = (
    "act", "cosmos3_edge", "cosmos3_nano", "diffusion", "dm05", "eo1", "evo1", "fastwam",
    "fineart_vla", "flux3", "g05", "gaussian_actor", "groot", "lawam", "lingbot_va", "lingbot_vla_v2",
    "longwam", "molmoact2", "multi_task_dit", "pi0", "pi05", "pi0_fast", "smolvla", "tdmpc",
    "vla_jepa", "vqbet", "wall_oss_05", "wall_x", "xvla",
)

STATUS = {
    "act": ("verified", "built-in wrapper and generic chunks"),
    "diffusion": ("runs, tiny checkpoint", "generic select_action; tiny DDIM checkpoint"),
    "gaussian_actor": ("runs, tiny checkpoint", "generic select_action; continuous actions"),
    "pi0": ("runs, random weights", "torch generic chunks; MLX port has no trained-checkpoint verification"),
    "pi0_fast": ("not verified", "weights loaded; random tokens failed FAST action decoding"),
    "tdmpc": ("runs, tiny checkpoint", "generic select_action; one camera named observation.image"),
    "vqbet": ("runs, tiny checkpoint", "generic select_action; one camera"),
    "pi05": ("verified", "built-in PyTorch MPS and MLX wrappers"),
    "smolvla": ("verified", "torch generic chunks; MLX port has no trained-checkpoint verification"),
    "xvla": ("verified", "SO-101 joint checkpoint; end-effector variants are unsupported"),
    "groot": ("not verified", "GR00T base model and robot-specific embodiment checkpoint needed"),
    "wall_x": ("not verified", "needs a compatible WALL-X checkpoint and Qwen vision-language assets"),
    "wall_oss_05": ("not verified", "separate WALL-OSS model and robot action convention needed"),
    "multi_task_dit": ("runs, tiny checkpoint", "generic select_action; tiny DDIM and local CLIP checkpoint"),
}


def run():
    try:
        from lerobot.configs import PreTrainedConfig

        import_module("lerobot.policies")  # Registers the built-in configs without loading models.
    except ImportError:
        kinds = BUILTINS
        print("LeRobot registry unavailable; showing the built-in 0.7.0 list.")
    else:
        kinds = PreTrainedConfig.get_known_choices()
    print(f"{'Policy':20} {'Backends':12} {'MPS status':26} Evidence / limitation")
    for kind in sorted(kinds):
        status, note = STATUS.get(kind, ("not verified", "generic candidate; no validated Mac checkpoint"))
        print(f"{kind:20} {', '.join(available(kind)):12} {status:26} {note}")
    print("\nCoverage is checkpoint-specific. See docs/policies.md for tests, options and limits.")
