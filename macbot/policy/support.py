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
    "act": ("verified", "Torch and full-precision MLX chunks on trained checkpoints"),
    "diffusion": ("runs, trained public checkpoint", "explicit MLX matches PyTorch on the CPU but is slower than MPS"),
    "gaussian_actor": ("runs, tiny checkpoint", "generic select_action; continuous actions"),
    "pi0": ("verified", "MLX chunks and RTC checked on a trained SO-101 checkpoint; torch generic adapter"),
    "pi0_fast": ("runs, trained checkpoint", "explicit MLX greedy decoding; tokens diverge from PyTorch"),
    "tdmpc": ("runs, tiny checkpoint", "torch generic and explicit MLX; one camera named observation.image"),
    "vqbet": ("runs, trained public checkpoint", "explicit MLX, 1.8x faster; matches PyTorch on the CPU"),
    "pi05": ("verified", "built-in PyTorch MPS and MLX wrappers"),
    "smolvla": ("verified", "official base checkpoint: torch and MLX chunks; see auto limits"),
    "xvla": ("verified", "Torch and MLX chunks on a public SO-101 joint checkpoint; no RTC"),
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
