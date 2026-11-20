"""Published policy coverage, independent of whether model extras are installed."""
from __future__ import annotations

from importlib import import_module

BUILTINS = (
    "act", "cosmos3_edge", "cosmos3_nano", "diffusion", "dm05", "eo1", "evo1", "fastwam",
    "fineart_vla", "flux3", "g05", "gaussian_actor", "groot", "lawam", "lingbot_va", "lingbot_vla_v2",
    "longwam", "molmoact2", "multi_task_dit", "pi0", "pi05", "pi0_fast", "smolvla", "tdmpc",
    "vla_jepa", "vqbet", "wall_oss_05", "wall_x", "xvla",
)

STATUS = {
    "act": ("verified", "built-in wrapper and generic chunks"),
    "diffusion": ("verified", "generic select_action; tiny DDIM checkpoint"),
    "tdmpc": ("verified", "generic select_action; one camera named observation.image"),
    "vqbet": ("verified", "generic select_action; one camera"),
    "pi05": ("verified", "built-in PyTorch MPS and MLX wrappers"),
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
    print(f"{'Policy':20} {'MPS status':26} Evidence / limitation")
    for kind in sorted(kinds):
        status, note = STATUS.get(kind, ("not verified", "generic candidate; no validated Mac checkpoint"))
        print(f"{kind:20} {status:26} {note}")
    print("\nCoverage is checkpoint-specific. See docs/policies.md for tests, options and limits.")
