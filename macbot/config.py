"""Configuration loading. One YAML file describes the task, the robot, the cameras and the paths."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .system import expand

PACKAGE = Path(__file__).resolve().parent
DEFAULT = PACKAGE / "default.yaml"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]
ARM = JOINTS[:6]
CAMERAS = ("top", "wrist")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Config(dict):
    """A dict with a few path helpers."""

    @property
    def root(self) -> Path:
        return expand(self["data_root"])

    @property
    def task(self) -> str:
        return self["task"]["name"]

    def raw_dir(self, task: str | None = None) -> Path:
        return self.root / "raw" / (task or self.task)

    def dataset_dir(self, name: str) -> Path:
        return self.root / "datasets" / name

    def rollout_dir(self, task: str | None = None) -> Path:
        return self.root / "rollouts" / (task or self.task)

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"


def load(path: str | Path | None = None) -> Config:
    """Load the default config, then overlay ``path`` if given."""
    cfg = yaml.safe_load(DEFAULT.read_text())
    if path:
        cfg = _merge(cfg, yaml.safe_load(expand(path).read_text()) or {})
    return Config(cfg)
