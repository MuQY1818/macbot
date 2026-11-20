"""Configuration loading. One YAML file describes the task, the robot, the cameras and the paths."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .system import expand

PACKAGE = Path(__file__).resolve().parent
DEFAULT = PACKAGE / "default.yaml"

CAMERAS = ("top", "wrist")
CONSOLES = ("init", "teleop", "run", "replay")


REPLACE = {"cameras"}      # sections a user file replaces as a whole instead of merging into


class ConsolePortError(ValueError):
    pass


def _merge(base: dict, over: dict, top: bool = True) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        merge = isinstance(v, dict) and isinstance(out.get(k), dict) and not (top and k in REPLACE)
        out[k] = _merge(out[k], v, top=False) if merge else v
    return out


class Config(dict):
    """A dict with a few path helpers."""

    def __init__(self, settings=(), explicit_ports=()):
        super().__init__(settings)
        self.explicit_ports = frozenset(explicit_ports)

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
    overrides = (yaml.safe_load(expand(path).read_text()) or {}) if path else {}
    explicit_ports = [section for section in CONSOLES if "port" in (overrides.get(section) or {})]
    return Config(_merge(cfg, overrides), explicit_ports)


def setup_path() -> Path:
    """The per-user profile created only when setup is explicitly saved."""
    return expand("~/.config/macbot/config.yaml")


def load_cli(path=None, *, resolve=None) -> Config:
    if path is None and setup_path().is_file():
        path = setup_path()
    cfg = load(path)
    if resolve is not None:
        from .robot.usb import resolve_robot

        cfg["robot"] = resolve_robot(cfg["robot"], leader=resolve)
    return cfg
