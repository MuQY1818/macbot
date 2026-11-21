"""Parse a LeRobot command as data, without invoking a shell or a driver."""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import PurePath

import yaml

from .robot.usb import FAMILIES

COMMANDS = {"lerobot-record", "lerobot-teleoperate"}
SERIAL_FAMILIES = set(FAMILIES) - {"rebot_b601"}


@dataclass
class ImportedSetup:
    profile: dict
    family: str
    camera_addresses: dict
    notes: list[str]


def _flags(command):
    tokens = shlex.split(re.sub(r"\\\r?\n", "", command))
    if tokens and PurePath(tokens[0]).name in COMMANDS:
        tokens.pop(0)
    flags = {}
    it = iter(tokens)
    for token in it:
        if not token.startswith("--"):
            raise ValueError("Paste LeRobot flags or a lerobot-record / lerobot-teleoperate command.")
        name, eq, value = token[2:].partition("=")
        if not eq:
            value = next(it, "")
            if not value or value.startswith("--"):
                raise ValueError(f"--{name} needs a value (use --{name}=VALUE).")
        if name in flags:
            raise ValueError(f"--{name} was supplied more than once.")
        flags[name] = value
    return flags


def _rate(value, label):
    try:
        rate = int(value)
        if isinstance(value, bool) or rate <= 0 or (not isinstance(value, str) and rate != value):
            raise ValueError
        return rate
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer.") from exc


def _yaml(value, label):
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse {label}: {exc}") from exc


def parse(command):
    flags = _flags(command)
    robot = {"type": "lerobot"}
    families = []
    for role, prefix in [("follower", "robot"), ("leader", "teleop")]:
        spec = {key: flags.pop(f"{prefix}.{key}") for key in ("type", "port", "id", "calibration_dir")
                if f"{prefix}.{key}" in flags}
        kind = spec.get("type", "")
        family = kind.removesuffix(f"_{role}")
        if kind != f"{family}_{role}" or family not in SERIAL_FAMILIES:
            raise ValueError(f"--{prefix}.type={kind or '(missing)'} is not a supported serial {role} for import. "
                             "Use a macbot YAML for other drivers; see docs/robots.md.")
        families.append(family)
        spec.setdefault("id", None)  # LeRobot looks for None.json when no ID is supplied.
        for key in ("use_degrees", "max_relative_target"):
            flag = f"{prefix}.{key}"
            if flag in flags:
                spec[key] = _yaml(flags.pop(flag), f"--{flag}")
        robot[role] = spec
    if families[0] != families[1] and not set(families) <= {"so100", "so101"}:
        raise ValueError("This follower/leader pair needs a manual macbot YAML; see docs/robots.md.")

    notes, cameras, addresses = [], {}, {}
    for key, value in _camera_map(flags.pop("robot.cameras", "{}")).items():
        if not isinstance(key, str) or not key or not isinstance(value, dict):
            raise ValueError("--robot.cameras must map nonempty camera names to configurations.")
        if value.get("type") != "opencv":
            raise ValueError(f"Camera {key!r}: {value.get('type')!r} cannot be imported. "
                             "Configure this camera in a macbot YAML; see docs/setup.md.")
        camera = {"type": "uvc"}
        if value.get("fps") is not None:
            camera["fps"] = _rate(value["fps"], f"camera {key} fps")
        size = [value.get(k) for k in ("width", "height")]
        if all(x is not None for x in size):
            camera["expect_size"] = camera["save_size"] = [_rate(x, f"camera {key} size") for x in size]
        elif any(x is not None for x in size):
            raise ValueError(f"Camera {key!r}: provide both width and height.")
        unused = value.keys() - {"type", "index_or_path", "width", "height", "fps"}
        unused -= {k for k, default in {"rotation": 0, "color_mode": "rgb"}.items() if value.get(k) == default}
        if unused:
            notes.append(f"Camera {key!r}: settings not mapped: {', '.join(sorted(map(str, unused)))}.")
        cameras[key] = camera
        addresses[key] = value.get("index_or_path")

    profile = {"robot": robot, "cameras": cameras}
    if (fps := flags.pop("dataset.fps", None)) is not None:
        profile["record"] = {"fps": _rate(fps, "--dataset.fps")}
    if (fps := flags.pop("fps", None)) is not None:
        profile["teleop"] = {"hz": _rate(fps, "--fps")}
    if (task := flags.pop("dataset.single_task", None)) is not None:
        profile["task"] = {"instruction": task}
    if flags:
        notes.append("Flags not mapped: " + ", ".join(f"--{key}" for key in flags) + ".")
    return ImportedSetup(profile, families[0], addresses, notes)


def _camera_map(raw):
    cameras = _yaml(raw, "--robot.cameras")
    if not isinstance(cameras, dict):
        raise ValueError("--robot.cameras must be a mapping.")
    return cameras
