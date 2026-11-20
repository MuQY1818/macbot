"""Capture and output settings shared by recording, export and policy preflight."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..system import expand
from . import _crop_box

FRAMING_FILE = "camera_framing.json"


def framing(cameras, fps):
    result = {}
    for role, camera in cameras.items():
        kind = camera["type"]
        size = camera.get("expect_size")
        if kind == "uvc":
            mode = {"preset": "explicit" if size else "1080p-if-available", "fps": camera.get("fps", fps)}
        elif kind == "orbbec":
            size = size or (1280, 720)
            exact = bool(camera.get("serial") and camera.get("fps") is not None)
            mode = {"preset": "explicit" if exact else "sdk-default", "fps": camera["fps"] if exact else None}
        else:
            mode = {"preset": "robot-observation", "key": camera.get("key", role)}
        if size:
            mode["size"] = list(map(int, size))
        crop = _crop_box(camera) if kind != "orbbec" else None
        result[role] = {"mode": {"type": kind, **mode}, "crop": list(map(int, crop)) if crop else None,
                        "save_size": list(map(int, camera["save_size"]))}
    return result


def episode_framing(meta):
    if "camera_framing" in meta:
        return meta["camera_framing"]
    # Older raw episodes recorded the crop and stored size, but not the capture mode.
    return {role: {"save_size": camera["size"], **({"crop": camera["crop"]} if "crop" in camera else {})}
            for role, camera in meta["cameras"].items()}


def dataset_framing(episodes):
    known, common = {}, None
    for path, meta in episodes:
        current = episode_framing(meta)
        for role, settings in current.items():
            seen = known.setdefault(role, {})
            for field, value in settings.items():
                if field in seen and seen[field] != value:
                    raise SystemExit(f"{path.name}: camera {role!r} {field} differs across episodes")
                seen[field] = value
        common = current if common is None else {
            role: {key: value for key, value in settings.items() if key in current.get(role, {})}
            for role, settings in common.items() if role in current
        }
    return common or {}


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def read_framing(path):
    data = _read_json(path)
    if data is None:
        return None
    if not isinstance(data, dict) or any(
        not isinstance(settings, dict) or not settings.keys() <= {"mode", "crop", "save_size"}
        for settings in data.values()
    ):
        raise ValueError(f"Invalid camera framing metadata: {path}")
    return data


def write_framing(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def training_framing(checkpoint: Path, data_root: Path):
    """Read the checkpoint sidecar, or the recorded dataset's locally available metadata."""
    if (expected := read_framing(checkpoint / FRAMING_FILE)) is not None:
        return expected
    training = _read_json(checkpoint / "train_config.json") or {}
    dataset = training.get("dataset") or {}
    root, repo = dataset.get("root"), dataset.get("repo_id")
    if root:
        root = expand(root)
        roots = [root] if root.is_absolute() else [checkpoint / root, root]
        for candidate in roots:
            if candidate.is_dir():
                return read_framing(candidate / "meta" / FRAMING_FILE)
    # The macbot training scripts use local/<name>. The absolute training-machine path
    # may not exist after the checkpoint is copied back to this Mac.
    parts = repo.split("/") if isinstance(repo, str) else []
    if len(parts) == 2 and all(part and part not in {".", ".."} for part in parts):
        if parts[0] == "local":
            if (local := data_root / "datasets" / parts[1]).is_dir():
                return read_framing(local / "meta" / FRAMING_FILE)
        hf_home = expand(os.environ.get("HF_HOME", "~/.cache/huggingface"))
        cache = expand(os.environ.get("HF_LEROBOT_HOME", hf_home / "lerobot"))
        return read_framing(cache.joinpath(*parts, "meta", FRAMING_FILE))
    return None


def describe(value):
    return "missing" if value is None else json.dumps(value, sort_keys=True, separators=(",", ":"))


def check_framing(expected, current, override=False):
    if not expected:
        print(f"No training camera framing recorded; current framing: {describe(current)}", flush=True)
        return
    for role, settings in expected.items():
        actual = current.get(role)
        if actual is None or any(actual.get(key) != value for key, value in settings.items()):
            message = (f"Camera {role!r} framing mismatch: trained {describe(settings)}; current {describe(actual)}; "
                       "use --allow-framing-mismatch to override.")
            if not override:
                raise SystemExit(message)
            print(f"Warning: {message}", flush=True)
