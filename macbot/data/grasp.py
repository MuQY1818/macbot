"""Detect empty closes using calibrated gripper plateaus, without opening camera data."""
from __future__ import annotations

from zipfile import ZipFile

import numpy as np

from .rollouts import read_array


def gripper_index(meta):
    names = meta.get("joint_names") or []
    name = meta.get("gripper") or "gripper"
    return names.index(name) if name in names else None


def schmitt(values, close_above, open_below, *, initial=False):
    """Binary labels with hysteresis, shared with demonstration export."""
    out = np.empty(len(values), np.uint8)
    closed = initial
    for i, value in enumerate(values):
        if not closed and value > close_above:
            closed = True
        elif closed and value < open_below:
            closed = False
        out[i] = closed
    return out


def settings(cfg):
    ex = cfg["export"]
    return {**cfg["grasp_miss"], "close_above": ex["gripper_close_above"], "open_below": ex["gripper_open_below"],
            **({"profile": cfg["robot"]["grasp_miss"]} if "grasp_miss" in cfg["robot"] else {})}


def profile(meta, command, cfg):
    if "profile" in cfg:
        return cfg["profile"]
    if "grasp_miss_profile" in meta:
        return meta["grasp_miss_profile"]
    if (meta.get("robot") or {}).get("kind") != "rebot_b601" or meta.get("fake"):
        return None
    binary = meta.get("binary_gripper")
    # The built-in profile is for this soft setup only. Older files identify it
    # by the two commanded endpoints, since they did not save the binary config.
    if isinstance(binary, dict):
        soft = abs(binary["open_deg"] + 120) < 0.5 and abs(binary["close_deg"]) < 0.5
    else:
        soft = binary is None and np.any(command == -120) and np.any(command == 0)
    return cfg.get("rebot_soft") if soft else None


def detect(meta, t, command, measured, cfg):
    """Return miss spans (exclusive end frame) and the number of inconclusive closes."""
    unknown = {"misses": [], "miss_count": None, "unknown_closes": None}
    gi = gripper_index(meta)
    if (gi is None or len(t) < 2 or command.ndim != 2 or measured.shape != command.shape
            or len(t) != len(command) or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0)):
        return unknown
    command, measured = command[:, gi], measured[:, gi]
    if not np.all(np.isfinite(command)):
        return unknown
    p = profile(meta, command, cfg)
    if not p:
        return unknown
    try:
        held, empty, tol = (float(p[k]) for k in ("held", "empty", "tolerance"))
        close = float(p.get("close_above", cfg["close_above"]))
        opened = float(p.get("open_below", cfg["open_below"]))
    except (AttributeError, KeyError, TypeError, ValueError):
        return unknown
    if (not np.isfinite([held, empty, tol, close, opened]).all()
            or not 0 < 2 * tol < abs(held - empty) or not opened < close):
        return unknown
    closed = schmitt(command, close, opened, initial=command[0] >= opened)
    changes = np.flatnonzero(np.diff(closed)) + 1
    misses, inconclusive = [], 0
    for first, end in zip(changes, changes[1:], strict=False):
        if not closed[first]:
            continue
        # Inspect a short tail before release, after giving the fingers time to close.
        # Require a stable plateau; travelling, overshoot and partial grabs are unknown.
        last = end - 1
        tail = np.searchsorted(t, max(t[last] - 0.20, t[first] + 0.35) - 1e-9)
        v = measured[tail:end]
        if (len(v) < 3 or t[last] - t[tail] < 0.12 or np.any(np.diff(t[first:end + 1]) > 2.5 / meta["fps"])
                or not np.isfinite(v).all()):
            inconclusive += 1
        elif np.mean(np.abs(v - empty) <= tol) >= 0.85:
            misses.append({"first": int(first), "end": int(end), "t": float(t[first]), "until": float(t[end])})
        elif np.mean(np.abs(v - held) <= tol) < 0.85:
            inconclusive += 1
    return {"misses": misses, "miss_count": len(misses), "unknown_closes": inconclusive}


def from_archive(archive: ZipFile, meta, cfg):
    if gripper_index(meta) is None:
        return {"misses": [], "miss_count": None, "unknown_closes": None}
    return detect(meta, *(read_array(archive, k) for k in ("t", "cmd", "qpos")), cfg)
