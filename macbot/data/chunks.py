"""Progress regression at chunk switches, measured from the chunks a rollout recorded.

A position-controlled arm lags its command, and a policy that plans from the measured arm can return a new chunk
that starts behind the current command. Aligning the new chunk with the current one estimates that shift:

    D(s) = mean_i || new[start + i] - cur[p - s + start + i] ||,   |s| <= J
    delta_hat = argmin_s D(s)   (ties: smallest |s|, then the positive shift),   r_plan = D(delta_hat)

``p`` is the index of the current chunk that the new chunk's index 0 replaces. ``delta_hat > 0`` means the new
chunk starts behind the current command. With ``start = 0`` this is the alignment of the progress-aligned
switching paper; ``start = delay`` skips the prefix that real-time chunking guides onto the current chunk.
"""
from __future__ import annotations

import numpy as np


def shifts(J: int):
    """0, 1, -1, 2, -2, ...: the order that implements the tie rule with a strict comparison."""
    yield 0
    for s in range(1, J + 1):
        yield s
        yield -s


def align(new, cur, p: int, *, w: int = 10, J: int = 15, start: int = 0, scale=None):
    """Return ``(delta_hat, r_plan)`` for one switch, or ``None`` if no shift can be compared."""
    new, cur = np.asarray(new, np.float64), np.asarray(cur, np.float64)
    if scale is not None:
        new, cur = new / scale, cur / scale
    valid = ~np.isnan(cur).any(axis=1)
    H = int(valid.nonzero()[0].max()) + 1 if valid.any() else 0
    best = None
    for s in shifts(J):
        c0 = p - s + start                                    # current-chunk index compared with new[start]
        if not 0 <= p - s < H or c0 >= H:
            continue
        h = min(w, H - c0, len(new) - start)
        if h <= 0:
            continue
        d = float(np.linalg.norm(new[start:start + h] - cur[c0:c0 + h], axis=1).mean())
        if best is None or d < best[1]:
            best = (s, d)
    return best


def tracking_lag(t, cmd, q, fps: float, max_lag: int = 10, min_speed: float = 5.0):
    """Frames by which the measured arm trails its command: the shift minimizing |cmd[t-l] - q[t]| while moving."""
    cmd, q = np.asarray(cmd, np.float64), np.asarray(q, np.float64)
    speed = np.linalg.norm(np.gradient(cmd, np.asarray(t), axis=0), axis=1)
    moving = speed > min_speed
    if moving.sum() < 2 * max_lag:
        return None
    err = [np.linalg.norm(cmd[:len(cmd) - lag] - q[lag:], axis=1)[moving[lag:]].mean() for lag in range(max_lag + 1)]
    return int(np.argmin(err))


def switch_shifts(d, *, w: int = 10, J: int = 15, after_guide: bool = False, arm: int = 6):
    """delta_hat and r_plan for every recorded switch of one rollout NPZ (``ex_chunk``/``ex_prev``/``ex_p``)."""
    if "ex_chunk" not in d.files:
        return np.zeros(0), np.zeros(0)
    new, cur, p = d["ex_chunk"][..., :arm], d["ex_prev"][..., :arm], d["ex_p"]
    delay = d["ex_delay"] if "ex_delay" in d.files and len(d["ex_delay"]) == len(p) else np.zeros(len(p))
    scale = np.asarray(d["cmd"], np.float64)[:, :arm].std(axis=0)
    scale[scale < 1e-6] = 1.0
    out = []
    for i in range(len(p)):
        if np.isnan(p[i]):
            continue
        r = align(new[i], cur[i], int(round(p[i])), w=w, J=J, start=int(delay[i]) if after_guide else 0, scale=scale)
        if r is not None:
            out.append(r)
    out = np.asarray(out, np.float64).reshape(-1, 2)
    return out[:, 0], out[:, 1]
