"""Quality checks run on every saved episode."""
from __future__ import annotations

import numpy as np

from ..i18n import tr


def check_episode(d: dict, cams, fps: int, checks: dict, min_s: float, joints=None, gripper=None) -> dict:
    """``d`` holds arrays ``qpos``, ``late`` and ``<cam>_age`` / ``<cam>_repeat`` per camera.
    Returns ``{ok, errors, warnings, stats}``; errors block export, warnings are shown."""
    n = len(d["qpos"])
    errors, warnings, stats = [], [], {}
    if n / fps < min_s:
        errors.append(tr(f"only {n / fps:.1f} s", f"只有 {n / fps:.1f} 秒"))
    for k in cams:
        age95 = float(np.percentile(d[f"{k}_age"], 95)) if n else 0.0
        rep = float(np.mean(d[f"{k}_repeat"])) if n else 0.0
        stats[f"{k}_age_p95_ms"] = round(age95 * 1e3, 1)
        stats[f"{k}_repeat_ratio"] = round(rep, 3)
        if age95 > checks["max_cam_age_s"]:
            warnings.append(tr(f"{k} frame age p95 {age95 * 1e3:.0f} ms", f"{k} 画面延迟 p95 {age95 * 1e3:.0f} ms"))
        if rep > checks["max_repeat_ratio"]:
            warnings.append(tr(f"{k} repeated frames {rep:.0%}", f"{k} 画面重复 {rep:.0%}"))
    if n > 1:
        q = np.asarray(d["qpos"])
        names = list(joints) if joints else [f"joint_{i}" for i in range(q.shape[1])]
        arm = [i for i, j in enumerate(names) if j != gripper]
        step = np.abs(np.diff(q[:, arm], axis=0)).max(axis=0)
        worst = names[arm[int(step.argmax())]]
        stats["max_joint_step"] = round(float(step.max()), 1)
        if step.max() > checks["max_joint_step_deg"]:
            warnings.append(tr(f"{worst} jumped {step.max():.0f} in one frame", f"{worst} 一帧跳 {step.max():.0f}"))
    late = float(np.mean(d["late"])) if n else 0.0
    stats["late_ratio"] = round(late, 3)
    if late > checks["max_late_ratio"]:
        warnings.append(tr(f"late samples {late:.0%}", f"采样迟到 {late:.0%}"))
    return {"ok": not errors, "errors": errors, "warnings": warnings, "stats": stats}
