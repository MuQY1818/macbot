"""Per-episode tracking and timing report of recorded data.

For every episode: the lag (frames) that best aligns command and measurement, p95 and max tracking error
of the arm joints at that lag, the median offset while the leader is still (a calibration drift
shows up here), the gripper range, the worst frame interval and the repeated camera frames.
``--split N`` prints medians before and after episode N, for example to compare data recorded before
and after a repair.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from ..config import Config
from ..system import expand
from .grasp import detect, settings
from .rollouts import metadata, rollout_files


def episode_stats(d, fps: float | None = None, cams=None, *, rollout_meta=None) -> dict:
    meta = rollout_meta if rollout_meta is not None else json.loads(str(d["meta"]))
    fps = fps if fps is not None else float(meta["fps"])
    cams = cams if cams is not None else meta["cameras"]
    q, a = d["qpos"].astype(float), d["cmd" if rollout_meta is not None else "action"].astype(float)
    names = meta.get("joint_names") or [f"joint_{i}" for i in range(q.shape[1])]
    g = names.index(meta["gripper"]) if meta.get("gripper") in names else (names.index("gripper") if "gripper" in names else None)
    arm = [i for i in range(q.shape[1]) if i != g] or list(range(q.shape[1]))
    best = None
    for lag in range(min(8, len(q))):
        err = np.abs(a[:len(a) - lag][:, arm] - q[lag:][:, arm])
        m = np.percentile(err, 95, axis=0).mean()
        if best is None or m < best[0]:
            best = (m, lag, err)
    _, lag, err = best
    still = np.abs(np.diff(a[:, arm], axis=0)).max(1) * fps < 2.0
    off = (a[1:][:, arm] - q[1:][:, arm])[still]
    wt = d["t"] if rollout_meta is not None else d["wall_time"] if "wall_time" in d.files else d["timestamp"]
    dt = np.diff(wt) if len(wt) > 1 else np.zeros(1)
    return {"n": len(q), "lag": lag, "p95": np.percentile(err, 95, axis=0), "max": err.max(0),
            "still": np.median(off, 0) if len(off) else np.full(len(arm), np.nan),
            "vmax": np.abs(np.diff(q[:, arm], axis=0)).max(0, initial=0) * fps,
            "gmin": q[:, g].min() if g is not None else np.nan, "gmax": q[:, g].max() if g is not None else np.nan, "dt_max": dt.max(), "dt_late": int((dt > 1.5 / fps).sum()),
            "repeat": {c: int(d[f"{c}_repeat"].sum()) for c in cams if f"{c}_repeat" in d.files},
            "nan": int(np.isnan(q).sum() + np.isnan(a).sum())}


def run(cfg: Config, args):
    data = expand(args.data) if args.data else cfg.raw_dir()
    files = sorted(data.glob("episode_*.npz"))
    rollouts = not files and args.data is not None
    if rollouts:
        files = sorted(rollout_files(data))
    if not files:
        sys.exit(f"no {'episode_*.npz or rollout_*.npz' if rollouts else 'episode_*.npz'} in {data}")
    np.set_printoptions(precision=1, suppress=True, linewidth=200)
    rows = []
    grasp_cfg = settings(cfg) if rollouts else None
    excluded = 0
    print(("run " if rollouts else "ep  ") +
          " frames lag  p95 tracking error per arm joint     offset while still               gripper range    dt max  late  repeats" +
          ("  misses  result / note" if rollouts else ""))
    for f in files:
        idx = int(f.stem.split("_")[-1])
        meta = metadata(f) if rollouts else None
        if meta and meta.get("exclude"):
            print(f"{idx:03d} excluded: {meta['exclude_reason']}")
            excluded += 1
            continue
        with np.load(f) as d:
            r = episode_stats(d, rollout_meta=meta)
            if rollouts:
                report = detect(meta, d["t"], d["cmd"], d["qpos"], grasp_cfg)
                r["result"] = meta.get("result")
                count = report["miss_count"]
                misses = "unknown" if count is None else str(count)
                if report["unknown_closes"]:
                    misses += f" (+{report['unknown_closes']} unknown)"
        r["idx"] = idx
        rows.append(r)
        rep = "/".join(str(v) for v in r["repeat"].values())
        print(f"{idx:03d} {r['n']:6d} {r['lag']:3d}  {r['p95']}  {r['still']}  {r['gmin']:6.1f}/{r['gmax']:6.1f}"
              f"  {r['dt_max'] * 1e3:5.0f}ms {r['dt_late']:4d}  {rep}" + (f"  NaN {r['nan']}" if r["nan"] else "") +
              (f"  {misses}  {meta.get('result', '–')}  {meta.get('note', '')}".rstrip() if meta is not None else ""))
    if rollouts:
        reviewed = [r["result"] for r in rows if r["result"] in ("success", "failure")]
        print(f"\n{len(rows)} included, {excluded} excluded; {reviewed.count('success')}/{len(reviewed)} successful trials")
    if args.split is not None:
        a = [r for r in rows if r["idx"] < args.split]
        b = [r for r in rows if r["idx"] >= args.split]
        if not a or not b:
            sys.exit(f"--split {args.split} leaves one side empty")
        print(f"\nmedian over episodes, before {args.split} ({len(a)}) vs from {args.split} ({len(b)}):")
        for label, k in [("p95 tracking", "p95"), ("max error", "max"), ("still offset", "still"), ("max speed", "vmax")]:
            print(f"  {label:13s} {np.median([r[k] for r in a], 0)}  ->  {np.median([r[k] for r in b], 0)}")
        for label, k in [("gripper min", "gmin"), ("gripper max", "gmax")]:
            print(f"  {label:13s} {np.median([r[k] for r in a]):7.1f}  ->  {np.median([r[k] for r in b]):7.1f}")
