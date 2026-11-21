"""Shared bookkeeping for seeded simulation batches."""
from __future__ import annotations

import json
import math
from statistics import NormalDist, fmean


def seeds(args):
    if args.episodes < 1 or args.seed < 0:
        raise SystemExit("--episodes must be positive and --seed must be nonnegative")
    return range(args.seed, args.seed + args.episodes)


def summary(rows, fps):
    n = len(rows)
    wins = sum(row["success"] for row in rows)
    fraction = wins / n
    z = NormalDist().inv_cdf(0.975)
    d = 1 + z**2 / n
    center = (fraction + z**2 / (2 * n)) / d
    half = z * math.sqrt(fraction * (1 - fraction) / n + z**2 / (4 * n**2)) / d
    length = fmean(row["steps"] for row in rows)
    return {"episodes": n, "successes": wins, "success_rate": fraction,
            "wilson95": [max(0.0, center - half), min(1.0, center + half)],
            "mean_steps": length, "mean_sim_seconds": length / fps}


def write_summary(path, rows, fps, **metadata):
    stats = summary(rows, fps)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps({**metadata, **stats, "results": rows}, indent=2) + "\n")
    tmp.replace(path)
    lo, hi = stats["wilson95"]
    print(f"success {stats['successes']}/{stats['episodes']} ({stats['success_rate']:.1%}, "
          f"95% Wilson {lo:.1%} to {hi:.1%}); mean {stats['mean_steps']:.1f} steps; {path}", flush=True)


def result_path(out, stem):
    i = 0
    while (path := out / f"{stem}_{i:03d}.json").exists():
        i += 1
    return path
