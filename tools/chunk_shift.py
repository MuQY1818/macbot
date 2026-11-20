"""Measure progress regression at chunk switches in recorded rollouts.

    python tools/chunk_shift.py ~/macbot-data/rollouts/<task> [--w 10] [--J 15]

Needs rollouts recorded by a macbot version that saves the chunks at each switch (``ex_chunk``). For each trial and
pooled per executor it prints the shift delta_hat by which the new chunk starts behind the current command
(full window, and after the guided prefix of real-time chunking), the residual r_plan, and the tracking lag
between command and measured arm.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from macbot.data.chunks import switch_shifts, tracking_lag


def summary(x) -> str:
    x = np.asarray(x)
    if not len(x):
        return "n 0"
    return f"n {len(x)} median {np.median(x):.1f} p10 {np.percentile(x, 10):.1f} p90 {np.percentile(x, 90):.1f} >0 {np.mean(x > 0) * 100:.0f}%"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="rollout NPZ files or folders")
    ap.add_argument("--w", type=int, default=10, help="compared actions (default 10)")
    ap.add_argument("--J", type=int, default=15, help="largest searched shift (default 15)")
    args = ap.parse_args(argv)
    files = []
    for p in map(Path, args.paths):
        files += sorted(p.expanduser().glob("rollout_*.npz")) if p.expanduser().is_dir() else [p.expanduser()]
    pooled = defaultdict(lambda: defaultdict(list))
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            if "ex_chunk" not in d.files:
                continue
            meta = json.loads(str(d["meta"]))
            ex = meta.get("executor", "?")
            full, r = switch_shifts(d, w=args.w, J=args.J)
            after, _ = switch_shifts(d, w=args.w, J=args.J, after_guide=True)
            lag = tracking_lag(d["t"], d["cmd"][:, :6], d["qpos"][:, :6], meta.get("fps", 30))
        print(f"{f.name} [{ex}] lag {lag} frames | delta_hat {summary(full)} | after guide {summary(after)}"
              f" | r_plan median {np.median(r) if len(r) else float('nan'):.3f}")
        for key, v in (("full", full), ("after", after), ("r", r)):
            pooled[ex][key] += list(v)
        if lag is not None:
            pooled[ex]["lag"].append(lag)
    for ex, v in pooled.items():
        print(f"\n{ex}: lag median {np.median(v['lag']) if v['lag'] else float('nan'):.1f} frames")
        print(f"  delta_hat (full window):  {summary(v['full'])}")
        print(f"  delta_hat (after guide):  {summary(v['after'])}")
        print(f"  r_plan median {np.median(v['r']) if v['r'] else float('nan'):.3f}")
    if not pooled:
        print("no rollouts with recorded chunks (ex_chunk)")


if __name__ == "__main__":
    main()
