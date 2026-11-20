"""Time policy inference on this Mac with blank camera images of the configured sizes."""
from __future__ import annotations

import time

import numpy as np

from ..config import Config
from ..system import expand
from .models import load_policy


def run(cfg: Config, args):
    pol = load_policy(expand(args.policy), args.device, cfg["task"]["instruction"])
    if pol.kind == "pi05" and not args.no_compile:
        pol.enable_compile()
        print("torch.compile on; the first calls compile (~25 s)", flush=True)
    imgs = {c: np.zeros((v["save_size"][1], v["save_size"][0], 3), np.uint8) for c, v in cfg["cameras"].items()}
    for _ in range(3):
        pol(imgs, [0.0] * 7)
    ts = []
    for _ in range(args.n):
        t = time.perf_counter()
        pol(imgs, [0.0] * 7)
        ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts)
    print(f"{pol.kind} on {args.device}, chunk {pol.H}: median {np.median(ts):.0f} ms, "
          f"p90 {np.percentile(ts, 90):.0f} ms, min {ts.min():.0f} ms over {args.n} calls")
