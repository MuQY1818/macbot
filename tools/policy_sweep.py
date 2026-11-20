"""Measure one generic policy/device per process and save actions for paired CPU/MPS comparison."""
from __future__ import annotations

import argparse
import json
import resource
import sys
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from macbot.config import load
from macbot.policy.generic import GenericPolicy


def comparable_randomness(stack, torch, seed):
    """Use the same CPU normal draws and categorical sampler on both devices for the error pass."""
    generator = torch.Generator().manual_seed(seed)
    randn, multinomial = torch.randn, torch.multinomial

    def normal(*size, **kwargs):
        device = kwargs.pop("device", None)
        kwargs["generator"] = generator
        return randn(*size, **kwargs, device="cpu").to(device)

    def normal_like(value):
        return normal(value.shape, dtype=value.dtype, device=value.device)

    def categorical(probs, *args, **kwargs):
        kwargs["generator"] = generator
        return multinomial(probs.float().cpu(), *args, **kwargs).to(probs.device)

    stack.enter_context(patch.object(torch, "randn", normal))
    stack.enter_context(patch.object(torch, "randn_like", normal_like))
    stack.enter_context(patch.object(torch, "multinomial", categorical))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("-n", type=int, default=30)
    args = parser.parse_args()
    cfg = load(args.config)
    camera_sizes = {k: v["save_size"] for k, v in cfg["cameras"].items()}
    print("Loading checkpoint", flush=True)
    p = GenericPolicy(args.checkpoint, args.device, cfg["task"]["instruction"], cameras=camera_sizes,
                      camera_map=cfg.get("policy", {}).get("camera_map", {}))
    torch = p.torch
    torch.set_num_threads(4)
    if args.device == "cpu":
        p.pol.float()
        for key in ("dtype", "use_bf16", "use_amp"):
            if hasattr(p.pol.config, key):
                setattr(p.pol.config, key, torch.float32 if key == "dtype" else False)
    rng = np.random.default_rng(124)
    observations = [
        ({c: rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for c, (w, h) in camera_sizes.items()},
         rng.standard_normal(p.width, dtype=np.float32))
        for _ in range(max(args.warmup, args.n))
    ]

    print("Warming up", flush=True)
    for imgs, q in observations[:args.warmup]:
        p(imgs, q)
    p.reset()
    samples = []
    for imgs, q in observations[:args.n]:
        t = perf_counter()
        p(imgs, q)
        samples.append((perf_counter() - t) * 1e3)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss if sys.platform == "darwin" else rss * 1024
    p.reset()
    np.random.seed(444)
    torch.manual_seed(444)
    with ExitStack() as stack:
        comparable_randomness(stack, torch, 444)
        output = [p(imgs, q) for imgs, q in observations[:args.n]]
    np.save(args.out, np.stack(output))
    print(json.dumps({
        "policy": p.kind, "device": args.device, "execution": "step" if p.stepwise else f"chunk {p.H}",
        "calls": args.n, "warmup": args.warmup, "median_ms": float(np.median(samples)),
        "p90_ms": float(np.percentile(samples, 90)), "peak_rss_gib": rss / 2**30,
        "driver_allocated_gib": torch.mps.driver_allocated_memory() / 2**30 if args.device == "mps" else None,
    }))


if __name__ == "__main__":
    main()
