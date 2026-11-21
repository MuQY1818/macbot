"""Time policy inference on this Mac with blank camera images of the configured sizes."""
from __future__ import annotations

import json
import resource
import sys
import time
from contextlib import ExitStack

import numpy as np

from ..config import Config
from ..system import expand
from .backends import select_backend
from .models import load_generic, load_mapped_mlx, load_policy, policy_backend, raw_state_shape, use_generic
from .profile import MLXStages, TorchStages


def run(cfg: Config, args):
    path = expand(args.policy)
    metadata = json.loads((path / "config.json").read_text())
    state = raw_state_shape(path, metadata) or [7]
    q = [0.0] * state[0]
    selection = select_backend(policy_backend(cfg, args), path, metadata, args, bench=True)
    selection.announce()
    backend = selection.backend
    kind = metadata.get("type", "act")
    generic = use_generic(kind, getattr(args, "policy_impl", "auto"), backend)
    mapped_mlx = backend == "mlx" and kind != "pi05"
    steps = getattr(args, "steps", None)
    if generic:
        pol = load_generic(path, cfg, args)
        q = [0.0] * pol.width
    elif mapped_mlx:
        pol = load_mapped_mlx(path, cfg, args)
        q = [0.0] * pol.width
    elif backend == "mlx":
        pol = load_policy(path, args.device, cfg["task"]["instruction"], steps, backend="mlx",
                          mlx_bits=args.mlx_bits, mlx_quantize=args.mlx_quantize, mlx_time_grid=args.mlx_time_grid)
    elif steps is not None:
        pol = load_policy(path, args.device, cfg["task"]["instruction"], steps)
    else:
        pol = load_policy(path, args.device, cfg["task"]["instruction"])
    if (mapped_mlx or not generic and pol.kind == "pi05") and not args.no_compile:
        pol.enable_compile()
        print(f"{'mx.compile' if backend == 'mlx' else 'torch.compile'} on; the first calls compile", flush=True)
    imgs = {c: np.zeros((v["save_size"][1], v["save_size"][0], 3), np.uint8) for c, v in cfg["cameras"].items()}
    mode = getattr(args, "mode", "auto")
    native = "step" if generic and pol.stepwise else "chunk"
    call_kwargs = {}
    if mode == "rtc":
        if not pol.supports_rtc:
            sys.exit(f"{pol.kind} on {backend} does not support --mode rtc")
        pol.enable_rtc(10)
        leftover = pol(imgs, q, return_raw=True)[1]
        if len(leftover) < 2:
            sys.exit("--mode rtc needs at least two actions in the chunk")
        call_kwargs = dict(leftover=leftover, delay=min(4, len(leftover) - 1), horizon=20)
    elif mode not in {"auto", native}:
        sys.exit(f"{pol.kind} on {backend} requires --mode {native} (or auto)")
    for _ in range(5):
        pol(imgs, q, **call_kwargs)
    if generic:
        pol.reset()
    if backend == "mlx":
        import mlx.core as mx

        mx.reset_peak_memory()
    ts = []
    for _ in range(args.n):
        t = time.perf_counter()
        pol(imgs, q, **call_kwargs)
        ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts)
    engine = "MLX" if backend == "mlx" else args.device
    execution = "step" if native == "step" else f"{'RTC chunk' if mode == 'rtc' else 'chunk'} {pol.H}"
    precision = 2 if generic else 0
    print(f"{pol.kind} on {engine}, {execution}: median {np.median(ts):.{precision}f} ms, "
          f"p90 {np.percentile(ts, 90):.{precision}f} ms, min {ts.min():.{precision}f} ms over {args.n} calls")
    if backend == "mlx":
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss *= 1024
        print(f"Peak process RSS: {rss / 2**30:.2f} GiB; MLX allocation peak: {mx.get_peak_memory() / 2**30:.2f} GiB")
        stages = MLXStages(mx)
        for _ in range(args.n):
            pol(imgs, q, stage=stages, **call_kwargs)
    elif generic:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss *= 1024
        print(f"Peak process RSS: {rss / 2**30:.2f} GiB")
        if not getattr(args, "profile", False):
            return
        stages = TorchStages(pol.torch.mps.synchronize if args.device == "mps" else lambda: None)
        pol.reset()
        for _ in range(args.n):
            pol(imgs, q, stage=stages)
    elif pol.kind == "pi05" and getattr(args, "profile", False):
        sync = pol.torch.mps.synchronize if args.device == "mps" else lambda: None
        stages = TorchStages(sync)
        with ExitStack() as stack:
            stages.instrument(pol, stack)
            for _ in range(args.n):
                stages.counts.clear()
                stages.wrap(pol, "profile_total")(imgs, q, **call_kwargs)
    else:
        return
    if stages.samples:
        print("Synchronized stages (median / p90 ms):")
        for name, values in stages.samples.items():
            print(f"  {name:36s} {np.median(values):7.2f} / {np.percentile(values, 90):.2f}")
