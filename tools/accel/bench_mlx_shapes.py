"""Measure warm MLX projections at the PI05 two-view replay shapes (synthetic inputs)."""

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import mlx.core as mx

from .eval_fidelity import stats


def measure(op, calls):
    for _ in range(5):
        mx.eval(op())
    samples = []
    for _ in range(calls):
        start = perf_counter_ns()
        mx.eval(op())
        samples.append((perf_counter_ns() - start) / 1e6)
    return stats(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=30)
    parser.add_argument("--tokens", type=int, default=557)
    args = parser.parse_args()
    if args.calls < 30 or args.tokens < 1:
        parser.error("Expected at least 30 calls and a positive token count")
    shapes = {
        "prefix": [(2048, 2560), (2048, 2048), (2048, 32768), (16384, 2048), (2048, 256)],
        "vision": [(1152, 3456), (1152, 1152), (1152, 4304), (4304, 1152)],
        "expert": [(1024, 2560), (2048, 1024), (1024, 8192), (4096, 1024)],
        "control": [(4096, 4096)],
    }
    rows = []
    mx.random.seed(19)
    for group, pairs in shapes.items():
        lengths = {
            "prefix": sorted({args.tokens, *((args.tokens + a - 1) // a * a for a in (32, 64, 128))}),
            "vision": [512], "expert": [50, 64], "control": [4096],
        }[group]
        for k, n in pairs:
            for dtype in (mx.float16, mx.bfloat16):
                w = mx.random.normal((n, k), dtype=dtype)
                for layout, b in (("NT", w.T), ("NN", mx.contiguous(w.T))):
                    mx.eval(b)
                    for m in lengths:
                        a = mx.random.normal((m, k), dtype=dtype)
                        mx.eval(a)
                        timing = measure(lambda a=a, b=b: a @ b, args.calls)
                        rows.append(dict(group=group, dtype=str(dtype), layout=layout, m=m, k=k, n=n,
                                         ms=timing, tflops=2 * m * k * n / (1e9 * timing["median"])))
    args.out.write_text(json.dumps({"device": mx.device_info(), "version": mx.__version__,
                                    "calls": args.calls, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
