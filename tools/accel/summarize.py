"""Render benchmark JSON as a Markdown table with seed-spread calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_PARTS = ("arm_all", "arm_first10", "gripper_all", "gripper_first10")


def pair(metrics, part):
    values = metrics.get(part)
    return f"{values['mean']:.4f} / {values['p99']:.4f}" if values else "-"


def worst_fraction(candidate, seed):
    ratios = []
    for part in candidate.keys() & seed.keys() & set(_PARTS):
        for metric in ("mean", "p99"):
            a, b = candidate[part][metric], seed[part][metric]
            ratios.append(a / b if b else float("inf") if a else 0.0)
    return max(ratios, default=None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-spread", type=Path, required=True)
    p.add_argument("--zh", action="store_true")
    p.add_argument("results", nargs="+", type=Path)
    args = p.parse_args()
    seed_run = json.loads(args.seed_spread.read_text())
    seed = seed_run["fidelity"]
    english = ("Configuration", "Median / p90 (ms)", "Peak RSS / GPU (GiB)", "Arm all (deg)", "Arm first 10 (deg)",
               "Gripper all", "Gripper first 10", "Max fraction of seed spread", "Fidelity")
    chinese = ("配置", "中位数 / p90（ms）", "RSS 峰值 / GPU（GiB）", "臂全段（°）", "臂前 10 步（°）",
               "夹爪全段", "夹爪前 10 步", "误差占跨种子差异的最大比例", "保真判定")
    headers = chinese if args.zh else english
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join([":--"] * len(headers)) + " |")
    for path in args.results:
        r = json.loads(path.read_text())
        time = r["latency_ms"]
        mem = r["memory_bytes"]
        rss = mem["peak_process_rss"] / 2**30
        gpu = mem.get("mlx_timing_peak", mem.get("max_sampled_mps_driver", 0)) / 2**30
        fidelity = r.get("fidelity", {})
        ratio = worst_fraction(fidelity, seed)
        unchanged = r["steps"] == seed_run["steps"] and not (r.get("pool_images") or r.get("bits"))
        if not fidelity:
            verdict = ("Reference", "基线")
        elif path.resolve() == args.seed_spread.resolve() or r.get("dtype") == "fp32":
            verdict = ("Control", "对照")
        elif ratio == 0 and unchanged:
            verdict = ("Exact on replay", "回放逐元素一致")
        elif unchanged and ratio is not None and ratio < 0.5:
            verdict = ("Lossless in practice", "实际保真")
        else:
            verdict = ("Lossy", "有损")
        values = [path.stem.replace("_", " "), f"{time['median']:.2f} / {time['p90']:.2f}", f"{rss:.2f} / {gpu:.2f}",
                  *[pair(fidelity, part) for part in _PARTS], f"{ratio:.0%}" if ratio is not None else "-",
                  verdict[args.zh]]
        print("| " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
