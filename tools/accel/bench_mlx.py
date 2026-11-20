"""Check stage parity against PyTorch traces, then benchmark and evaluate MLX PI05."""

from __future__ import annotations

import argparse
import json
import resource
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from macbot.policy.profile import MLXStages

from .eval_fidelity import load_torch, metrics, observations, stats


def noise_for(seed, count, config):
    shape = (count, 1, config.chunk_size, config.max_action_dim)
    return np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)


def capture_torch(args, records):
    import torch

    from macbot.policy.models import _to_obs

    policy, _ = load_torch(args, records[0][2])
    policy.enable_compile()
    m = policy.pol.model
    v = m.paligemma_with_expert
    noise = noise_for(args.seed, len(records), m.config)
    output = {}
    for i in args.trace_indices:
        imgs, state, policy.task = records[i]
        trace = {}

        def take(fn, name, extract, trace=trace):
            def wrapper(*a, **kw):
                result = fn(*a, **kw)
                if name not in trace:
                    trace[name] = extract(result)
                return result

            return wrapper

        with ExitStack() as stack, torch.inference_mode(), torch.autocast("mps", dtype=torch.bfloat16):
            for obj, attr, key, extract in (
                (v, "embed_image", "vision", lambda x: x),
                (v, "forward", "prefix", lambda x: x[1]),
                (m, "denoise_step", "velocity", lambda x: x),
            ):
                stack.enter_context(patch.object(obj, attr, take(getattr(obj, attr), key, extract)))
            obs = policy.pre(_to_obs(torch, imgs, state, policy.task, policy.rename))
            action = policy.post(policy.pol.predict_action_chunk(obs, noise=torch.from_numpy(noise[i])))
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK

        tokens = obs[OBS_LANGUAGE_ATTENTION_MASK].cpu().numpy()[0]
        valid = np.r_[np.ones(np.prod(trace["vision"].shape[:2]), dtype=bool), tokens]
        for name, tensor in (("vision", trace["vision"]), ("velocity", trace["velocity"]), ("actions", action)):
            output[f"{i}_{name}"] = tensor.float().cpu().numpy()
        for layer, (k, v_, _) in enumerate(trace["prefix"]):
            output[f"{i}_k{layer}"] = k.float().cpu().numpy()[:, :, valid]
            output[f"{i}_v{layer}"] = v_.float().cpu().numpy()[:, :, valid]
        print(f"Captured reference stages for observation {i}", flush=True)
    np.savez_compressed(args.out, **output)


def compare_trace(policy, records, noise, args):
    import mlx.core as mx

    def compare(reference, candidate):
        a, b = reference.astype(np.float32), candidate.astype(np.float32)
        if a.shape != b.shape or not np.isfinite(b).all():
            raise ValueError(f"Stage shape/nonfinite mismatch: {a.shape}, {b.shape}")
        delta = np.abs(a - b)
        return {
            "mean_abs": float(delta.mean()),
            "p99_abs": float(np.percentile(delta, 99)),
            "relative_rms": float(np.linalg.norm(delta) / max(np.linalg.norm(a), 1e-12)),
        }

    def to_np(tensor):
        return np.array(tensor.astype(mx.float32))

    result = {}
    with np.load(args.trace) as ref, policy.torch.inference_mode():
        for i in args.trace_indices:
            images, state, policy.task = records[i]
            pixels, tokens, mask, all_valid = policy.prepare(images, state)
            v = policy.vision(pixels)
            valid, cache = policy.prefix(policy.pool(v), tokens, mask, not all_valid)
            pos, keep = policy.network.expert_inputs(valid, noise[i].shape[1], masked=not all_valid)
            cond, modulation = policy.velocity_args(0)
            speed = policy.velocity(mx.array(noise[i]), cond, cache, pos, keep, modulation)
            action = policy(images, state, noise=noise[i])
            result[i] = {
                "vision": compare(ref[f"{i}_vision"], to_np(v)),
                "velocity": compare(ref[f"{i}_velocity"], to_np(speed)),
                "action": compare(ref[f"{i}_actions"], action[None]),
            }
            keep = np.array(valid)[0]
            if policy.pool_images or args.keep_visual_fraction is not None:
                result[i]["kv_comparison"] = "Visual token reduction changes cache length."
                continue
            kv = [
                compare(ref[f"{i}_{name}{j}"], to_np(tensor)[:, :, keep])
                for j, layer in enumerate(cache) for name, tensor in zip(("k", "v"), layer, strict=True)
            ]
            result[i]["kv_worst_relative_rms"] = max(k["relative_rms"] for k in kv)
            result[i]["kv_by_layer"] = kv
    return result


def run(args, records, joints):
    import mlx.core as mx

    from macbot.policy.mlx_pi05 import PI05MLX

    if len(records) < 50 or args.calls < 30 or args.warmup < 1:
        raise ValueError("Need at least 50 observations, 30 timed calls and one warm-up call")
    mx.reset_peak_memory()
    policy = PI05MLX(
        args.checkpoint, records[0][2], steps=args.steps, bits=args.bits, quantize=args.quantize, trim=args.trim,
        fused_attention=not args.unfused_attention, fused_rope=args.fused_rope,
        dtype={"bf16": mx.bfloat16, "fp16": mx.float16}[args.precision], precompute_adarms=args.precompute_adarms,
        pack_linears=args.pack_linears, gelu_fp32=args.gelu_fp32, pool_images=args.pool_images,
        time_grid="quadratic" if args.solver == "quadratic" else "uniform",
    )
    load_peak = mx.get_peak_memory()
    if args.keep_visual_fraction is not None or args.solver in ("ab2", "midpoint"):
        from .mlx_variants import install_pruning, install_solver

        if args.keep_visual_fraction is not None:
            install_pruning(policy, args.keep_visual_fraction)
        if args.solver in ("ab2", "midpoint"):
            install_solver(policy, args.solver)
    if args.compile:
        policy.enable_compile()
    noise = noise_for(args.seed, len(records), policy.config)
    trace = compare_trace(policy, records, noise, args)
    args.out.with_suffix(".stages.json").write_text(json.dumps(trace, indent=2) + "\n")
    for i, trace_i in trace.items():
        print(f"Stage comparison {i}: " + json.dumps({k: v for k, v in trace_i.items() if k != "kv_by_layer"}), flush=True)
    stage_drift = any(
        t["vision"]["relative_rms"] > 0.15 or t.get("kv_worst_relative_rms", 0) > 0.25
        or t["velocity"]["relative_rms"] > 0.25
        for t in trace.values()
    )
    if stage_drift and not (args.pool_images or args.keep_visual_fraction is not None
                            or (args.bits and args.measure_lossy_drift)):
        raise ValueError("Stage error exceeds the port sanity limits; latency was not measured")
    if stage_drift:
        print("Stage drift exceeds the port sanity limits; measuring an explicitly lossy configuration.", flush=True)

    def call(i, **kw):
        imgs, q, policy.task = records[i % len(records)]
        return policy(imgs, q, **kw)

    for i in range(args.warmup):
        call(i)
    extra_warmup = 0
    if args.trim and args.compile:
        seen_shapes = set()
        with policy.torch.inference_mode():
            for i, (imgs, state, policy.task) in enumerate(records):
                images, tokens, _, all_valid = policy.prepare(imgs, state)
                key = (images.shape, tokens.shape, all_valid)
                if key not in seen_shapes and i >= args.warmup:
                    call(i)
                    extra_warmup += 1
                seen_shapes.add(key)
    times = []
    mx.reset_peak_memory()
    for i in range(args.calls):
        mx.synchronize()
        start = perf_counter()
        call(i)
        mx.synchronize()
        times.append((perf_counter() - start) * 1e3)
    latency = {**stats(times), "calls": len(times), "samples": times}
    memory = {
        "peak_process_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mlx_load_peak": load_peak, "mlx_timing_peak": mx.get_peak_memory(),
    }
    print(f"Timing complete: {stats(times)}", flush=True)
    actions = np.stack([call(i, noise=noise[i]) for i in range(len(records))])
    gripper = args.checkpoint / "gripper.json"
    binary = gripper.exists() and json.loads(gripper.read_text()).get("mode") == "binary"
    result = {
        "engine": "mlx", "version": mx.__version__, "steps": policy.steps, "bits": args.bits,
        "quantize": args.quantize, "trim": args.trim, "seed": args.seed, "precision": args.precision,
        "compile": args.compile, "fused_rope": args.fused_rope, "unfused_attention": args.unfused_attention,
        "precompute_adarms": args.precompute_adarms,
        "pack_linears": args.pack_linears, "gelu_fp32": args.gelu_fp32, "pool_images": args.pool_images,
        "stage_drift": stage_drift, "solver": args.solver, "keep_visual_fraction": args.keep_visual_fraction,
        "count": len(records), "warmup": args.warmup, "extra_shape_warmup": extra_warmup,
        "latency_ms": latency, "memory_bytes": memory,
        "fidelity": metrics(actions, np.load(args.reference), joints, binary),
    }
    if args.profile:
        stages = MLXStages(mx)
        for i in range(args.calls):
            call(i, stage=stages)
        result["profile_ms"] = {k: {**stats(v), "calls": len(v)} for k, v in stages.samples.items()}
    np.save(args.out.with_suffix(".npy"), actions)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--observations", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--trace", type=Path)
    p.add_argument("--capture-torch", action="store_true")
    p.add_argument("--trace-indices", type=int, nargs="+", default=[0, 24, 49])
    p.add_argument("--calls", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=941)
    p.add_argument("--steps", type=int)
    p.add_argument("--solver", choices=["euler", "quadratic", "midpoint", "ab2"], default="euler",
                   help="offline flow experiment; --steps is the velocity evaluation budget")
    p.add_argument("--keep-visual-fraction", type=float, help="lossy attention-ranked fraction kept per camera")
    p.add_argument("--bits", type=int, choices=[4, 8])
    p.add_argument("--quantize", choices=["prefix", "expert", "both"], default="prefix")
    p.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--trim", action="store_true")
    p.add_argument("--fused-rope", action="store_true")
    p.add_argument("--unfused-attention", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--precompute-adarms", action="store_true")
    p.add_argument("--pack-linears", action="store_true")
    p.add_argument("--gelu-fp32", action="store_true")
    p.add_argument("--pool-images", action="store_true", help="lossy 2x2 pooling of projected vision tokens")
    p.add_argument("--measure-lossy-drift", action="store_true",
                   help="time quantized experiments after recording a failed numerical stage check")
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records, joints = observations(args.observations)
    if args.capture_torch:
        args.device, args.dtype = "mps", "bf16"
        capture_torch(args, records)
    else:
        if args.trace is None or args.reference is None:
            p.error("MLX runs require --trace and --reference")
        run(args, records, joints)


if __name__ == "__main__":
    main()
