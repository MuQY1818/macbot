"""Offline PI05 latency and action-fidelity experiments on recorded macbot observations."""

from __future__ import annotations

import argparse
import json
import resource
import sys
from collections import defaultdict
from functools import wraps
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np

from macbot.policy.models import PI05, _to_obs


def prepare(args):
    if args.episodes < 2 or args.samples < max(50, args.episodes):
        raise ValueError("Need at least 50 observations across at least two episodes")
    episodes = []
    for path in sorted(args.data.glob("episode_*.npz")):
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
        if not meta.get("fake") and (meta.get("check") or {}).get("ok", True):
            episodes.append(path)
    if len(episodes) < args.episodes:
        raise ValueError(f"Need {args.episodes} usable episodes; found {len(episodes)}")
    fields = defaultdict(list)
    for j, index in enumerate(np.linspace(0, len(episodes) - 1, args.episodes, dtype=int)):
        path = episodes[index]
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            q = data["qpos"].astype(np.float32)
            count = args.samples // args.episodes + (j < args.samples % args.episodes)
            if len(q) < count:
                raise ValueError(f"{path.name} has fewer than {count} frames")
            frames = np.linspace(0, len(q) - 1, count, dtype=int)
            fields["state"].extend(q[frames])
            fields["episode"].extend([path.name] * count)
            fields["frame"].extend(frames)
            fields["task"].extend([meta["instruction"]] * count)
            fields["joints"].append(meta["joint_names"])
            for camera in args.cameras:
                offsets, blob = data[f"{camera}_offsets"], data[f"{camera}_jpeg"]
                for frame in frames:
                    image = cv2.imdecode(blob[offsets[frame]:offsets[frame + 1]], cv2.IMREAD_COLOR)
                    fields[f"image_{camera}"].append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if any(names != fields["joints"][0] for names in fields["joints"]):
        raise ValueError("Inconsistent joint ordering in evaluation episodes")
    fields["joints"] = fields["joints"][0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **fields)
    print(f"Prepared {args.samples} observations across {args.episodes} episodes")


def observations(path):
    with np.load(path) as data:
        cameras = {k.removeprefix("image_"): data[k] for k in data.files if k.startswith("image_")}
        return [
            ({k: stream[i] for k, stream in cameras.items()}, q, str(task))
            for i, (q, task) in enumerate(zip(data["state"], data["task"], strict=True))
        ], list(data["joints"])


def stats(values):
    return {"median": float(np.median(values)), "p90": float(np.percentile(values, 90))}


def metrics(a, b, joints, binary_gripper):
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != len(joints):
        raise ValueError("Action shapes or joint labels differ")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite actions")
    arm = [i for i, name in enumerate(joints) if name != "gripper"]
    gripper = [i for i, name in enumerate(joints) if name == "gripper"]
    delta = np.abs(a - b)
    result = {}
    for label, stop in (("all", a.shape[1]), ("first10", 10)):
        for part, indices, units in (
            ("arm", arm, "degrees"), ("gripper", gripper, "score" if binary_gripper else "degrees")
        ):
            if indices:
                d = delta[:, :stop, indices]
                result[f"{part}_{label}"] = {
                    "mean": float(d.mean()), "p99": float(np.percentile(d, 99)), "units": units,
                }
    if gripper and binary_gripper:
        result["binary_threshold_disagreement"] = {
            label: float(np.mean((a[:, :stop, gripper] >= 0.5) != (b[:, :stop, gripper] >= 0.5)))
            for label, stop in (("all", a.shape[1]), ("first10", 10))
        }
    return result


class Stages:
    """Measure separately from uninstrumented throughput; barriers can inflate timings."""

    def __init__(self, sync):
        self.sync = sync
        self.samples = defaultdict(list)
        self.counts = defaultdict(int)

    def wrap(self, fn, name, indexed=False):
        @wraps(fn)
        def timed(*args, **kwargs):
            key = name
            if indexed:
                key = f"{name}_{self.counts[name]:02d}"
                self.counts[name] += 1
            self.sync()
            start = perf_counter()
            value = fn(*args, **kwargs)
            self.sync()
            self.samples[key].append((perf_counter() - start) * 1e3)
            return value

        return timed

    def instrument(self, policy, stack):
        from macbot.policy import models

        m = policy.pol.model
        v = m.paligemma_with_expert
        prefix = v.paligemma.model.language_model
        for obj, attr, name, indexed in (
            (models, "_to_obs", "rgb_to_host_tensors", False),
            (policy.pol, "_preprocess_images", "image_resize_normalize", False),
            (v, "embed_image", "vision_and_projection", True),
            (prefix, "forward", "prefix_and_kv", False),
            (m, "denoise_step", "denoise", True),
            (m, "sample_noise", "noise", False),
            (policy.torch.Tensor, "cpu", "explicit_cpu_copy", True),
            (policy.torch.mps, "synchronize", "explicit_mps_sync", True),
        ):
            stack.enter_context(patch.object(obj, attr, self.wrap(getattr(obj, attr), name, indexed)))
        for stem, pipeline in (("pre", policy.pre), ("post", policy.post)):
            steps = [self.wrap(step, f"{stem}_{type(step).__name__}") for step in pipeline.steps]
            stack.enter_context(patch.object(pipeline, "steps", steps))


def load_torch(args, task):
    import torch
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    loaded = False
    original = PI05Policy.load_state_dict

    def strict_load(self, state, *a, **kw):
        nonlocal loaded
        result = original(self, state, *a, **kw)
        loaded = not (result.missing_keys or result.unexpected_keys)
        return result

    with patch.object(PI05Policy, "load_state_dict", strict_load):
        policy = PI05(args.checkpoint, args.device, task, steps=args.steps)
    if not loaded:
        raise RuntimeError("LeRobot did not strictly load the requested checkpoint")

    # Reload from disk to avoid quantizing originally FP32 tensors through BF16.
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if dtype != torch.bfloat16:
        from safetensors.torch import load_file

        policy.pol.to(dtype=dtype)
        weights = load_file(args.checkpoint / "model.safetensors")
        policy.pol.load_state_dict(policy.pol._convert_openpi_state_dict(weights), strict=True)

    return policy, dtype


def torch_variants(args, policy, stack):
    from lerobot.policies.pi05 import modeling_pi05
    from transformers.models.gemma import modeling_gemma

    torch = policy.torch
    m = policy.pol.model
    if args.sdpa:
        def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            key = modeling_gemma.repeat_kv(key, module.num_key_value_groups)
            value = modeling_gemma.repeat_kv(value, module.num_key_value_groups)
            mask = attention_mask.to(query.dtype) if attention_mask is not None else None
            x = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, scale=scaling,
            )
            return x.transpose(1, 2).contiguous(), None

        stack.enter_context(patch.object(modeling_gemma, "eager_attention_forward", attention))
    if args.share_cache:
        def share_cache(cache):
            # Fresh cache metadata. Gemma's dynamic update concatenates, rather than mutating the original tensors.
            return type(cache)(tuple(cache))

        stack.enter_context(patch.object(modeling_pi05, "clone_past_key_values", share_cache))
    if args.pool_images:
        embed = m._embed_images

        def embed_pooled(images, masks):
            def pool(x):
                if not x.shape[1]:
                    return x
                side = int(x.shape[1] ** 0.5)
                return x.reshape(x.shape[0], side // 2, 2, side // 2, 2, x.shape[-1]).mean((2, 4)).flatten(1, 2)

            return [pool(x) for x in embed(images, masks)]

        stack.enter_context(patch.object(m, "_embed_images", embed_pooled))


def run(args):
    from contextlib import ExitStack

    import torch

    records, joints = observations(args.observations)
    if len(records) < 50 or args.calls < 30 or args.warmup < 1:
        raise ValueError("Need at least 50 observations, 30 timed calls and one warm-up call")
    policy, dtype = load_torch(args, records[0][2])
    m = policy.pol.model
    sync = torch.mps.synchronize if args.device == "mps" else lambda: None
    if args.separate_cameras:
        m._embed_images = lambda images, masks: [
            m._embed_one_image(image, mask) if bool(mask.any())
            else images[0].new_empty((len(image), 0, m.paligemma_with_expert.paligemma.config.text_config.hidden_size))
            for image, mask in zip(images, masks, strict=True)
        ]
    raw_noise = m.sample_noise
    raw_autocast = torch.autocast
    shape = (1, m.config.chunk_size, m.config.max_action_dim)
    noises = np.random.default_rng(args.seed).standard_normal((len(records), *shape), dtype=np.float32)
    gripper_path = args.checkpoint / "gripper.json"
    binary = gripper_path.is_file() and json.loads(gripper_path.read_text()).get("mode") == "binary"

    def call(i, fixed=False):
        imgs, q, policy.task = records[i % len(records)]
        if fixed:
            noise = torch.from_numpy(noises[i]).to(args.device)
            m.sample_noise = lambda shape, device: noise
        try:
            if args.single_readback:
                with torch.inference_mode(), raw_autocast(args.device, dtype=dtype):
                    policy.pol.reset()
                    raw = policy.pol.predict_action_chunk(policy.pre(_to_obs(torch, imgs, q, policy.task, policy.rename)))
                    return policy.post(raw).float().cpu().numpy()[0]
            return policy(imgs, q)
        finally:
            if fixed:
                m.sample_noise = raw_noise

    def autocast(device, *a, **kw):
        if device == args.device:
            if dtype == torch.float32:
                kw["enabled"] = False
            else:
                kw["dtype"] = dtype
        return raw_autocast(device, *a, **kw)

    result = {
        "device": args.device, "torch": torch.__version__, "dtype": args.dtype, "seed": args.seed,
        "steps": m.config.num_inference_steps, "compile": not args.eager and args.dtype != "fp32",
        "separate_cameras": args.separate_cameras, "count": len(records), "warmup": args.warmup,
        "sdpa": args.sdpa, "share_cache": args.share_cache, "pool_images": args.pool_images,
        "single_readback": args.single_readback, "dynamic_compile": args.dynamic_compile,
    }
    with ExitStack() as stack:
        torch_variants(args, policy, stack)
        if dtype != torch.bfloat16:
            stack.enter_context(patch.object(torch, "autocast", autocast))
        if result["compile"]:
            if args.dynamic_compile:
                m.denoise_step = torch.compile(m.denoise_step, dynamic=True)
                lm = m.paligemma_with_expert.paligemma.model.language_model
                lm.forward = torch.compile(lm.forward, dynamic=True)
            else:
                policy.enable_compile()
        for i in range(args.warmup):
            call(i)
        times, memory = [], []
        for i in range(args.calls):
            sync()
            t = perf_counter()
            call(i)
            sync()
            times.append((perf_counter() - t) * 1e3)
            if args.device == "mps":
                memory.append((torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()))
        result["latency_ms"] = {**stats(times), "calls": len(times), "samples": times}
        result["memory_bytes"] = {"peak_process_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        if sys.platform != "darwin":
            result["memory_bytes"]["peak_process_rss"] *= 1024
        if memory:
            result["memory_bytes"].update({
                "max_sampled_mps_live": max(x[0] for x in memory),
                "max_sampled_mps_driver": max(x[1] for x in memory),
            })
        print(f"Timing complete: {stats(times)}", flush=True)
        actions = np.stack([call(i, fixed=True) for i in range(len(records))])
        if not np.isfinite(actions).all():
            raise ValueError("Nonfinite actions")
        if args.reference:
            result["fidelity"] = metrics(actions, np.load(args.reference), joints, binary)
        if args.profile:
            stages = Stages(sync)
            with ExitStack() as profile_stack:
                stages.instrument(policy, profile_stack)
                for i in range(args.calls):
                    stages.counts.clear()
                    stages.wrap(call, "profile_total")(i)
            result["profile_ms"] = {k: {**stats(v), "calls": len(v)} for k, v in stages.samples.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out.with_suffix(".npy"), actions)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "latency_ms"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cameras", nargs=2, default=["top", "wrist"])
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--episodes", type=int, default=10)
    p = sub.add_parser("run")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--device", choices=["mps", "cpu"], default="mps")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=941)
    p.add_argument("--steps", type=int)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--calls", type=int, default=30)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--separate-cameras", action="store_true")
    p.add_argument("--sdpa", action="store_true")
    p.add_argument("--share-cache", action="store_true")
    p.add_argument("--pool-images", action="store_true")
    p.add_argument("--single-readback", action="store_true")
    p.add_argument("--dynamic-compile", action="store_true")
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()
