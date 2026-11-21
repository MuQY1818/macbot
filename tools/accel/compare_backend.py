"""Hardware-free matched-input timing and fidelity for local LeRobot checkpoints."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from macbot.config import load
from macbot.policy.models import load_generic, load_mapped_mlx, load_policy, raw_state_shape
from macbot.policy.torch_options import TorchOptions


def package(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def machine():
    if platform.system() != "Darwin":
        return platform.machine(), platform.processor()
    return tuple(subprocess.check_output(["sysctl", "-n", key], text=True).strip()
                 for key in ("hw.model", "machdep.cpu.brand_string"))


def error(candidate, reference, gripper, joint_units, degrees_per_unit=None):
    if candidate.shape != reference.shape:
        raise ValueError("Candidate and reference actions have different shapes")
    if not 0 <= gripper < candidate.shape[-1]:
        raise ValueError("Gripper index is outside the checkpoint's action vector")
    differences = np.abs(candidate - reference)
    joints = np.delete(differences, gripper, axis=-1)
    if degrees_per_unit is not None:
        scale = np.asarray(degrees_per_unit)
        if scale.shape != (joints.shape[-1],) or not np.all(np.isfinite(scale) & (scale > 0)):
            raise ValueError("Provide one positive, finite degree scale per non-gripper joint")
        joints = joints * scale
    elif joint_units == "rad":
        joints = np.rad2deg(joints)
    return {
        name: {"mean": float(values.mean()), "p99": float(np.percentile(values, 99))}
        for name, values in (("joints_degrees", joints), ("gripper_native", differences[..., gripper]))
    }


def run(args):
    import torch

    torch.set_num_threads(args.cpu_threads)
    cfg = load(args.config)
    path = args.policy
    meta = json.loads((path / "config.json").read_text())
    kind = meta["type"]
    options = TorchOptions.from_args(args)
    options.validate(args.backend, kind, args.device)
    if args.backend == "mlx" and args.device == "cpu":
        import mlx.core as mx

        mx.set_default_device(mx.cpu)
        args.device = "mps"  # The deployment loader only exposes MLX on Apple silicon.
        use_mlx_cpu = True
    else:
        use_mlx_cpu = False
    if kind == "pi05":
        pol = load_policy(path, args.device, cfg["task"]["instruction"], backend=args.backend)
        # The deployment defaults cover the reference robot. This comparison
        # uses the camera suffixes in the public checkpoint instead.
        pol.rename = cfg.get("policy", {}).get("camera_map", {})
        if args.compile_mlx or args.compile_pi05_torch:
            pol.enable_compile()
    elif args.backend == "mlx":
        pol = load_mapped_mlx(path, cfg, args)
        if args.compile_mlx:
            pol.enable_compile()
    elif kind == "act":
        pol = load_policy(path, args.device, cfg["task"]["instruction"])
    else:
        if kind == "smolvla":
            # The saved checkpoint includes the full backbone. On the FP32
            # reference, instantiate from config and require a strict weight load.
            from lerobot.configs import PreTrainedConfig
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

            load_config = PreTrainedConfig.from_pretrained
            load_weights = SmolVLAPolicy.from_pretrained

            def fp32_config(*a, **kw):
                c = load_config(*a, **kw)
                c.load_vlm_weights = False
                return c

            with (patch.object(PreTrainedConfig, "from_pretrained", fp32_config),
                  patch.object(SmolVLAPolicy, "from_pretrained", lambda *a, **kw: load_weights(*a, strict=True, **kw))):
                pol = load_generic(path, cfg, args)
            if args.precision == "bf16-backbone":
                pol.pol.model.vlm_with_expert.vlm.to(dtype=torch.bfloat16)
        else:
            pol = load_generic(path, cfg, args)
        if args.device == "cpu":
            pol.pol.float()
    options.install(pol)
    width = (raw_state_shape(path, meta) or meta["output_features"]["action"]["shape"])[0]
    rng = np.random.default_rng(624)
    inputs = [
        ({name: rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
          for name, cam in cfg["cameras"].items() for w, h in [cam["save_size"]]},
         rng.normal(0, 1, size=width).astype(np.float32))
        for _ in range(args.calls)
    ]
    noises = np.random.default_rng(args.seed)
    device = "cpu" if use_mlx_cpu else args.device
    guided = args.mode == "rtc"
    if guided:
        if kind not in {"pi0", "pi05"}:
            raise ValueError("This harness only supports RTC for pi0 and pi05")
        if args.backend == "torch" and kind == "pi0":
            from lerobot.configs.types import RTCAttentionSchedule
            from lerobot.policies.rtc.configuration_rtc import RTCConfig

            pol.pol.config.rtc_config = RTCConfig(enabled=True, mode="guided", max_guidance_weight=10.0,
                                                  prefix_attention_schedule=RTCAttentionSchedule.LINEAR)
            pol.pol.init_rtc_processor()
            pol.pol.requires_grad_(False)
        else:
            pol.enable_rtc(10.0)
    actual_width = meta["output_features"]["action"]["shape"][0]
    left = np.random.default_rng(177).normal(0, 0.25, size=(min(35, pol.H), actual_width)).astype(np.float32)

    def call(i, *, capture_plan=False):
        imgs, q = inputs[i]
        if kind == "act":
            return pol(imgs, q)
        if kind == "diffusion":
            # reset was performed outside the timed region for a full replan on each call
            size = (1, meta["horizon"], actual_width)
        else:
            size = (1, meta["chunk_size"], meta["max_action_dim"])
        n = noises.standard_normal(size, dtype=np.float32)
        if args.backend == "mlx":
            kwargs = dict(leftover=left, delay=4, horizon=20) if guided else {}
            return pol(imgs, q, noise=n, **kwargs)
        if kind == "pi05":
            noise = torch.from_numpy(n).to(args.device)
            kwargs = dict(leftover=left, delay=4, horizon=20) if guided else {}
            with patch.object(pol.pol.model, "sample_noise", lambda shape, device: noise):
                return pol(imgs, q, **kwargs)
        if kind == "diffusion":
            # Preserve real step-mode history. The first call prefills it using the
            # current observation; capture the entire queued plan for fidelity.
            first = pol(imgs, q, noise=torch.from_numpy(n))
            rest = pol.pol._queues["action"]
            if not capture_plan or not rest:
                return first
            with torch.inference_mode():
                return np.concatenate((first, pol.post(torch.stack(tuple(rest), dim=1))[0].float().cpu().numpy()))
        if guided:
            # The generic adapter's inference_mode disables RTC gradients.
            with torch.no_grad():
                raw = pol.pol.predict_action_chunk(
                    pol.pre(pol._observation(imgs, q)), noise=torch.from_numpy(n), inference_delay=4,
                    prev_chunk_left_over=torch.from_numpy(left)[None].to(args.device), execution_horizon=20,
                )
                return pol._result(pol.post(raw))
        return pol(imgs, q, noise=torch.from_numpy(n))

    for i in range(args.warmup):
        if hasattr(pol, "reset"):
            pol.reset()
        call(i)
    noises = np.random.default_rng(args.seed)
    ts, actions = [], []
    for i in range(args.calls):
        if hasattr(pol, "reset"):
            pol.reset()
        start = perf_counter()
        actions.append(call(i))
        ts.append((perf_counter() - start) * 1000)
    if kind == "diffusion":
        # DDPM has a second source of randomness inside every scheduler step.
        # Seed one CPU generator and supply its draws on each reference device.
        scheduler = import_module(type(pol.pol.diffusion.noise_scheduler).__module__)
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 174)

        def noise_tensor(shape, *, device=None, dtype=None, layout=None, **_):
            return torch.randn(shape, generator=generator, dtype=dtype, layout=layout or torch.strided).to(device)

        noises = np.random.default_rng(args.seed)
        actions = []
        with patch.object(scheduler, "randn_tensor", noise_tensor):
            for i in range(args.calls):
                pol.reset()
                actions.append(call(i, capture_plan=True))
    actions = np.stack(actions)
    if not np.isfinite(actions).all():
        raise ValueError("Nonfinite actions")
    model, chip = machine()
    result = {
        "machine": model, "chip": chip, "torch": package("torch"), "mlx": package("mlx"), "lerobot": package("lerobot"),
        "checkpoint": args.checkpoint_name, "kind": kind, "device": device, "backend": args.backend,
        "precision": args.precision, "chunk_length": actions.shape[1], "execution": args.mode,
        "denoising_steps": getattr(pol, "steps", None) or meta.get("num_inference_steps") or meta.get("num_steps")
        or (meta.get("num_train_timesteps") if kind == "diffusion" else None),
        "warmup": args.warmup, "calls": args.calls, "seed": args.seed,
        "median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)),
        "torch_options": vars(options), "compile_mlx": args.compile_mlx, "compile_pi05_torch": args.compile_pi05_torch,
    }
    if args.reference:
        result["fidelity"] = error(actions, np.load(args.reference), args.gripper_index,
                                   args.joint_units, args.joint_degrees_per_unit)
    if args.joint_degrees_per_unit is not None:
        result["joint_degrees_per_unit"] = args.joint_degrees_per_unit
    np.save(args.out.with_suffix(".npy"), actions)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--precision", required=True, help="precision label for the tested configuration")
    parser.add_argument("--backend", choices=("torch", "mlx"), default="torch")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--mode", choices=("chunk", "rtc", "step-replan"), default="chunk")
    parser.add_argument("--compile-mlx", action="store_true")
    parser.add_argument("--compile-pi05-torch", action="store_true")
    parser.add_argument("--torch-autocast", choices=("fp16", "bf16"))
    parser.add_argument("--torch-channels-last", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--torch-single-readback", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    units = parser.add_mutually_exclusive_group(required=True)
    units.add_argument("--joint-units", choices=("deg", "rad"), help="units of the checkpoint's joint outputs")
    units.add_argument("--joint-degrees-per-unit", type=float, nargs="+", metavar="SCALE",
                       help="degrees per normalized unit, in non-gripper joint order")
    parser.add_argument("--gripper-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=941)
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.set_defaults(mlx_bits=None, mlx_quantize="prefix", mlx_time_grid="uniform")
    args = parser.parse_args()
    if args.calls < 50 or not 1 <= args.warmup <= args.calls:
        parser.error("Need at least 50 timed calls and between 1 and --calls warm-up calls")
    run(args)


if __name__ == "__main__":
    main()
