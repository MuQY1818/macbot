"""Hardware-free matched-input timing and fidelity for local LeRobot checkpoints."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from contextlib import ExitStack
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


def recorded_inputs(path, cameras, prefix=""):
    with np.load(path, allow_pickle=False) as archive:
        state, task = archive[prefix + "q"], archive[prefix + "task"]
        frames = {name: archive[f"{prefix}rgb_{name}"] for name in cameras}
        if len(state) < 20 or len(set(archive["episode_index"])) < 2:
            raise ValueError("Need at least 20 recorded observations spanning multiple episodes")
        for name, (w, h) in cameras.items():
            if frames[name].shape != (len(state), h, w, 3) or frames[name].dtype != np.uint8:
                raise ValueError(f"Invalid recorded RGB shape for {name}")
        return [({name: image[i] for name, image in frames.items()}, q, text)
                for i, (q, text) in enumerate(zip(state, task, strict=True))]


def load_pi_torch(path, cfg, args, kind):
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    cls = get_policy_class(kind)
    from_pretrained = PreTrainedConfig.from_pretrained
    to_device = cls.to
    load_weights = cls.load_state_dict
    loaded = False

    def config(*a, **kw):
        result = from_pretrained(*a, **kw)
        if args.pi_fp32:
            result.dtype = torch.float32
        if args.pi_fp32 or args.skip_pi_checkpoint_compile:
            result.compile_model = False
        return result

    def fp32_to(model, *a, **kw):
        if "dtype" in kw:
            kw["dtype"] = torch.float32
        return to_device(model, *a, **kw)

    def verified_load(model, *a, **kw):
        nonlocal loaded
        result = load_weights(model, *a, **kw)
        loaded = not (result.missing_keys or result.unexpected_keys)
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(PreTrainedConfig, "from_pretrained", config))
        stack.enter_context(patch.object(cls, "load_state_dict", verified_load))
        if args.pi_fp32:
            stack.enter_context(patch.object(cls, "to", fp32_to))
        pol = (load_policy(path, args.device, cfg["task"]["instruction"]) if kind == "pi05"
               else load_generic(path, cfg, args))
    if not loaded:
        raise RuntimeError("LeRobot did not fully load the requested checkpoint")
    if args.pi_fp32 and any(p.dtype != torch.float32 for p in pol.pol.parameters() if p.is_floating_point()):
        raise RuntimeError("The FP32 reference contains reduced-precision model parameters")
    return pol


def run(args):
    import torch

    torch.set_num_threads(args.cpu_threads)
    cfg = load(args.config)
    path = args.policy
    meta = json.loads((path / "config.json").read_text())
    kind = meta["type"]
    if args.observations and kind not in {"pi0", "pi05"}:
        raise ValueError("Recorded comparisons are supported for pi0 and pi05")
    if args.pi_fp32 and (kind not in {"pi0", "pi05"} or args.backend != "torch" or args.compile_pi05_torch):
        raise ValueError("--pi-fp32 requires eager Torch pi0 or pi05")
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
        pol = (load_pi_torch(path, cfg, args, kind) if args.backend == "torch" else
               load_policy(path, args.device, cfg["task"]["instruction"], backend=args.backend))
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
    elif kind == "pi0":
        pol = load_pi_torch(path, cfg, args, kind)
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
    cameras = {name: cam["save_size"] for name, cam in cfg["cameras"].items()}
    if args.observations:
        inputs = recorded_inputs(args.observations, cameras)
        if args.calls % len(inputs):
            raise ValueError("Use a whole number of passes over the recorded frame set")
    else:
        rng = np.random.default_rng(624)
        inputs = [
            ({name: rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for name, (w, h) in cameras.items()},
             rng.normal(0, 1, size=width).astype(np.float32), cfg["task"]["instruction"])
            for _ in range(args.calls)
        ]
    noises = np.random.default_rng(args.seed)
    device = "cpu" if use_mlx_cpu else args.device
    guided = args.mode == "rtc"
    actual_width = meta["output_features"]["action"]["shape"][0]
    synthetic_left = np.random.default_rng(177).normal(0, 0.25, size=(min(35, pol.H), actual_width)).astype(np.float32)
    leftovers = np.load(args.rtc_leftovers, allow_pickle=False) if args.rtc_leftovers else None

    def call(i, *, capture_plan=False, previous=None):
        imgs, q, task = inputs[i % len(inputs)] if previous is None else previous
        if args.observations:
            pol.task = str(task)
        use_rtc = guided and previous is None
        left = (leftovers[i % len(inputs)] if use_rtc and leftovers is not None else synthetic_left)
        kwargs = dict(leftover=left, delay=4, horizon=20) if use_rtc else {}
        if kind == "act":
            return pol(imgs, q)
        if kind == "diffusion":
            # reset was performed outside the timed region for a full replan on each call
            size = (1, meta["horizon"], actual_width)
        else:
            size = (1, meta["chunk_size"], meta["max_action_dim"])
        n = noises.standard_normal(size, dtype=np.float32)
        if args.backend == "mlx":
            return pol(imgs, q, noise=n, **kwargs)
        if kind == "pi05":
            noise = torch.from_numpy(n).to(args.device)
            with patch.object(pol.pol.model, "sample_noise", lambda shape, device: noise):
                if args.pi_fp32:
                    autocast = torch.autocast
                    with patch.object(torch, "autocast", lambda *a, **kw: autocast(*a, **{**kw, "enabled": False})):
                        result = pol(imgs, q, return_raw=previous is not None, **kwargs)
                else:
                    result = pol(imgs, q, **kwargs)
            return result[1] if previous is not None else result
        if kind == "diffusion":
            # Preserve real step-mode history. The first call prefills it using the
            # current observation; capture the entire queued plan for fidelity.
            first = pol(imgs, q, noise=torch.from_numpy(n))
            rest = pol.pol._queues["action"]
            if not capture_plan or not rest:
                return first
            with torch.inference_mode():
                return np.concatenate((first, pol.post(torch.stack(tuple(rest), dim=1))[0].float().cpu().numpy()))
        if previous is not None:
            with torch.inference_mode():
                batch = pol.pre(pol._observation(imgs, q))
                return pol.pol.predict_action_chunk(batch, noise=torch.from_numpy(n))[0].float().cpu().numpy()
        if use_rtc:
            # The generic adapter's inference_mode disables RTC gradients.
            with torch.no_grad():
                raw = pol.pol.predict_action_chunk(
                    pol.pre(pol._observation(imgs, q)), noise=torch.from_numpy(n), inference_delay=4,
                    prev_chunk_left_over=torch.from_numpy(left)[None].to(args.device), execution_horizon=20,
                )
                return pol._result(pol.post(raw))
        return pol(imgs, q, noise=torch.from_numpy(n))

    if args.save_rtc_leftovers:
        if not (guided and args.observations and args.pi_fp32) or leftovers is not None:
            raise ValueError("Generating RTC leftovers requires recorded FP32 RTC without --rtc-leftovers")
        previous = recorded_inputs(args.observations, cameras, "prior_")
        noises = np.random.default_rng(177)
        plans = []
        for i, frame in enumerate(previous):
            if hasattr(pol, "reset"):
                pol.reset()
            plans.append(call(i, previous=frame)[4:, :actual_width])
        leftovers = np.stack(plans)
        np.save(args.save_rtc_leftovers, leftovers)
        noises = np.random.default_rng(args.seed)
    if guided:
        if kind not in {"pi0", "pi05"}:
            raise ValueError("This harness only supports RTC for pi0 and pi05")
        if args.observations and (leftovers is None or leftovers.shape != (len(inputs), pol.H - 4, actual_width)
                                  or not np.isfinite(leftovers).all()):
            raise ValueError("Recorded RTC needs one finite model-space leftover per observation")
        if args.backend == "torch" and kind == "pi0":
            from lerobot.configs.types import RTCAttentionSchedule
            from lerobot.policies.rtc.configuration_rtc import RTCConfig

            pol.pol.config.rtc_config = RTCConfig(enabled=True, mode="guided", max_guidance_weight=10.0,
                                                  prefix_attention_schedule=RTCAttentionSchedule.LINEAR)
            pol.pol.init_rtc_processor()
            pol.pol.requires_grad_(False)
        else:
            pol.enable_rtc(10.0)
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
    if args.observations:
        result.update(recorded_frames=len(inputs), pi_fp32=args.pi_fp32,
                      skip_pi_checkpoint_compile=args.skip_pi_checkpoint_compile)
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
    parser.add_argument("--pi-fp32", action="store_true", help="load the full-precision pi0/pi05 Torch reference")
    parser.add_argument("--skip-pi-checkpoint-compile", action="store_true", help="disable the saved Torch compiler")
    parser.add_argument("--observations", type=Path, help="NPZ from tools.accel.recorded_observations")
    parser.add_argument("--rtc-leftovers", type=Path, help="use shared model-space history for recorded RTC")
    parser.add_argument("--save-rtc-leftovers", type=Path, help="create shared history with the FP32 reference")
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
