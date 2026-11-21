"""Policy wrappers: observation (camera images, measured joints) -> action chunk (H, joints).

Both wrappers run on Apple silicon through PyTorch MPS.

pi0.5 notes (all measured on an M-series MacBook):
* Loading: the model constructor randomly initializes about 4B parameters (~80 s) that the checkpoint
  then overwrites. We turn ``normal_`` and ``uniform_`` into no-ops while constructing.
* Precision: weights in bfloat16 under MPS autocast. The noise is created as float32 and MPS refuses
  matmuls with mixed dtypes, so autocast is required.
* Empty camera slots: the base checkpoint has three image slots. With two cameras the third is a masked
  blank image, which still costs a vision pass and 256 prefix tokens. Masked tokens take no part in
  attention, so we drop them; outputs are identical and 10 denoising steps go from ~523 to ~453 ms.
* ``torch.compile``: each denoising step dispatches ~4300 small kernels and the CPU-side dispatch
  dominates. Compiling ``denoise_step`` and the Gemma prefix forward brings 10 steps from ~412 to
  ~282 ms. Compiling the SigLIP vision tower makes it slower, so it is left alone. First call compiles (~25 s).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def prep_image(img_bgr, size, quality: int):
    """Same path as recording: resize (INTER_AREA) -> JPEG -> decode -> RGB. Returns (RGB uint8, JPEG bytes)."""
    small = cv2.resize(img_bgr, tuple(size), interpolation=cv2.INTER_AREA)
    _, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    rgb = cv2.cvtColor(cv2.imdecode(jpg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return rgb, jpg.reshape(-1)


def _to_obs(torch, imgs: dict, q, task: str, rename: dict | None = None):
    obs = {"observation.state": torch.tensor(np.asarray(q, np.float32)).unsqueeze(0)}
    for cam, img in imgs.items():
        key = (rename or {}).get(cam, cam)
        obs[f"observation.images.{key}"] = torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0)
    obs["task"] = [task]
    return obs


class ACT:
    kind = "act"
    supports_rtc = False

    def __init__(self, path: Path, device: str, task: str):
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        self.torch, self.device, self.task = torch, device, task
        self.pol = ACTPolicy.from_pretrained(path)
        self.pol.config.device = device
        self.pol.to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.pol.config, pretrained_path=str(path), preprocessor_overrides={"device_processor": {"device": device}})
        feats = self.pol.config.input_features
        self.shapes = {k.removeprefix("observation.images."): tuple(feats[k].shape)
                       for k in feats if k.startswith("observation.images.")}    # (3, H, W)
        self.H = self.pol.config.chunk_size

    def __call__(self, imgs: dict, q, **_) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            a = self.post(self.pol.predict_action_chunk(self.pre(_to_obs(torch, imgs, q, self.task))))
        if self.device == "mps":
            torch.mps.synchronize()
        return a.squeeze(0).float().cpu().numpy()


class PI05:
    kind = "pi05"
    supports_rtc = True
    RENAME = {"top": "base_0_rgb", "wrist": "left_wrist_0_rgb"}

    def __init__(self, path: Path, device: str, task: str, steps: int | None = None, rename: dict | None = None):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        self.torch, self.device, self.task = torch, device, task
        self.rename = rename or self.RENAME
        saved = {k: getattr(torch.Tensor, k) for k in ("normal_", "uniform_")}
        try:                                          # skip the random init that the checkpoint overwrites
            for k in saved:
                setattr(torch.Tensor, k, lambda self, *a, **kw: self)
            self.pol = PI05Policy.from_pretrained(path)
        finally:
            for k, v in saved.items():
                setattr(torch.Tensor, k, v)
        self.pol.config.device = device
        if steps:
            self.pol.config.num_inference_steps = steps
        self.pol.to(device=device, dtype=torch.bfloat16).eval()
        self.pre, self.post = make_pre_post_processors(
            self.pol.config, pretrained_path=str(path), preprocessor_overrides={"device_processor": {"device": device}})
        self.shapes = None                            # pi0.5 resizes to 224 itself
        self.H = self.pol.config.n_action_steps
        self.rtc = False
        self._skip_empty_cams()

    def _skip_empty_cams(self):
        model = self.pol.model
        orig = model._embed_images

        def embed(images, img_masks):
            live = [j for j, m in enumerate(img_masks) if bool(m.any())]
            if not live or len(live) == len(img_masks):
                return orig(images, img_masks)
            es = list(orig([images[j] for j in live], [img_masks[j] for j in live]))
            it = iter(es)
            return [next(it) if j in live else es[0][:, :0] for j in range(len(img_masks))]
        model._embed_images = embed

    def enable_compile(self):
        torch, m = self.torch, self.pol.model
        m.denoise_step = torch.compile(m.denoise_step, dynamic=False)
        lm = m.paligemma_with_expert.paligemma.model.language_model
        lm.forward = torch.compile(lm.forward, dynamic=False)   # pi0.5 calls .forward directly

    def enable_rtc(self, beta: float, schedule: str = "linear"):
        """Real-Time Chunking (Black et al., 2025) in LeRobot's guided mode."""
        from lerobot.configs.types import RTCAttentionSchedule
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        self.pol.config.rtc_config = RTCConfig(enabled=True, mode="guided", max_guidance_weight=beta,
                                               execution_horizon=self.H,
                                               prefix_attention_schedule=RTCAttentionSchedule[schedule.upper()])
        self.pol.init_rtc_processor()
        self.pol.requires_grad_(False)                # guidance differentiates w.r.t. x_t only
        self.rtc = True

    def __call__(self, imgs: dict, q, leftover=None, delay: int = 0, horizon: int | None = None, return_raw=False):
        """``leftover``: unexecuted rest of the current chunk in model space (for RTC guidance)."""
        torch = self.torch
        obs = _to_obs(torch, imgs, q, self.task, self.rename)
        ac = (torch.autocast(self.device, dtype=torch.bfloat16) if self.device in ("mps", "cuda")
              else torch.autocast("cpu", enabled=False))
        kw = {}
        if leftover is not None and len(leftover) > delay > 0:
            kw = dict(prev_chunk_left_over=torch.from_numpy(np.ascontiguousarray(leftover, np.float32)).unsqueeze(0).to(self.device),
                      inference_delay=int(delay), execution_horizon=int(min(len(leftover), horizon or len(leftover))))
        with (torch.no_grad() if self.rtc else torch.inference_mode()), ac:   # guidance needs autograd
            self.pol.reset()
            raw = self.pol.predict_action_chunk(self.pre(obs), **kw)
            raw_np = raw.float().cpu().numpy().copy()
            a = self.post(raw)
        if self.device == "mps":
            torch.mps.synchronize()
        a = a.float().cpu().numpy()
        a, raw_np = (a[0] if a.ndim == 3 else a), (raw_np[0] if raw_np.ndim == 3 else raw_np)
        return (a, raw_np) if return_raw else a


def policy_backend(cfg, args):
    backend = getattr(args, "backend", None) or cfg.get("policy", {}).get("backend", "torch")
    if backend not in ("torch", "mlx", "auto"):
        sys.exit("policy.backend must be torch, mlx or auto")
    return backend


def use_generic(policy_type, implementation, backend):
    if implementation == "generic" and backend != "torch":
        sys.exit("--policy-impl generic requires the torch backend")
    return backend == "torch" and (implementation == "generic" or policy_type not in {"act", "pi05"})


def load_generic(path, cfg, args, joint_count=None):
    from .generic import GenericPolicy

    if getattr(args, "steps", None) is not None:
        sys.exit("--steps is only supported by the built-in pi0.5 wrappers")
    try:
        return GenericPolicy(
            path, args.device, cfg["task"]["instruction"],
            cameras={k: v["save_size"] for k, v in cfg["cameras"].items()},
            camera_map=cfg.get("policy", {}).get("camera_map", {}), joint_count=joint_count,
        )
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        sys.exit(f"Cannot load generic policy: {exc}")


def load_mapped_mlx(path, cfg, args, joint_count=None):
    try:
        return load_policy(
            path, args.device, cfg["task"]["instruction"], getattr(args, "steps", None), backend="mlx",
            mlx_bits=args.mlx_bits, mlx_quantize=args.mlx_quantize, mlx_time_grid=args.mlx_time_grid,
            cameras={k: v["save_size"] for k, v in cfg["cameras"].items()},
            camera_map=cfg.get("policy", {}).get("camera_map", {}), joint_count=joint_count,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(f"Cannot load MLX policy: {exc}")


def load_policy(path: Path, device: str, task: str, steps: int | None = None, *,
                backend: str = "torch", mlx_bits: int | None = None, mlx_quantize: str = "prefix",
                mlx_time_grid: str = "uniform", cameras=None, camera_map=None, joint_count=None):
    typ = json.loads((Path(path) / "config.json").read_text()).get("type", "act")
    if backend == "mlx":
        from .backends import MLX_BACKENDS

        if typ not in MLX_BACKENDS:
            sys.exit(f"No MLX backend is implemented for {typ}")
        if device != "mps":
            sys.exit("The MLX backend runs on the Apple GPU; use --device mps")
        try:
            import mlx.core as mx

            from .mlx_pi05 import PI05MLX
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("mlx"):
                sys.exit("Install the optional MLX backend with: pip install 'macbot[mlx]'")
            raise
        if typ == "pi0":
            from .mlx_pi0 import PI0MLX

            if mlx_bits is not None or mlx_time_grid != "uniform":
                sys.exit("MLX quantization and alternative time grids are only supported for pi0.5")
            return PI0MLX(Path(path), task, steps, cameras=cameras or {}, camera_map=camera_map,
                          joint_count=joint_count)
        return PI05MLX(
            Path(path), task, steps, bits=mlx_bits, quantize=mlx_quantize, trim=True, precompute_adarms=True,
            dtype=mx.float16 if mlx_bits is None else mx.bfloat16, pack_linears=mlx_bits is None,
            fused_rope=mlx_bits is None, time_grid=mlx_time_grid,
        )
    if backend != "torch":
        raise ValueError(f"Unsupported policy backend: {backend}")
    if typ == "pi05":
        return PI05(path, device, task, steps)
    if typ == "act":
        return ACT(path, device, task)
    sys.exit(f"unsupported policy type {typ} (ACT and pi0.5 are supported)")


def raw_state_shape(path: Path, cfg: dict) -> list[int] | None:
    """Resolve the raw input width when PI05 metadata describes the padded model state."""
    key = "observation.state"
    shape = ((cfg.get("input_features") or {}).get(key) or {}).get("shape")
    if cfg.get("type") != "pi05" or not shape or shape != [cfg.get("max_state_dim")]:
        return shape
    processor = path / "policy_preprocessor.json"
    if not processor.is_file():
        return shape
    for step in json.loads(processor.read_text()).get("steps", []):
        filename = step.get("state_file")
        if step.get("registry_name") != "normalizer_processor" or not filename:
            continue
        stats = path / Path(filename).name
        if not stats.is_file():
            continue
        from safetensors import safe_open

        with safe_open(str(stats), framework="np") as weights:
            widths = {tuple(weights.get_slice(name).get_shape()) for name in weights.keys()
                      if name.startswith(key + ".") and name != key + ".count"}
        if len(widths) == 1:
            dims = next(iter(widths))
            if len(dims) == 1 and 0 < dims[0] <= shape[0]:
                return list(dims)
    return shape


def check_joint_dimensions(path: Path, joints):
    """Check checkpoint metadata before loading weights or normalizing an observation."""
    cfg = json.loads((path / "config.json").read_text())
    for section, key, label in (("input_features", "observation.state", "state"),
                                ("output_features", "action", "action")):
        shape = (raw_state_shape(path, cfg) if label == "state"
                 else ((cfg.get(section) or {}).get(key) or {}).get("shape"))
        if shape is not None and shape != [len(joints)]:
            sys.exit(f"policy {label} dimension mismatch: checkpoint expects shape {shape}, "
                     f"but the robot has {len(joints)} joints ({', '.join(joints)})")
