"""Select a policy backend from published evidence, without timing inference."""
from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass
from importlib.util import find_spec
from typing import NamedTuple


class Backend(NamedTuple):
    name: str
    modes: frozenset[str]


CHUNK = frozenset({"chunk", "ensemble", "async"})
MLX_BACKENDS = {"pi05": Backend("mlx", CHUNK | {"rtc"})}


@dataclass(frozen=True)
class Qualification:
    """A measured winner for this checkpoint, hardware, runtime and mode."""

    backend: str
    checkpoint: str
    machine: str
    modes: frozenset[str]
    compiled: bool
    packages: tuple[tuple[str, str], ...]


# Entries must link to a trained public checkpoint and matched fidelity and latency
# results in docs/acceleration.md. Results from private checkpoints do not transfer.
QUALIFIED: dict[str, tuple[Qualification, ...]] = {}


@dataclass(frozen=True)
class Selection:
    requested: str
    backend: str
    reason: str

    def announce(self):
        prefix = "Backend auto selected" if self.requested == "auto" else "Backend"
        print(f"{prefix}: {self.backend} ({self.reason})", flush=True)


def available(kind):
    return ("torch", MLX_BACKENDS[kind].name) if kind in MLX_BACKENDS else ("torch",)


def checkpoint_digest(path):
    """Hash the names and content of a flat, local LeRobot checkpoint."""
    identity = hashlib.sha256()
    for entry in sorted(path.iterdir()):
        if not entry.is_file() or entry.suffix not in {".json", ".safetensors"}:
            continue
        with entry.open("rb") as data:
            digest = hashlib.file_digest(data, "sha256").digest()
        identity.update(entry.name.encode("utf-8") + b"\0" + digest)
    return identity.hexdigest()


def machine_model():
    import subprocess

    return subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()


def select_backend(requested, path, metadata, args, *, bench=False):
    if requested != "auto":
        return Selection(requested, requested, "configured")
    if (getattr(args, "mlx_bits", None) is not None or getattr(args, "mlx_time_grid", "uniform") != "uniform"):
        raise SystemExit("Lossy MLX options require explicit --backend mlx")
    steps = getattr(args, "steps", None)
    if steps is not None and steps != metadata.get("num_inference_steps", 10):
        raise SystemExit("Changing denoising steps requires an explicit backend")

    def fallback(reason):
        return Selection("auto", "torch", reason)

    if getattr(args, "policy_impl", "auto") == "generic":
        return fallback("generic implementation requires torch")
    if args.device != "mps":
        return fallback(f"no accelerated backend qualified on {args.device}")
    kind = metadata.get("type", "act")
    capability = MLX_BACKENDS.get(kind)
    if capability is None:
        return fallback(f"no alternative backend for {kind}")
    mode = getattr(args, "mode", "auto")
    if mode == "auto":
        mode = "rtc" if kind == "pi05" and not bench else "chunk"
    if mode not in capability.modes:
        return fallback(f"{capability.name} does not support {mode} for {kind}")
    if any(metadata.get(k) for k in ("use_visual_memory", "use_proprioceptive_memory")) or (
        metadata.get("rtc_config") or {}
    ).get("enabled"):
        return fallback("checkpoint uses memory or checkpoint-configured RTC")
    if not (rows := QUALIFIED.get(kind)):
        return fallback(f"no public checkpoint qualified for {kind}")
    if platform.system() != "Darwin" or platform.machine() != "arm64" or find_spec("mlx") is None:
        return fallback("optional Apple silicon MLX runtime unavailable")

    from importlib.metadata import PackageNotFoundError, version

    compiled = not getattr(args, "no_compile", False)
    machine = machine_model()
    applicable = [row for row in rows if row.backend == capability.name and mode in row.modes
                  and row.machine == machine and row.compiled == compiled]
    if not applicable:
        return fallback(f"no qualified {mode} result on this Mac with these compiler settings")
    try:
        digest = checkpoint_digest(path)
        for row in applicable:
            if row.checkpoint == digest and all(version(name) == expected for name, expected in row.packages):
                return Selection("auto", row.backend, f"qualified {kind} checkpoint for {mode}")
    except (OSError, PackageNotFoundError):
        return fallback("checkpoint or runtime identity could not be confirmed")
    return fallback("checkpoint or runtime versions do not match published results")
