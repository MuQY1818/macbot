"""``macbot doctor``: check that this Mac is ready, without touching the robot."""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
from pathlib import Path

from .config import Config
from .system import user_home

OK, WARN, BAD = "ok ", "!  ", "x  "


def _line(mark, what, detail=""):
    print(f"  {mark} {what:<28s} {detail}")


def run(cfg: Config):
    bad = 0
    print(f"macOS {platform.mac_ver()[0]} on {platform.machine()}, Python {platform.python_version()}")
    print("\npackages")
    for mod, need in [("lerobot", True), ("torch", True), ("cv2", True), ("yaml", True), ("pyorbbecsdk", False),
                      ("transformers", False)]:
        found = importlib.util.find_spec(mod) is not None
        ver = ""
        if found:
            try:
                ver = getattr(__import__(mod), "__version__", "")
            except Exception as e:                      # noqa: BLE001
                ver = f"import failed: {e}"
        _line(OK if found else (BAD if need else WARN), mod, ver or ("missing" + ("" if need else " (optional)")))
        bad += need and not found

    print("\ncompute")
    try:
        import torch
        mps = torch.backends.mps.is_available()
        _line(OK if mps else BAD, "PyTorch MPS", "available" if mps else "not available: policies would run on CPU")
        bad += not mps
    except Exception as e:                              # noqa: BLE001
        _line(BAD, "PyTorch", str(e))
        bad += 1

    print("\nrobot")
    lib = Path("/usr/local/lib/libPCBUSB.dylib")
    _line(OK if lib.exists() else BAD, "PCAN-USB driver (PCBUSB)", str(lib) if lib.exists() else "install it, see docs/setup_macos.md")
    bad += not lib.exists()
    from .robot.rebot import find_leader_port
    try:
        _line(OK, "leader arm (USB serial)", find_leader_port())
    except Exception as e:                              # noqa: BLE001
        _line(WARN, "leader arm (USB serial)", str(e))
    calib = user_home() / ".cache/huggingface/lerobot/calibration/robots"
    found = sorted(f"{p.parent.name}/{p.name}" for p in calib.glob("*/*.json")) if calib.is_dir() else []
    _line(OK if found else WARN, "follower calibration", ", ".join(found) or f"none in {calib}")
    urdf = cfg["robot"].get("urdf")
    _line(OK if urdf else WARN, "URDF for gravity feedforward", urdf or "not set: gravity compensation stays off")

    print("\ncameras")
    sw = shutil.which("swiftc")
    _line(OK if sw else BAD, "swiftc (Xcode tools)", sw or "run: xcode-select --install")
    bad += not sw
    if sw:
        from .cameras import list_cameras, resolve_uid
        try:
            cams = list_cameras()
            ids = {c["unique_id"] for c in cams}
            for c in cams:
                _line(OK, c["name"][:28], c["unique_id"])
            for name, c in cfg["cameras"].items():
                if c["type"] == "uvc":
                    uid = resolve_uid(str(c["unique_id"]))
                    _line(OK if uid in ids else WARN, f"config: {name}",
                          uid if uid in ids else f"uniqueID {c['unique_id']} is not connected (see macbot cameras)")
        except Exception as e:                          # noqa: BLE001
            _line(WARN, "camera list", str(e))
    if any(c["type"] == "orbbec" for c in cfg["cameras"].values()):
        root = os.geteuid() == 0
        _line(OK if root else WARN, "Orbbec camera", "running as root" if root else "needs sudo -E (the Orbbec SDK opens USB as root)")

    print("\nfolders")
    _line(OK, "data_root", str(cfg.root))
    print("\n" + ("all required checks passed" if not bad else f"{bad} required check(s) failed"))
    return 1 if bad else 0
