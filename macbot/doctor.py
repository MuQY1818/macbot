"""Check the selected macbot setup without opening robot connections."""
from __future__ import annotations

import importlib
import os
import platform
import shutil

from .config import Config
from .system import PCBUSB, expand, user_home

OK, WARN, BAD = "ok ", "!  ", "x  "


def _line(mark, what, detail=""):
    print(f"  {mark} {what:<28s} {detail}")


def _import(module, hint=""):
    try:
        pkg = importlib.import_module(module)
    except Exception as exc:
        _line(BAD, module, f"import failed: {exc}" + (f"; {hint}" if hint else ""))
        return None
    _line(OK, module, getattr(pkg, "__version__", "imported"))
    return pkg


def _check_rebot(rcfg):
    found = PCBUSB.is_file()
    _line(OK if found else BAD, "PCAN-USB driver (PCBUSB)", str(PCBUSB) if found else "install it; see docs/setup.md")
    from .robot.rebot import find_leader_port

    bad = int(not found)
    try:
        configured = rcfg.get("leader_port")
        port = configured or find_leader_port()
        exists = not configured or expand(port).exists()
        _line(OK if exists else BAD, "leader arm (USB serial)", port)
        bad += not exists
    except RuntimeError as exc:
        _line(WARN, "leader arm (USB serial)", str(exc))
    calib = user_home() / ".cache/huggingface/lerobot/calibration/robots"
    follower = calib / "rebot_b601_follower" / f"{rcfg.get('follower_id', 'follower1')}.json"
    _line(OK if follower.is_file() else WARN, "follower calibration", str(follower))
    urdf = rcfg.get("urdf")
    _line(OK if urdf else WARN, "URDF for gravity feedforward", urdf or "not set: gravity compensation stays off")
    return bad


def _driver_name(config):
    """Find the driver by its registered config, without running a device constructor."""
    cls = type(config)
    module = cls.__module__
    package, _, leaf = module.rpartition(".")
    if leaf.startswith("config_"):
        module = f"{package}.{leaf.removeprefix('config_')}"
    drivers = {
        driver for driver in vars(importlib.import_module(module)).values()
        if isinstance(driver, type) and driver.__module__ == module
        and isinstance(getattr(driver, "config_class", None), type) and isinstance(config, driver.config_class)
    }
    # SO-100/101 share a driver and use its name, not the config type, in calibration paths.
    if len(drivers) != 1:
        raise ImportError(f"could not identify the driver for {cls.__name__} in {module}")
    return drivers.pop().name


def calibration_path(role, name, device_id, directory=None):
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS, TELEOPERATORS

    group = ROBOTS if role == "follower" else TELEOPERATORS
    directory = expand(directory) if directory else HF_LEROBOT_CALIBRATION / group / name
    return directory / f"{device_id}.json"


def _check_lerobot_device(role, spec):
    from .robot.lerobot_generic import device_config

    if not spec:
        _line(BAD if role == "follower" else WARN, f"{role} driver",
              f"robot.{role} is not set" + (" (needed for teleoperation)" if role == "leader" else ""))
        return int(role == "follower")
    try:
        config = device_config(spec, leader=role == "leader")
        name = _driver_name(config)
    except (Exception, SystemExit) as exc:
        _line(BAD, f"{role} driver", f"{exc}; check the device type and its LeRobot extra")
        return 1
    _line(OK, f"{role} driver", name)

    bad = 0
    if name in {"so_follower", "so_leader", "koch_follower", "koch_leader", "omx_follower", "omx_leader"}:
        bad += _import("rustypot", "install the LeRobot serial-motors extra") is None

    port = getattr(config, "port", None)
    if port and (not getattr(config, "can_interface", None) or str(port).startswith("/")):
        exists = expand(port).exists()
        _line(OK if exists else BAD, f"{role} serial port",
              str(port) if exists else f"{port} not found; run lerobot-find-port")
        bad += not exists
    elif port:
        _line(WARN, f"{role} port", f"{port} (CAN channel; not probed)")

    path = calibration_path(role, name, config.id, config.calibration_dir)
    exists = path.is_file()
    _line(OK if exists else BAD, f"{role} calibration",
          str(path) if exists else f"{path} not found; run lerobot-calibrate")
    return bad + int(not exists)


def _check_sim(rcfg):
    from pathlib import Path

    from .robot.sim import cached_envhub_path, envhub_proxy_hint, sim_config

    try:
        config = sim_config(rcfg.get("env"), rcfg.get("task_id"))
    except (Exception, SystemExit) as exc:
        _line(BAD, "LeRobot simulator", f"{exc}; see configs/so101_sim.yaml")
        return 1
    _line(OK, "LeRobot simulator", f"{config.env.type} (local configuration)")
    hub_path = getattr(config.env, "hub_path", None)
    proxy_bad = 0
    if hub_path and (hint := envhub_proxy_hint()) and not Path(hub_path.partition(":")[0]).is_dir():
        if cached_envhub_path(hub_path):
            _line(OK, "EnvHub cache", "using local files; no SOCKS client required")
        else:
            _line(BAD, "EnvHub download", hint)
            proxy_bad = 1
    if str(hub_path).split("@", 1)[0] == "lerobot/so101-tabletop-mujoco":
        return proxy_bad + int(_import("mujoco", "install macbot[sim]") is None)
    _line(WARN, "simulator backend", "scene-specific dependencies were not checked; see your scene's setup")
    return proxy_bad


def _check_uvc(cameras):
    sw = shutil.which("swiftc")
    _line(OK if sw else BAD, "swiftc (Xcode tools)", sw or "run: xcode-select --install")
    if not sw:
        return 1
    from .cameras import list_cameras, resolve_uid

    bad = 0
    try:
        if any(c.get("exact_id") for c in cameras.values()):
            from .cameras.discovery import uvc_inventory

            cams = uvc_inventory()
        else:
            cams = list_cameras()
        ids = {c["unique_id"] for c in cams}
        for c in cams:
            _line(OK, c["name"][:28], c["unique_id"])
        for name, c in cameras.items():
            if c["type"] == "uvc":
                exact = c.get("exact_id", False)
                uid = c["unique_id"] if exact else resolve_uid(str(c["unique_id"]))
                _line(OK if uid in ids else BAD if exact else WARN, f"config: {name}",
                      uid if uid in ids else f"uniqueID {c['unique_id']} is not connected (see macbot cameras)")
                if exact and uid not in ids:
                    bad += 1
    except Exception as exc:
        strict = any(c.get("exact_id") for c in cameras.values())
        _line(BAD if strict else WARN, "camera list", str(exc))
        bad += strict
    return bad


def run(cfg: Config):
    system = platform.mac_ver()[0]
    host = f"macOS {system}" if system else platform.system()
    print(f"{host} on {platform.machine()}, Python {platform.python_version()}")
    print("\npackages")
    lerobot = _import("lerobot")
    torch = _import("torch")
    bad = sum(pkg is None for pkg in (lerobot, torch, _import("cv2"), _import("yaml")))

    print("\ncompute")
    if torch is not None:
        mps = torch.backends.mps.is_available()
        _line(OK if mps else WARN, "PyTorch MPS", "available" if mps else "not available; use --device cpu")

    rcfg = cfg["robot"]
    kind = rcfg.get("type", "rebot_b601")
    print(f"\nrobot ({kind})")
    if kind == "rebot_b601":
        bad += _check_rebot(rcfg)
    elif kind == "lerobot":
        if lerobot is not None:
            for role in ("follower", "leader"):
                bad += _check_lerobot_device(role, rcfg.get(role))
    elif kind == "sim":
        if lerobot is not None:
            bad += _check_sim(rcfg)
    else:
        _line(BAD, "robot.type", f"{kind!r} is unknown; use rebot_b601, lerobot, or sim")
        bad += 1

    print("\ncameras")
    cameras = cfg["cameras"]
    types = {c["type"] for c in cameras.values()}
    if kind == "sim" and types - {"robot"}:
        _line(BAD, "camera config", "simulation requires cameras with type: robot")
        bad += 1
    elif "uvc" in types:
        bad += _check_uvc(cameras)
    if kind != "sim" and "orbbec" in types:
        sdk = _import("pyorbbecsdk", "install the SDK; see the quickstart")
        bad += sdk is None
        root = os.geteuid() == 0
        _line(OK if root else WARN, "Orbbec camera", "running as root" if root else "run with sudo -E on macOS")
        selected = {name: c["serial"] for name, c in cameras.items() if c["type"] == "orbbec" and c.get("serial")}
        if sdk is not None and selected:
            from .cameras import orbbec_devices, orbbec_serials

            try:
                context, devices = orbbec_devices()
                serials = orbbec_serials(devices)
                for name, serial in selected.items():
                    found = serials.count(serial) == 1
                    _line(OK if found else BAD, f"config: {name}", serial if found else f"{serial} not found uniquely")
                    bad += not found
                del devices, context
            except Exception as exc:
                _line(BAD, "Orbbec device list", str(exc))
                bad += 1
    if kind == "sim" and "robot" in types:
        _line(OK, "robot cameras", "frames supplied by the simulator")
    if not types:
        _line(WARN, "camera config", "no cameras configured")

    print("\nfolders")
    _line(OK, "data_root", str(cfg.root))
    print("\n" + ("all required checks passed" if not bad else f"{bad} required check(s) failed"))
    return 1 if bad else 0
