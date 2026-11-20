"""Any follower / leader that LeRobot 0.7 provides, configured by its LeRobot type name.

    robot:
      type: lerobot
      follower: {type: so101_follower, port: /dev/tty.usbmodem5A46081, id: my_follower}
      leader:   {type: so101_leader,   port: /dev/tty.usbmodem5A46082, id: my_leader}
      limits: {shoulder_pan: [-110, 110], gripper: [0, 100]}     # optional, robot units

Every key under ``follower`` / ``leader`` is passed to the LeRobot config class, so anything
``lerobot-teleoperate --robot.xxx=...`` accepts works here. Cameras are always handled by macbot,
so the LeRobot ``cameras`` field is left empty.

Like the reBot profile, macbot never calibrates: it connects with ``calibrate=False`` and refuses to
connect a device that has no calibration file. Calibrate once with ``lerobot-calibrate``.
On quit the follower moves slowly back to the pose it had when macbot connected (start from a rest
pose), then disconnects; most LeRobot followers disable torque on disconnect.
"""
from __future__ import annotations

import importlib
import pkgutil

from .base import Robot

_loaded = False


def _load_registry():
    """LeRobot registers device types when their packages are imported; import them all once.
    Packages whose drivers are missing are skipped (their types then stay unknown)."""
    global _loaded
    if _loaded:
        return
    import lerobot.robots as R
    import lerobot.teleoperators as T
    for pkg in (R, T):
        for m in pkgutil.iter_modules(pkg.__path__):
            try:
                importlib.import_module(f"{pkg.__name__}.{m.name}")
            except Exception:                       # noqa: BLE001  (optional driver not installed)
                pass
    _loaded = True


def known_types() -> tuple[list[str], list[str]]:
    _load_registry()
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig
    return sorted(RobotConfig.get_known_choices()), sorted(TeleoperatorConfig.get_known_choices())


def _config(base, spec: dict):
    import dataclasses
    spec = dict(spec)
    typ = spec.pop("type")
    try:
        cls = base.get_choice_class(typ)
    except KeyError:
        raise SystemExit(f"LeRobot has no device type '{typ}' here (driver not installed?). "
                         f"Run `macbot robots` to list the available types.") from None
    names = {f.name for f in dataclasses.fields(cls)}
    if "cameras" in names:
        spec.setdefault("cameras", {})
    unknown = set(spec) - names
    if unknown:
        raise SystemExit(f"{typ} does not take {sorted(unknown)}; it takes {sorted(names)}")
    return cls(**spec)


def make_follower(spec: dict):
    _load_registry()
    from lerobot.robots.config import RobotConfig
    from lerobot.robots.utils import make_robot_from_config
    return make_robot_from_config(_config(RobotConfig, spec))


def make_leader(spec: dict):
    _load_registry()
    from lerobot.teleoperators.config import TeleoperatorConfig
    from lerobot.teleoperators.utils import make_teleoperator_from_config
    return make_teleoperator_from_config(_config(TeleoperatorConfig, spec))


def joint_names(device) -> list[str]:
    """Position joints from a LeRobot device's action features, gripper moved last."""
    names = [k.removesuffix(".pos") for k in device.action_features if k.endswith(".pos")]
    grip = [n for n in names if "gripper" in n]
    return [n for n in names if n not in grip] + grip


class LeRobotArm(Robot):
    kind = "lerobot"

    def __init__(self, rcfg: dict, need_leader: bool):
        super().__init__()
        if not rcfg.get("follower"):
            raise SystemExit("robot.type lerobot needs robot.follower: {type: ..., port: ..., id: ...}")
        self.spec = rcfg
        self.follower = make_follower(rcfg["follower"])
        if need_leader:
            if not rcfg.get("leader"):
                raise SystemExit("teleoperation needs robot.leader: {type: ..., port: ..., id: ...}")
            self.leader = make_leader(rcfg["leader"])
        self.joints = joint_names(self.follower)
        grip = [n for n in self.joints if "gripper" in n]
        self.gripper = grip[0] if len(grip) == 1 else None
        self.limits = {n: tuple(v) for n, v in (rcfg.get("limits") or {}).items() if n in self.joints}
        if self.leader is not None:
            lj = joint_names(self.leader)
            if set(lj) != set(self.joints):
                raise SystemExit(f"leader joints {lj} do not match follower joints {self.joints}")
        for dev in (self.follower, self.leader):
            if dev is not None and not dev.is_calibrated:
                raise SystemExit(f"missing calibration file {dev.calibration_fpath}; run lerobot-calibrate first "
                                 "(macbot never calibrates on its own)")

    def info(self):
        return {"kind": self.kind, "follower": dict(self.spec["follower"], port="..."),
                "leader": dict(self.spec["leader"], port="...") if self.spec.get("leader") else None}
