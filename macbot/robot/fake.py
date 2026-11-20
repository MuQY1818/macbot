"""Simulated follower and leader for ``--fake`` runs: no hardware, same interface as a real robot."""
from __future__ import annotations

import math
import time

from .base import Robot
from .telemetry import synthetic

REBOT_LIMITS = {"shoulder_pan": (-145, 145), "shoulder_lift": (-170, 0), "elbow_flex": (-200, 0),
                "wrist_flex": (-90, 80), "wrist_yaw": (-90, 90), "wrist_roll": (-90, 90), "gripper": (-270, 0)}
SO_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class _Follower:
    """First-order lag per joint (time constant 70 ms), roughly like a real servo loop."""

    def __init__(self, joints, limits, tau=0.07):
        self.joints, self.limits, self.tau = joints, limits, tau
        self.pos = {n: 0.0 for n in joints}
        self.cmd = dict(self.pos)
        self.t = None

    def connect(self, calibrate=False):
        pass

    def send_action(self, a):
        for k, v in a.items():
            n = k.removesuffix(".pos")
            lo, hi = self.limits.get(n, (-math.inf, math.inf))
            self.cmd[n] = max(lo, min(hi, float(v)))
        return {f"{n}.pos": self.cmd[n] for n in self.joints}

    def get_observation(self):
        now = time.perf_counter()
        dt = 0.0 if self.t is None else now - self.t
        self.t = now
        for n in self.joints:
            self.pos[n] += (self.cmd[n] - self.pos[n]) * min(1.0, dt / self.tau)
        return {f"{n}.pos": self.pos[n] for n in self.joints}

    def disconnect(self):
        pass


class _Leader:
    """Leader that waves slowly inside the joint limits."""

    def __init__(self, joints, limits):
        self.joints, self.limits = joints, limits

    def connect(self, calibrate=False):
        self.t0 = time.time()

    def get_action(self):
        t = time.time() - self.t0
        out = {}
        for i, n in enumerate(self.joints):
            lo, hi = self.limits.get(n, (-90, 90))
            mid, amp = (lo + hi) / 2, min(25.0, (hi - lo) / 4)
            if n == "gripper":
                amp = (hi - lo) / 6
            out[f"{n}.pos"] = mid + amp * math.sin((0.7 if n == "gripper" else 0.4) * t + i)
        return out

    def disconnect(self):
        pass


class FakeArm(Robot):
    kind = "fake"

    def __init__(self, joints, limits, robot_type):
        super().__init__()
        self.joints = list(joints)
        grips = [n for n in joints if "gripper" in n]
        self.gripper = grips[0] if len(grips) == 1 else None
        self.limits = dict(limits)
        self.robot_type = robot_type
        self.follower, self.leader = _Follower(self.joints, self.limits), _Leader(self.joints, self.limits)
        self._telemetry = synthetic(self.joints)

    def motor_telemetry(self):
        return self._telemetry

    def info(self):
        return {"kind": self.kind, "joints": self.joints, "robot_type": self.robot_type}

    def shutdown(self, moved: bool = True):
        pass


def fake_for(rcfg: dict) -> FakeArm:
    """Use joint layouts without importing or constructing device drivers."""
    kind = rcfg.get("type", "rebot_b601")
    if kind == "rebot_b601":
        return FakeArm(list(REBOT_LIMITS), REBOT_LIMITS, "rebot_b601_follower")
    spec = rcfg.get("follower", {})
    typ = spec.get("type") if kind == "lerobot" else "so_follower"
    joints = rcfg.get("joints")
    if joints is None:
        if typ in {"so100_follower", "so101_follower", "so_follower", "koch_follower", "omx_follower"}:
            joints = SO_JOINTS
        elif typ == "openarm_follower":
            joints = list(spec.get("motor_config") or [*(f"joint_{i}" for i in range(1, 8)), "gripper"])
        elif typ == "yam_follower":
            joints = [*(f"joint_{i}" for i in range(1, 7)), "gripper"]
        else:
            raise SystemExit(f"--fake needs robot.joints for follower type {typ!r}")
    if not isinstance(joints, list) or not joints or not all(isinstance(n, str) and n for n in joints) or len(set(joints)) != len(joints):
        raise SystemExit("robot.joints must list unique, nonempty joint names")
    joints = sorted(joints, key=lambda n: "gripper" in n)
    limits = {n: (-150.0, 150.0) for n in joints}
    if "gripper" in joints:
        limits["gripper"] = (0.0, 100.0)
    limits.update({n: tuple(v) for n, v in (rcfg.get("limits") or {}).items() if n in joints})
    return FakeArm(joints, limits, typ or "fake")
