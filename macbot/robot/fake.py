"""Simulated follower and leader for ``--fake`` runs: no hardware, same interface as a real robot."""
from __future__ import annotations

import math
import time

from .base import Robot

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

    def __init__(self, joints, limits):
        super().__init__()
        self.joints = list(joints)
        self.gripper = "gripper" if "gripper" in joints else None
        self.limits = dict(limits)
        self.follower, self.leader = _Follower(self.joints, self.limits), _Leader(self.joints, self.limits)

    def info(self):
        return {"kind": self.kind, "joints": self.joints}

    def shutdown(self, moved: bool = True):
        pass


def fake_for(rcfg: dict) -> FakeArm:
    """A simulated arm with the same joints as the configured robot. For LeRobot types the real device
    class is built (not connected) to read its joints; if its driver is missing, SO-100/101 joints are used."""
    if rcfg.get("type", "rebot_b601") == "rebot_b601":
        return FakeArm(list(REBOT_LIMITS), REBOT_LIMITS)
    joints = SO_JOINTS
    try:
        from .lerobot_generic import joint_names, make_follower
        joints = joint_names(make_follower(dict(rcfg["follower"], port=rcfg["follower"].get("port") or "/dev/null")))
    except BaseException as e:                     # noqa: BLE001  (driver missing, bad config: still simulate)
        print(f"fake: could not build {rcfg.get('follower', {}).get('type')} ({e}); using SO-101 joints", flush=True)
    limits = {n: (-150.0, 150.0) for n in joints}
    if "gripper" in joints:
        limits["gripper"] = (0.0, 100.0)
    limits.update({n: tuple(v) for n, v in (rcfg.get("limits") or {}).items() if n in joints})
    return FakeArm(joints, limits)
