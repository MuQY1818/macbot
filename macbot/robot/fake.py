"""Simulated follower and leader for `--fake` runs: no hardware, same interface."""
from __future__ import annotations

import math
import time

from ..config import JOINTS

LIMITS = {"shoulder_pan": (-145, 145), "shoulder_lift": (-170, 0), "elbow_flex": (-200, 0),
          "wrist_flex": (-90, 80), "wrist_yaw": (-90, 90), "wrist_roll": (-90, 90), "gripper": (-270, 0)}


class FakeFollower:
    """First-order lag per joint (time constant 70 ms), roughly like the real servo loop."""

    class _Cfg:
        joint_limits = LIMITS

    def __init__(self, tau: float = 0.07):
        self.config, self.tau = self._Cfg(), tau
        self.pos = {n: 0.0 for n in JOINTS}
        self.cmd = dict(self.pos)
        self.t = None
        self.feedforward = {"fake": True}

    def connect(self, calibrate=False):
        pass

    def send_action(self, a):
        for k, v in a.items():
            n = k.removesuffix(".pos")
            lo, hi = LIMITS[n]
            self.cmd[n] = max(lo, min(hi, float(v)))
        return {f"{n}.pos": self.cmd[n] for n in JOINTS}

    def get_observation(self):
        now = time.perf_counter()
        dt = 0.0 if self.t is None else now - self.t
        self.t = now
        for n in JOINTS:
            self.pos[n] += (self.cmd[n] - self.pos[n]) * min(1.0, dt / self.tau)
        return {f"{n}.pos": self.pos[n] for n in JOINTS}

    def disconnect(self):
        pass


class FakeLeader:
    """Leader that waves slowly."""

    def connect(self, calibrate=False):
        self.t0 = time.time()

    def get_action(self):
        t = time.time() - self.t0
        a = {f"{n}.pos": 25 * math.sin(0.4 * t + i) * (i < 6) for i, n in enumerate(JOINTS)}
        a["shoulder_lift.pos"] -= 40
        a["elbow_flex.pos"] -= 60
        a["gripper.pos"] = -30 + 30 * math.sin(0.7 * t)
        return a

    def disconnect(self):
        pass
