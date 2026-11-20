"""One interface for every arm macbot drives.

The consoles never talk to LeRobot classes directly. They use a ``Robot``:

    r.joints            joint names in a fixed order, gripper last if there is one
    r.gripper           the gripper joint name, or None
    r.limits            {name: (lo, hi)} in robot units, or {} if unknown
    r.connect(leader)   connect the follower (and the leader for teleoperation)
    r.read()            measured follower joints, list in ``joints`` order
    r.read_leader()     leader joints, same order
    r.send(target, q)   send a joint target; returns what was actually commanded
    r.home_pose()       where "home" goes: zero for reBot, the pose at connect time otherwise
    r.shutdown(moved)   slowly return to the home pose (if the arm moved), then disconnect
    r.info()            settings recorded into every episode

Joint values are in the units of the LeRobot robot class: degrees for most arm joints when the
robot supports ``use_degrees``, 0-100 for many grippers.
"""
from __future__ import annotations

import math
import time


class Robot:
    kind = "base"

    def __init__(self):
        self.joints: list[str] = []
        self.gripper: str | None = None
        self.limits: dict = {}
        self.follower = None
        self.leader = None
        self._start = None

    # ---- helpers shared by all implementations
    @property
    def arm(self) -> list[str]:
        return [n for n in self.joints if n != self.gripper]

    @property
    def gripper_index(self) -> int | None:
        return self.joints.index(self.gripper) if self.gripper else None

    def clamp(self, name: str, v: float) -> float:
        lo, hi = self.limits.get(name, (-math.inf, math.inf))
        return min(hi, max(lo, v))

    def _vec(self, d: dict) -> list[float]:
        return [float(d[f"{n}.pos"]) for n in self.joints]

    def _dict(self, v) -> dict:
        return {f"{n}.pos": float(x) for n, x in zip(self.joints, v)}

    # ---- interface
    def connect(self, leader: bool = False):
        if leader:
            self.leader.connect(calibrate=False)
        self.follower.connect(calibrate=False)
        self._start = self.read()

    def read(self) -> list[float]:
        return self._vec(self.follower.get_observation())

    def read_leader(self) -> list[float]:
        return self._vec(self.leader.get_action())

    def send(self, target, measured=None) -> list[float]:
        sent = self.follower.send_action(self._dict(target))
        return [float(sent.get(f"{n}.pos", t)) for n, t in zip(self.joints, target)]

    def home_pose(self) -> list[float]:
        return list(self._start) if self._start is not None else [0.0] * len(self.joints)

    def info(self) -> dict:
        return {"kind": self.kind}

    def move_slowly(self, goal, step: float = 0.5, dt: float = 0.02):
        """Interpolate to ``goal`` at ``step`` units per ``dt`` for the joint that has the farthest to go."""
        q = self.read()
        n = max(1, math.ceil(max(abs(g - x) for g, x in zip(goal, q)) / step))
        for k in range(1, n + 1):
            self.send([x + (g - x) * k / n for x, g in zip(q, goal)], None)
            time.sleep(dt)
        time.sleep(0.3)

    def shutdown(self, moved: bool = True):
        try:
            if moved and self._start is not None:
                self.move_slowly(self.home_pose())
        finally:
            for dev in (self.leader, self.follower):
                if dev is not None:
                    try:
                        dev.disconnect()
                    except Exception as e:         # noqa: BLE001  (already disconnected, or never connected)
                        print(f"disconnect: {e}", flush=True)
