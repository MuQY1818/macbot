"""Local LeRobot simulation with absolute joint position control."""
from __future__ import annotations

import math
import time

import cv2

from .base import Robot
from .lerobot_generic import _config, joint_names


def _make_sim(spec, task_id):
    from lerobot.robots.sim_robot import SimRobot, SimRobotConfig
    from lerobot.robots.sim_robot.config_sim_robot import SimulatedEnvConfig

    if not spec or not spec.get("type"):
        raise SystemExit("robot.type sim needs robot.env: {type: ..., ...}")
    if spec.get("server"):
        raise SystemExit("macbot sim supports only local environments; remove robot.env.server")
    env = _config(SimulatedEnvConfig, spec)
    try:
        return SimRobot(SimRobotConfig(env=env, task_id=task_id))
    except ImportError as exc:
        raise SystemExit(f"cannot load simulator: {exc}. For SO-101 install macbot[sim].") from exc


class SimArm(Robot):
    kind = "sim"

    def __init__(self, cfg):
        super().__init__()
        if cfg.get("leader") or cfg.get("follower"):
            raise SystemExit("robot.type sim uses robot.env and a scripted leader")
        self.follower = _make_sim(cfg.get("env"), cfg.get("task_id"))
        try:
            desc = self.follower.descriptor
            self.joints = joint_names(self.follower)
            keys = {f"{n}.pos" for n in self.joints}
            if (desc.control_mode != "position" or not keys or keys != set(self.follower.action_features)
                    or not keys <= set(self.follower.observation_features)):
                raise SystemExit("macbot sim needs absolute joint position actions and observations with matching .pos names")
            grip = [n for n in self.joints if "gripper" in n]
            self.gripper = grip[0] if len(grip) == 1 else None
            self.limits = {n: tuple(v) for n, v in (cfg.get("limits") or {}).items() if n in self.joints}
            self.fps = desc.fps
            self.camera_shapes = {k: self.follower.observation_features[k] for k in self.follower.cameras}
            self._frames = {}
            self._q = []
        except BaseException:
            self.follower.disconnect()
            raise

    def check_rate(self, hz, setting):
        if not math.isclose(hz, self.fps):
            raise SystemExit(f"set {setting}: {self.fps:g} to match the simulator control rate")

    def _publish(self):
        obs = self.follower.get_observation()
        self._q = self._vec(obs)
        stamp = time.time()
        self._frames = {k: (cv2.cvtColor(obs[k], cv2.COLOR_RGB2BGR), stamp) for k in self.camera_shapes}

    def _set_start(self):
        self._publish()
        self._start = self.read()
        self._t0 = time.monotonic()

    def connect(self, leader=False):
        self.follower.connect(calibrate=False)
        self._set_start()

    def reset_scene(self):
        self.follower.start_episode()
        self._set_start()

    @property
    def ended(self):
        return self.follower.episode_ended

    def camera_frame(self, key):
        return self._frames.get(key, (None, 0.0))

    def read(self):
        return list(self._q)

    def read_leader(self):
        elapsed = time.monotonic() - self._t0
        return [self.clamp(n, q + (12 if n == self.gripper else 18)
                           * math.sin(0.8 * elapsed) * math.cos(0.4 * elapsed + i))
                for i, (n, q) in enumerate(zip(self.joints, self._start, strict=False))]

    def send(self, target, measured=None):
        sent = [self.clamp(n, v) for n, v in zip(self.joints, target, strict=True)]
        if not self.ended:
            sent = super().send(sent, measured)
            self._publish()
        return sent

    def shutdown(self, moved=True):
        self.follower.disconnect()

    def info(self):
        return {"kind": self.kind, "robot_type": self.follower.robot_type,
                "semantics": self.follower.descriptor.semantics}
