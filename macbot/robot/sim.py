"""Local LeRobot simulation with absolute joint position control."""
from __future__ import annotations

import math
import time
from importlib.util import find_spec
from pathlib import Path
from urllib.request import getproxies

import cv2

from .base import Robot
from .lerobot_generic import _config, joint_names
from .sim_render import TABLETOP_REPO, tabletop_renderer, use_tabletop_cgl
from .telemetry import synthetic


def sim_config(spec, task_id):
    from lerobot.robots.sim_robot import SimRobotConfig
    from lerobot.robots.sim_robot.config_sim_robot import SimulatedEnvConfig

    if not spec or not spec.get("type"):
        raise SystemExit("robot.type sim needs robot.env: {type: ..., ...}")
    if spec.get("server"):
        raise SystemExit("macbot sim supports only local environments; remove robot.env.server")
    return SimRobotConfig(env=_config(SimulatedEnvConfig, spec), task_id=task_id)


SOCKS_HINT = "The EnvHub scene is not fully cached and the SOCKS proxy needs socksio. Install `pip install 'httpx[socks]'`."


def cached_envhub_path(hub_path):
    """Use LeRobot's local-directory syntax to avoid constructing a Hub HTTP client."""
    from huggingface_hub import try_to_load_from_cache

    repo_rev, _, filename = hub_path.partition(":")
    if Path(repo_rev).is_dir() or repo_rev.count("/") != 1:
        return None
    repo, _, revision = repo_rev.partition("@")
    filename = filename or "env.py"
    cached = try_to_load_from_cache(repo, filename, revision=revision or None)
    if not isinstance(cached, str):
        return None
    root = Path(cached)
    for _ in Path(filename).parts:
        root = root.parent
    if repo == TABLETOP_REPO:
        # The scene's env.py fetches a snapshot itself if its XML is absent.
        if not (root / "so101_tabletop.xml").is_file() or not (root / "robotstudio_so101/so101.xml").is_file():
            return None
    return f"{root}:{filename}"


def envhub_proxy_hint():
    proxies = getproxies()
    if any(proxies.get(k, "").lower().startswith(("socks5:", "socks5h:")) for k in ("all", "http", "https")):
        if find_spec("socksio") is None:
            return SOCKS_HINT
    return None


def _make_sim(spec, task_id):
    try:
        from lerobot.robots.sim_robot import SimRobot

        config = sim_config(spec, task_id)
        env = config.env
        hub_path = getattr(env, "hub_path", None)
        if hub_path and env.trust_remote_code and not Path(hub_path.partition(":")[0]).is_dir():
            if local := cached_envhub_path(hub_path):
                env.hub_path = local
            elif hint := envhub_proxy_hint():
                raise SystemExit(hint)
        return SimRobot(config)
    except ImportError as exc:
        if "socksio" in str(exc):
            raise SystemExit(SOCKS_HINT) from exc
        raise SystemExit(f"cannot load simulator: {exc}. For SO-101 install macbot[sim].") from exc


class SimArm(Robot):
    kind = "sim"

    def __init__(self, cfg, teleop=None):
        super().__init__()
        if cfg.get("leader") or cfg.get("follower"):
            raise SystemExit("robot.type sim uses robot.env; select the input with teleop.type")
        self.teleop = dict(teleop or {})
        self.teleop_type = self.teleop.get("type", "scripted")
        self.cartesian = None
        self._leader_connected = False
        self.tabletop_cgl = use_tabletop_cgl(cfg.get("env"))
        if self.teleop_type not in {"scripted", "keyboard", "so101_leader"}:
            raise SystemExit("sim teleop.type must be scripted, keyboard, or so101_leader")
        self.follower = _make_sim(cfg.get("env"), cfg.get("task_id"))
        try:
            desc = self.follower.descriptor
            self.joints = joint_names(self.follower)
            self._telemetry = synthetic(self.joints)
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
        if leader and self.teleop_type == "so101_leader":
            from .lerobot_generic import make_leader, require_calibration

            options = {k: v for k, v in self.teleop.items()
                       if k not in {"port", "hz", "soft_start_s", "align_max_deg", "leader_port", "seed"}}
            # The top-level port is the collect console port.
            options["port"] = self.teleop.get("leader_port")
            if not options["port"]:
                raise SystemExit("sim SO-101 teleoperation needs the serial device in teleop.port")
            if options.get("use_degrees", True) is not True:
                raise SystemExit("sim requires a SO-101 leader with use_degrees: true")
            self.leader = make_leader(options)
            if set(joint_names(self.leader)) != set(self.joints):
                raise SystemExit("SO-101 leader and simulator joint names do not match")
            require_calibration(self.leader)
            self.leader.connect(calibrate=False)
            self._leader_connected = True
            if not self.leader.is_calibrated:
                raise SystemExit(f"calibration does not match {self.leader}; run lerobot-calibrate first")
        with tabletop_renderer(self.tabletop_cgl):
            self.follower.connect(calibrate=False)
        if leader and self.teleop_type == "keyboard":
            from .sim_cartesian import Cartesian, tabletop_world

            self.cartesian = Cartesian(tabletop_world(self))
        self._set_start()

    def reset_scene(self, seed=None):
        if seed is None:
            self.follower.start_episode()
        else:
            self.follower.start_episode(seed=seed)
        self._set_start()
        if self.cartesian is not None:
            self.cartesian.reset()

    def outcome(self):
        latest = self.follower.latest
        return {"reward": float(latest.reward[0]), "success": bool(latest.is_success[0]),
                "terminated": bool(latest.terminated[0]), "truncated": bool(latest.truncated[0])}

    def episode_limit(self):
        return int(self.follower._env.call("_max_episode_steps")[0])

    @property
    def ended(self):
        return self.follower.episode_ended

    def camera_frame(self, key):
        return self._frames.get(key, (None, 0.0))

    def read(self):
        return list(self._q)

    def motor_telemetry(self):
        return self._telemetry

    def read_leader(self, keys=()):
        if self.leader is not None:
            return super().read_leader()
        if self.cartesian is not None:
            return self.cartesian.keys(keys)
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
        try:
            if self.leader is not None and (self._leader_connected or self.leader.is_connected):
                self.leader.disconnect()
        finally:
            self.follower.disconnect()

    def info(self):
        return {"kind": self.kind, "robot_type": self.follower.robot_type,
                "semantics": self.follower.descriptor.semantics}
