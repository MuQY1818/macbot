"""reBot B601-RS follower and reBot Arm 102 leader on LeRobot 0.7.0, with safety defaults.

What this adds on top of the LeRobot classes
--------------------------------------------
1. **No accidental re-zeroing.** Without a calibration file, ``connect()`` enters calibration and
   one Enter key writes the current pose as the motor zero. We always connect with
   ``calibrate=False`` and refuse to connect if the calibration file is missing.
2. **Safe shutdown.** ``disconnect()`` disables the motors at once and the forearm drops. We first
   move to zero in two stages (pan and wrist joints, then shoulder lift and elbow), at most
   0.5 deg per 20 ms, then disable.
3. **Feedforward.** The stock MIT command uses target velocity 0, so the damping term drags every
   motion (about 100 ms of lag), and gravity is held only by the position error (2-3 deg sag). We
   add velocity feedforward (filtered target velocity) and gravity feedforward (URDF model).
4. **Gripper.** Position control with a lead clamp (the target never leads the measured angle by
   more than ``gripper_lead_deg``), which bounds the squeeze force. The gripper multi-turn counter
   can report one extra turn after power-up; we read the raw position and remove whole turns.
   A small software zero trim is available for grippers whose zero drifted.
"""
from __future__ import annotations

import glob
import math
import time

import numpy as np

from .base import Robot

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]
ARM = JOINTS[:6]

STAGE_1 = ["shoulder_pan", "wrist_flex", "wrist_yaw", "wrist_roll"]
STAGE_2 = ["shoulder_lift", "elbow_flex"]

VEL_LPF_ALPHA = 0.35
VEL_DEADBAND = 2.0        # deg/s
VEL_MAX = 120.0           # deg/s, cap of the feedforward velocity
TAU_FF_MAX = {"shoulder_pan": 4.0, "shoulder_lift": 14.0, "elbow_flex": 10.0,
              "wrist_flex": 3.0, "wrist_yaw": 2.0, "wrist_roll": 1.0}   # N*m
FF_RAMP_S = 1.0
GRIP_WINDOW = (-300.0, 60.0)   # plausible gripper range in public coordinates (360 deg wide)


def _lerobot():
    from lerobot.robots.rebot_b601_follower.config_rebot_b601_follower import RebotB601FollowerRobotConfig
    from lerobot.robots.rebot_b601_follower.motor_family import GRIPPER_MOTOR, ArmControlMode
    from lerobot.robots.rebot_b601_follower.rebot_b601_follower import RebotB601Follower
    return RebotB601FollowerRobotConfig, RebotB601Follower, GRIPPER_MOTOR, ArmControlMode


def find_leader_port() -> str:
    """The leader's CH340 serial port name changes with the USB socket; pick the only one."""
    ports = sorted(glob.glob("/dev/cu.usbserial-*"))
    if len(ports) != 1:
        raise RuntimeError(f"found {len(ports)} usbserial ports {ports}; cannot tell which one is the leader")
    return ports[0]


def _make_ff_class():
    _, RebotB601Follower, GRIPPER_MOTOR, ArmControlMode = _lerobot()

    class FeedforwardFollower(RebotB601Follower):
        """Stock follower plus velocity and gravity feedforward in the MIT command. Same interface."""

        def __init__(self, config, vel_ff: float, grav_comp: float, urdf=None):
            super().__init__(config)
            self.vel_ff, self.grav_comp = float(vel_ff), float(grav_comp)
            self._gm = None
            if self.grav_comp > 0:
                from .gravity import GravityModel
                self._gm = GravityModel(urdf)
            self._prev_goal = self._prev_t = self._t_first = None
            self._vel = {n: 0.0 for n in ARM}
            self._ff = {}

        def send_action(self, action):
            now = time.perf_counter()
            goal = {k.removesuffix(".pos"): float(v) for k, v in action.items() if k.endswith(".pos")}
            goal = {n: min(max(v, self._joint_limits[n][0]), self._joint_limits[n][1]) if n in self._joint_limits else v
                    for n, v in goal.items()}
            if self._t_first is None:
                self._t_first = now
            ramp = min(1.0, (now - self._t_first) / FF_RAMP_S)
            if self._prev_goal is not None and now - self._prev_t > 1e-3:
                dt = now - self._prev_t
                for n in ARM:
                    if n in goal and n in self._prev_goal:
                        raw = (goal[n] - self._prev_goal[n]) / dt
                        self._vel[n] = VEL_LPF_ALPHA * raw + (1 - VEL_LPF_ALPHA) * self._vel[n]
            self._prev_goal, self._prev_t = goal, now
            g = (self._gm.gravity_torque_public([goal[n] for n in ARM])
                 if self._gm is not None and all(n in goal for n in ARM) else np.zeros(6))
            for j, n in enumerate(ARM):
                v = self._vel[n]
                v = math.copysign(max(abs(v) - VEL_DEADBAND, 0.0), v)
                v = max(-VEL_MAX, min(VEL_MAX, v)) * self.vel_ff
                tau = max(-TAU_FF_MAX[n], min(TAU_FF_MAX[n], self.grav_comp * ramp * float(g[j])))
                self._ff[n] = (v, tau)
            return super().send_action(action)

        def _send_joint(self, motor_name, position_deg, state=None):
            ff = self._ff.get(motor_name)
            if motor_name == GRIPPER_MOTOR or ff is None or self.config.control_mode is not ArmControlMode.MIT:
                return super()._send_joint(motor_name, position_deg, state)
            lo, hi = self._joint_limits[motor_name]
            position_deg = max(lo, min(hi, position_deg))
            d = self.profile.joint_directions[motor_name]        # public -> motor sign
            v_pub, tau_pub = ff
            self.motors[motor_name].send_mit(math.radians(position_deg * d), math.radians(v_pub * d),
                                             self._mit_kp[motor_name], self._mit_kd[motor_name], tau_pub * d)
            return position_deg

    return FeedforwardFollower


def read_gripper_raw(channel: str) -> float:
    """Read the gripper position (parameter 0x7019, mechPos) without enabling the motor. Public degrees."""
    from lerobot.robots.rebot_b601_follower.rebot_b601_follower import MotorBridgeController
    bus = MotorBridgeController(channel=channel)
    m = bus.add_robstride_motor(7, 0xFD, "rs-00")
    try:
        return -math.degrees(m.robstride_get_param_f32(0x7019, 500))   # RS direction sign is -1
    finally:
        m.close()
        bus.close()


def apply_gripper_offset(robot, rcfg: dict) -> float:
    """Remove whole extra turns of the gripper counter and apply the software zero trim."""
    _, _, GRIPPER_MOTOR, _ = _lerobot()
    raw = read_gripper_raw(robot.config.port)
    turns = -360.0 * math.floor((raw - GRIP_WINDOW[0]) / 360.0)
    trim = float(rcfg.get("gripper_zero_offset_deg") or 0.0)
    off = turns + trim
    print(f"gripper raw {raw:.1f} deg, turn correction {turns:+.0f}, zero trim {trim:+.1f} -> {raw + off:.1f} deg", flush=True)
    robot.gripper_offset = off
    if off:
        to_pub, to_mot = robot._motor_to_public_position, robot._public_to_motor_position
        robot._motor_to_public_position = lambda n, d: to_pub(n, d) + (off if n == GRIPPER_MOTOR else 0.0)
        robot._public_to_motor_position = lambda n, p: to_mot(n, p - (off if n == GRIPPER_MOTOR else 0.0))
    return off


def make_follower(rcfg: dict):
    """Build (not connect) the follower described by the ``robot`` section of the config."""
    Cfg, RebotB601Follower, _, _ = _lerobot()
    extra = {}
    if rcfg.get("gripper_mode", "mit") == "mit":
        extra["gripper_control_mode"] = "mit"
    elif rcfg["gripper_mode"] != "impedance":
        raise ValueError(f"unknown gripper_mode {rcfg['gripper_mode']}")
    cfg = Cfg(port=rcfg.get("can_channel", "can0"), id=rcfg.get("follower_id", "follower1"), motor_family="rs",
              can_adapter="socketcan", cameras={}, **extra)
    vel_ff = float(rcfg.get("velocity_feedforward") or 0.0)
    grav = float(rcfg.get("gravity_compensation") or 0.0)
    if grav > 0 and not rcfg.get("urdf"):
        print("gravity compensation needs robot.urdf; running without it", flush=True)
        grav = 0.0
    if vel_ff == 0 and grav == 0:
        robot = RebotB601Follower(cfg)
    else:
        from ..system import expand
        robot = _make_ff_class()(cfg, vel_ff, grav, expand(rcfg["urdf"]) if grav > 0 else None)
    robot.feedforward = {"velocity": vel_ff, "gravity": grav}
    print(f"follower feedforward: velocity {vel_ff}, gravity {grav}", flush=True)
    if not robot.is_calibrated:
        raise RuntimeError(f"missing calibration file {robot.calibration_fpath}; not connecting (avoids re-zeroing)")
    apply_gripper_offset(robot, rcfg)
    return robot


def make_leader(port: str | None = None, leader_id: str = "rebot_arm_102_leader"):
    from lerobot.teleoperators.rebot_102_leader.config_rebot_102_leader import RebotArm102LeaderTeleopConfig
    from lerobot.teleoperators.rebot_102_leader.rebot_102_leader import RebotArm102Leader
    leader = RebotArm102Leader(RebotArm102LeaderTeleopConfig(port=port or find_leader_port(), id=leader_id))
    if not leader.is_calibrated:
        raise RuntimeError(f"missing calibration file {leader.calibration_fpath}; not connecting")
    return leader


def clip_gripper(action: dict, measured_deg: float, lead_deg: float) -> dict:
    """Keep the gripper target within +-lead_deg of the measured angle. Free motion stays real-time;
    when an object blocks the fingers, the torque is bounded by kp * lead."""
    if "gripper.pos" not in action:
        return action
    a = dict(action)
    a["gripper.pos"] = max(measured_deg - lead_deg, min(measured_deg + lead_deg, a["gripper.pos"]))
    return a


def safe_zero(robot, step_s: float = 0.02, deg_per_step: float = 0.5) -> None:
    obs = robot.get_observation()
    pos = {n: float(obs[f"{n}.pos"]) for n in JOINTS}
    for stage in (STAGE_1, STAGE_2):
        start = {n: pos[n] for n in stage}
        frames = max(1, math.ceil(max(abs(v) for v in start.values()) / deg_per_step))
        for k in range(1, frames + 1):
            for n in stage:
                pos[n] = start[n] * (1.0 - k / frames)
            robot.send_action({f"{n}.pos": pos[n] for n in JOINTS})
            time.sleep(step_s)
    time.sleep(0.3)


def safe_disconnect(robot, go_zero: bool = True) -> None:
    try:
        if go_zero:
            safe_zero(robot)
    finally:
        robot.disconnect()


def read_joints_passive(rcfg: dict) -> dict:
    """Read follower joint angles without enabling any motor (parameter reads only). Public degrees."""
    robot = make_follower(dict(rcfg, velocity_feedforward=0, gravity_compensation=0))
    from lerobot.robots.rebot_b601_follower.rebot_b601_follower import MotorBridgeController
    robot.bus = MotorBridgeController(channel=robot.config.port)
    for name, (sid, rid) in robot._motor_can_ids.items():
        robot.motors[name] = robot.bus.add_robstride_motor(sid, rid, robot.profile.motor_models[name])
    try:
        return {n: robot._motor_to_public_position(n, math.degrees(m.robstride_get_param_f32(0x7019, 500)))
                for n, m in robot.motors.items()}
    finally:
        for m in robot.motors.values():
            m.close()
        robot.bus.close()


class RebotArm(Robot):
    """reBot B601-RS follower (+ reBot Arm 102 leader) with feedforward, gripper clamp and two-stage zeroing."""
    kind = "rebot_b601"

    def __init__(self, rcfg: dict, need_leader: bool):
        super().__init__()
        self.rcfg = rcfg
        self.joints, self.gripper = list(JOINTS), "gripper"
        self.follower = make_follower(rcfg)
        self.leader = make_leader(leader_id=rcfg.get("leader_id", "rebot_arm_102_leader")) if need_leader else None
        self.limits = {n: tuple(v) for n, v in self.follower.config.joint_limits.items()}
        self.lead = float(rcfg.get("gripper_lead_deg", 20.0))

    def send(self, target, measured=None):
        a = self._dict(target)
        if measured is not None:
            a = clip_gripper(a, float(measured[6]), self.lead)
        sent = self.follower.send_action(a)
        out = [float(sent.get(f"{n}.pos", t)) for n, t in zip(self.joints, target)]
        out[6] = float(target[6])          # record the requested gripper target, not the force-clamped one
        return out

    def home_pose(self):
        return [0.0] * 7

    def info(self):
        return dict(self.follower.feedforward, kind=self.kind, gripper_mode=self.rcfg.get("gripper_mode"),
                    gripper_lead_deg=self.lead)

    def shutdown(self, moved: bool = True):
        try:
            if self.leader is not None and self.leader.is_connected:
                self.leader.disconnect()
        finally:
            if self.follower.is_connected:
                safe_disconnect(self.follower, go_zero=moved)
