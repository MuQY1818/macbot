"""Cartesian control for the local MuJoCo SO-101 tabletop arm."""
from __future__ import annotations

import numpy as np

KEYS = frozenset("ijkluonmfg") | {"Shift"}
XYZ_KEYS = (("i", "k"), ("j", "l"), ("u", "o"))


def tabletop_world(robot, feature="Cartesian control"):
    """Isolate the private LeRobot 0.7 / EnvHub adapter in one place."""
    envs = getattr(robot.follower._env, "_envs", ())
    if len(envs) != 1 or not hasattr(envs[0].unwrapped, "_world"):
        raise SystemExit(f"{feature} requires the local SO-101 tabletop MuJoCo scene")
    world = envs[0].unwrapped._world
    if world is None or robot.follower.descriptor.semantics != "so101-joint-degrees-v1":
        raise SystemExit(f"{feature} requires the SO-101 tabletop model and joint units")
    return world


class Cartesian:
    """Damped IK for XYZ and tool pitch; wrist roll and gripper stay independent."""

    # In the gripper body's frame, between the pads near their tips.
    TIP = np.array([0.006, 0.0, -0.093])
    STEP = 0.003
    FINE_STEP = 0.001
    ROLL_STEP = 3.0

    def __init__(self, world):
        import mujoco

        self.mj = mujoco
        self.world = world
        self.model = world.model
        self.data = mujoco.MjData(self.model)
        self.body = self.model.body("gripper").id
        self.dof = np.array([self.model.joint(n).dofadr[0] for n in
                             ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")])
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros_like(self.jacp)
        self.reset()

    @staticmethod
    def tip(data, body):
        axes = data.xmat[body].reshape(3, 3)
        return data.xpos[body] + axes @ Cartesian.TIP

    def reset(self):
        self.q = self.world.data.qpos[self.world.qadr].copy()
        self.xyz = self.tip(self.world.data, self.body)
        self.grip = float(self.world.joint_state()[-1])
    def action(self, xyz, roll, grip, *, pitch=None):
        """Solve from the preceding joint target, tilting outward near full reach."""
        if pitch is None:
            reach = np.hypot(*xyz[:2]) + 0.35 * max(0.0, xyz[2] - 0.03)
            pitch = -np.clip((reach - 0.253) * 8.0, 0, 0.9)
        self.data.qpos[:] = self.world.data.qpos
        q = self.q.copy()
        q[4] = np.clip(np.deg2rad(roll), self.world.lo[4], self.world.hi[4])
        for _ in range(16):
            self.data.qpos[self.world.qadr] = q
            self.mj.mj_kinematics(self.model, self.data)
            self.mj.mj_comPos(self.model, self.data)
            tip = self.tip(self.data, self.body)
            axis = self.data.xmat[self.body].reshape(3, 3)[:, 2]
            radial = self.data.xpos[self.body].copy()
            radial[2] = 0
            radial /= max(1e-8, np.linalg.norm(radial))
            tangent = np.array([-radial[1], radial[0], 0.0])
            current_pitch = np.arctan2(axis @ radial, axis[2])
            err = np.r_[xyz - tip, 0.08 * (pitch - current_pitch)]
            if np.linalg.norm(err) < 0.00015:
                break
            self.mj.mj_jac(self.model, self.data, self.jacp, self.jacr, tip, self.body)
            jac = np.vstack((self.jacp[:, self.dof], 0.08 * tangent @ self.jacr[:, self.dof]))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 2.5e-6 * np.eye(4), err)
            q[:4] = np.clip(q[:4] + np.clip(delta, -0.12, 0.12), self.world.lo[:4], self.world.hi[:4])
        self.data.qpos[self.world.qadr] = q
        self.mj.mj_kinematics(self.model, self.data)
        self.q = q
        return [*np.rad2deg(q[:5]), grip]

    def keys(self, keys):
        step = self.FINE_STEP if "Shift" in keys else self.STEP
        move = np.array([int(a in keys) - int(b in keys) for a, b in XYZ_KEYS])
        roll = int("m" in keys) - int("n" in keys)
        if "f" in keys:
            self.grip = 100.0
        elif "g" in keys:
            self.grip = 0.0
        if np.any(move) or roll:
            target = self.xyz + step * move
            target[2] = np.clip(target[2], 0.009, 0.30)
            action = self.action(target, np.rad2deg(self.q[4]) + self.ROLL_STEP * roll, self.grip)
            # Cap unreachable motion at the actual kinematic solution.
            solved = self.tip(self.data, self.body)
            self.xyz = target if np.linalg.norm(solved - target) < 0.004 else solved
            return action
        return [*np.rad2deg(self.q[:5]), self.grip]
