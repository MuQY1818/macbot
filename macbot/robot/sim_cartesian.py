"""Cartesian control for the local MuJoCo SO-101 tabletop arm."""
from __future__ import annotations

import numpy as np

XYZ_KEYS = (("d", "a"), ("w", "s"), ("e", "q"))
KEYS = frozenset("wasdeqzx") | {"Shift", "Space"}


def tabletop_env(robot, feature="Cartesian control"):
    """Isolate the private LeRobot 0.7 / EnvHub adapter in one place."""
    envs = getattr(robot.follower._env, "_envs", ())
    if len(envs) != 1 or not hasattr(envs[0].unwrapped, "_world"):
        raise SystemExit(f"{feature} requires the local SO-101 tabletop MuJoCo scene")
    world = envs[0].unwrapped._world
    if world is None or robot.follower.descriptor.semantics != "so101-joint-degrees-v1":
        raise SystemExit(f"{feature} requires the SO-101 tabletop model and joint units")
    return envs[0].unwrapped


def tabletop_world(robot, feature="Cartesian control"):
    return tabletop_env(robot, feature)._world


class Cartesian:
    """Damped IK for XYZ and tool pitch; wrist roll and gripper stay independent."""

    # In the gripper body's frame, between the pads near their tips.
    TIP = np.array([0.006, 0.0, -0.093])

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

    def pitch(self, data):
        axis = data.xmat[self.body].reshape(3, 3)[:, 2]
        radial = data.xpos[self.body].copy()
        radial[2] = 0
        radial /= max(1e-8, np.linalg.norm(radial))
        return np.arctan2(axis @ radial, axis[2]), radial
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
            current_pitch, radial = self.pitch(self.data)
            tangent = np.array([-radial[1], radial[0], 0.0])
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

class KeyboardCartesian(Cartesian):
    """Hold a bounded Cartesian setpoint while the keys are up."""

    SPEED = 0.060  # metres per second
    ROLL_SPEED = 45.0  # degrees per second
    FINE = 0.25
    RAMP_S = 0.10
    MIN_HEIGHT = 0.009
    MAX_HEIGHT = 0.30

    def __init__(self, world, camera, fps):
        self.camera = world.model.camera(camera).id
        if world.model.cam_bodyid[self.camera] != 0:
            raise SystemExit("keyboard teleoperation needs a fixed camera first (for example, front or top)")
        self.dt = 1.0 / fps
        table = world.model.geom("table").id
        self.table_z = float(world.data.geom_xpos[table, 2] + world.model.geom_size[table, 2])
        super().__init__(world)

    def reset(self):
        super().reset()
        cam = self.world.data.cam_xmat[self.camera].reshape(3, 3)
        right = cam[:2, 0].copy()
        right /= np.linalg.norm(right)
        forward = np.array([-right[1], right[0]])
        if forward @ cam[:2, 1] < 0:
            forward *= -1
        self.axes = np.eye(3)
        self.axes[:2, :2] = np.column_stack((right, forward))
        self.tool_pitch = self.pitch(self.world.data)[0]
        self.velocity = np.zeros(4)
        self.limit = None

    def keys(self, keys):
        # Space is a single pulse from the collect server, even for a quick tap.
        if "Space" in keys:
            self.grip = 0.0 if self.grip >= 15 else 100.0
        move = np.array([int(a in keys) - int(b in keys) for a, b in XYZ_KEYS], dtype=float)
        move /= max(1.0, np.linalg.norm(move))
        wanted = np.r_[move, int("x" in keys) - int("z" in keys)]
        # Ramp each active axis from rest. Release and reversal cancel the old velocity immediately.
        self.velocity[np.sign(wanted) != np.sign(self.velocity)] = 0
        delta = self.dt / self.RAMP_S
        self.velocity += np.clip(wanted - self.velocity, -delta, delta)
        speed = self.FINE if "Shift" in keys else 1.0
        step = self.axes @ self.velocity[:3] * (self.SPEED * speed * self.dt)
        roll = self.velocity[3] * (self.ROLL_SPEED * speed * self.dt)
        level = self.tool_pitch - np.clip(self.tool_pitch, -self.dt, self.dt)
        self.limit = None
        if np.any(step) or roll or level != self.tool_pitch:
            requested_z = self.xyz[2] + step[2]
            z = np.clip(requested_z, self.table_z + self.MIN_HEIGHT, self.table_z + self.MAX_HEIGHT)
            prev = self.q.copy()
            requested_roll = prev[4] + np.deg2rad(roll)
            roll_target = np.clip(requested_roll, self.world.lo[4], self.world.hi[4])
            if z != requested_z:
                self.limit = "height"
            elif roll_target != requested_roll:
                self.limit = "roll"
            step[2] = z - self.xyz[2]
            # Reject an IK branch jump. Use the closest valid fraction of this step at the boundary.
            fractions = ((1.0, 1.0), (0.5, 0.5), (0.25, 0.25))
            # A custom reset pose can be too far out to level in place. Let the operator move back in.
            if level != self.tool_pitch and np.any(step):
                fractions += ((1.0, 0.0),)
            for fraction, level_fraction in fractions:
                target = self.xyz + fraction * step
                pitch = self.tool_pitch + level_fraction * (level - self.tool_pitch)
                wrist = prev[4] + fraction * (roll_target - prev[4])
                self.q = prev
                action = self.action(target, np.rad2deg(wrist), self.grip, pitch=pitch)
                if (np.linalg.norm(self.tip(self.data, self.body) - target) <= 0.0002
                        and abs(self.pitch(self.data)[0] - pitch) <= 0.004
                        and np.max(np.abs(self.q - prev)) <= 0.10):
                    self.xyz, self.tool_pitch = target, pitch
                    if min(fraction, level_fraction) < 1:
                        self.limit = "workspace"
                    return action
            self.q = prev
            self.limit = "workspace"
        return [*np.rad2deg(self.q[:5]), self.grip]

    def status(self, shape, crop=None):
        data = self.world.data
        tip = self.tip(data, self.body)
        floor = np.array([*tip[:2], self.table_z])
        x, y, depth = (floor - data.cam_xpos[self.camera]) @ data.cam_xmat[self.camera].reshape(3, 3)
        scale = -depth * np.tan(np.deg2rad(self.model.cam_fovy[self.camera]) / 2)
        h, w = shape[:2]
        if scale <= 1e-8:
            shadow = None
        else:
            u, v = 0.5 + x * h / w / (2 * scale), 0.5 - y / (2 * scale)
            if crop:
                x0, y0, x1, y1 = crop
                u, v = (u * w - x0) / (x1 - x0), (v * h - y0) / (y1 - y0)
            shadow = [float(u), float(v)] if 0 <= u <= 1 and 0 <= v <= 1 else None
        return {"height_cm": round(float(tip[2] - self.table_z) * 100, 1),
                "closed": self.grip < 15, "limit": self.limit, "shadow": shadow}
