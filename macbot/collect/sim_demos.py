"""Generate privileged demonstrations using the tabletop arm's Cartesian controller."""
from __future__ import annotations

import time
from contextlib import ExitStack

import numpy as np

from ..cameras import open_cameras
from ..robot.sim import SimArm
from ..robot.sim_cartesian import XYZ_KEYS
from ..simulation import result_path, seeds, write_summary
from ..system import ensure_dir, expand
from .server import Recorder, State


def play(arm, advance):
    """Move one cube at a time, retrying missed grasps with a different jaw orientation."""
    ctrl = arm.cartesian
    world = ctrl.world
    box = world.data.xpos[world.box_body].copy()
    rotation = world.data.xmat[world.box_body].reshape(3, 3).copy()

    def hold(keys=(), ticks=1):
        for _ in range(ticks):
            if arm.ended:
                return
            advance(keys)

    def move(xyz):
        for _ in range(90):
            err = xyz - ctrl.xyz
            fine = np.max(np.abs(err)) < 0.007
            keys = [pair[int(d < 0)] for pair, d in zip(XYZ_KEYS, err, strict=True)
                    if abs(d) > (0.00051 if fine else 0.00151)]
            if arm.ended or not keys:
                break
            hold((*keys, "Shift") if fine else keys)
        hold(ticks=6)

    def inside(body):
        local = (world.data.xpos[body] - box) @ rotation
        return max(abs(local[:2])) < 0.045 and local[2] < 0.06

    cubes = sorted(world.cube_bodies, key=lambda b: -np.linalg.norm(world.data.xpos[b, :2]))
    slots = [box + rotation @ np.array([x, y, 0.0])
             for y in (-0.021, 0.021) for x in (-0.021, 0.021)]
    for i, body in enumerate(cubes):
        for attempt in range(3):
            if arm.ended or inside(body):
                break
            xyz = world.data.xpos[body].copy()
            move([*xyz[:2], 0.065])
            hold("f", 10)
            yaw = np.arctan2(world.data.xmat[body, 3], world.data.xmat[body, 0])
            tool_yaw = np.arctan2(world.data.xmat[ctrl.body, 3], world.data.xmat[ctrl.body, 0])
            error = (yaw - tool_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
            error += (0.0, np.pi / 2, -np.pi / 12)[attempt]
            error = (error + np.pi / 2) % np.pi - np.pi / 2
            ticks = round(np.rad2deg(error) / ctrl.ROLL_STEP)
            hold("m" if ticks > 0 else "n", abs(ticks))
            move([*xyz[:2], xyz[2] + (0.006, 0.006, 0.011)[attempt]])
            hold("g", 15)
            move([*xyz[:2], 0.075])
            if world.data.xpos[body, 2] < 0.04:
                continue
            # A separate slot for each cube reduces pileups and knocks on release.
            move([*slots[i][:2], 0.070])
            hold("f", 20)


def run(cfg, args):
    trial_seeds = seeds(args)
    out = ensure_dir(expand(args.out) if args.out else cfg.raw_dir())
    st = State(cfg, out, False)
    rows = []
    with ExitStack() as cleanup:
        arm = SimArm(cfg["robot"], {"type": "keyboard"})
        cleanup.callback(arm.shutdown)
        arm.check_rate(st.fps, "record.fps")
        st.cams = open_cameras(cfg["cameras"], st.fps, False, arm)
        for cam in st.cams.values():
            cleanup.callback(cam.close)
        arm.connect(leader=True)
        st.ff, st.joints, st.gripper = arm.info(), arm.joints, arm.gripper
        recorder = Recorder(st, threaded=False)
        for seed in trial_seeds:
            arm.reset_scene(seed)
            start = time.perf_counter()
            buf = st.new_buffer()

            def advance(keys, start=start, buf=buf):
                before = recorder.frame(arm.read(), arm.motor_telemetry())
                elapsed = time.perf_counter() - start
                cmd = arm.send(arm.read_leader(keys))
                recorder.append_step(buf, before, cmd, elapsed, realtime=False)

            play(arm, advance)
            count = len(buf["qpos"])
            outcome = {"source": "scripted", "seed": seed, **arm.outcome()}
            buf["simulation"] = outcome
            path = None
            if count and (outcome["success"] or args.keep_failures):
                path = st._save(st.next_idx(), buf, st.instruction, batch=True)
            rows.append({**outcome, "steps": count, "wall_seconds": time.perf_counter() - start,
                         "episode": path.name if path else None})
            print(f"seed {seed}: {'success' if outcome['success'] else 'failed'}, "
                  f"{count} steps, reward {outcome['reward']:g}", flush=True)
    write_summary(result_path(out, "demos"), rows, st.fps, scene=cfg["robot"]["env"], source="scripted")
