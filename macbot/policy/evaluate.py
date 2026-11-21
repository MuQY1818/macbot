"""Seeded headless evaluation on the simulator's step clock."""
from __future__ import annotations

import math
import time
from contextlib import ExitStack
from types import SimpleNamespace

import numpy as np

from ..cameras import open_cameras
from ..robot import open_robot
from ..simulation import result_path, seeds, write_summary
from ..system import ensure_dir, expand
from .executors import _interp, switch_record
from .gripper import BinaryGripper, find_gripper_config
from .models import prep_image
from .runner import State, check_policy_framing, prepare_policy, save_rollout


class StepClock:
    """The simulator advances once per target, independent of wall clock inference time."""

    def __init__(self, policy, args, fps, joints, limit_chunk):
        self.policy, self.args, self.joints = policy, args, joints
        self.fps, self.k = fps, min(args.k, policy.H) if limit_chunk else args.k
        if args.mode == "ensemble":
            if (not math.isfinite(args.te_hz) or args.te_hz <= 0
                    or not math.isfinite(args.te_m)):
                raise SystemExit("--te-hz must be positive and --te-m must be finite")
            self.interval = max(1, math.ceil(fps / args.te_hz))
        elif args.mode in {"async", "rtc"}:
            self.interval = max(1, args.query_every, args.delay_steps)
        elif args.mode == "step":
            self.interval = 1
        else:
            if self.k < 1:
                raise SystemExit("--k must be positive")
            self.interval = self.k
        self.chunks = []
        self.raw = None
        self.id = -1

    def target(self, step, obs, q, log):
        args = self.args
        if step % self.interval == 0:
            tick = time.perf_counter()
            previous = self.chunks[-1] if self.chunks else None
            age = step - previous[0] if previous else 0
            delay = args.delay_steps if args.mode in {"async", "rtc"} and previous else 0
            if args.mode == "rtc":
                left = self.raw[age:] if self.raw is not None else None
                left = left if left is not None and len(left) > delay else None
                action, self.raw = self.policy(obs, q, leftover=left, delay=delay if left is not None else 0,
                                               horizon=args.rtc_horizon, return_raw=True)
            else:
                action = self.policy(obs, q)
            action = np.asarray(action)
            if (action.ndim != 2 or not len(action) or action.shape[-1] != self.joints
                    or not np.isfinite(action).all()):
                raise ValueError(f"Expected a finite action chunk with {self.joints} joint values; got {action.shape}")
            self.id += 1
            if args.mode not in {"ensemble", "async", "rtc"}:
                self.chunks.clear()
            self.chunks.append((step, action, self.id, step + delay))
            log(step / self.fps, (time.perf_counter() - tick) * 1000,
                **switch_record(action, previous[1] if previous else None, age, delay, q))
        live = [(step - start, a, i) for start, a, i, ready in self.chunks if ready <= step]
        if args.mode == "ensemble":
            while len(self.chunks) > 1 and step - self.chunks[0][0] >= len(self.chunks[0][1]):
                self.chunks.pop(0)
            keep = [c for c in live if c[0] < len(c[1])] or live[-1:]
            weights = args.te_m * np.asarray([c[0] for c in keep])
            weights = np.exp(weights - weights.max())
            action = np.average([_interp(a, age) for age, a, _ in keep], axis=0, weights=weights)
            return action, keep[-1][2]
        if len(live) > 1:  # Keep only the newest ready chunk and a pending announcement.
            self.chunks = [c for c in self.chunks if c[2] >= live[-1][2]]
        age, action, i = live[-1]
        return _interp(action, age), i

    def ahead(self, step, horizon):
        ready = [(start, a) for start, a, _, available in self.chunks if available <= step]
        return _interp(ready[-1][1], step + horizon - ready[-1][0]) if ready else None


def seed_policy(seed, backend, device):
    if backend == "mlx":
        import mlx.core as mx

        mx.random.seed(seed)
    else:
        import torch

        torch.random.default_generator.manual_seed(seed)
        if device == "mps":
            torch.mps.manual_seed(seed)
        elif device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)


def episode(st, robot, policy, sizes, quality, limit_chunk, seed, gripper):
    args = st.args
    fps = st.cfg["record"]["fps"]
    robot.reset_scene(seed)
    if reset := getattr(policy, "reset", None):
        reset()
    seed_policy(seed, st.backend_selection.backend, args.device)
    clock = StepClock(policy, args, fps, len(st.joints), limit_chunk)
    rec = st.new_record(0.0)
    st.rec = rec
    started = time.perf_counter()
    gi = robot.gripper_index
    if gripper is not None:
        gripper.sync(robot.read()[gi])
    for step in range(robot.episode_limit()):
        q = robot.read()
        encoded = {c: prep_image(st.cams[c].latest()[0], sizes[c], quality) for c in st.cam_names}
        cmd, chunk = clock.target(step, {c: v[0] for c, v in encoded.items()}, q, st.log_chunk)
        cmd = cmd.tolist()
        grip, ahead = cmd[gi] if gi is not None else float("nan"), float("nan")
        if gripper is not None:
            preview = clock.ahead(step, args.grip_lead * fps) if args.grip_lead else None
            ahead = float(preview[gi]) if preview is not None else float("nan")
            cmd[gi] = gripper(grip, step / fps, ahead if preview is not None else None)
        rec["motor"].append(robot.motor_telemetry())
        rec["q"].append(q)
        rec["cmd"].append(robot.send(cmd, q))
        rec["gp"].append(grip)
        rec["gp_ahead"].append(ahead)
        rec["t"].append(step / fps)
        rec["chunk"].append(chunk)
        for c, (_, jpg) in encoded.items():
            rec[f"{c}_jpg"].append(jpg)
        if robot.ended:
            break
    st.rec = None
    return rec, {"source": "eval", "seed": seed, "steps": len(rec["t"]), **robot.outcome(),
                 "wall_seconds": time.perf_counter() - started}


def run(cfg, args):
    trial_seeds = seeds(args)
    check_policy_framing(cfg, args)
    if args.delay_steps < 0 or args.query_every < 0:
        raise SystemExit("--delay-steps and --query-every must be nonnegative")
    if not (math.isfinite(args.grip_lead) and 0 <= args.grip_lead <= 2):
        raise SystemExit("--grip-lead must be between 0 and 2 seconds")
    fps = int(cfg["record"]["fps"])
    out = ensure_dir(expand(args.out) if args.out else cfg.rollout_dir())
    st = State(cfg, out, args)
    st.takeover = False
    rows = []
    with ExitStack() as cleanup:
        robot = open_robot(cfg["robot"])
        cleanup.callback(robot.shutdown, moved=False)
        robot.check_rate(fps, "record.fps")
        robot.check_rate(cfg["run"]["hz"], "run.hz")
        st.joints, st.gripper, st.robot_info = robot.joints, robot.gripper, robot.info()
        pol, sizes, quality, limit_chunk = prepare_policy(st, robot, warmup=False)
        if args.delay_steps and args.mode not in {"async", "rtc"}:
            raise SystemExit("--delay-steps applies to --mode async or rtc only")
        gcfg = find_gripper_config(args.policy, args.gripper)
        if gcfg and st.gripper is None:
            raise SystemExit("gripper.json found but the robot has no joint named gripper; use --gripper raw")
        gripper = BinaryGripper(gcfg, args.grip_debounce) if gcfg else None
        st.binary_gripper = {"open_deg": gripper.open_deg, "close_deg": gripper.close_deg} if gripper else False
        st.cams = open_cameras(cfg["cameras"], fps, False, robot)
        for camera in st.cams.values():
            cleanup.callback(camera.close)
        robot.connect()
        st.executor = f"{args.mode} (simulated step clock)"
        for seed in trial_seeds:
            rec, row = episode(st, robot, pol, sizes, quality, limit_chunk, seed, gripper)
            result = "success" if row["success"] else "failure"
            name, _ = save_rollout(st, rec, result, simulation={**row, "delay_steps": args.delay_steps})
            rows.append({**row, "rollout": name})
            print(f"seed {seed}: {result}, {row['steps']} steps, "
                  f"{row['steps'] / row['wall_seconds']:.1f} steps/s, reward {row['reward']:g}", flush=True)
            if args.videos:
                from ..viz.video import run as video

                video(SimpleNamespace(rollout=out / name, out=None, height=240))
    write_summary(result_path(out, "eval"), rows, fps, scene=cfg["robot"]["env"],
                  policy=str(args.policy), policy_hub=args.policy_hub, backend=st.backend_selection.backend,
                  mode=args.mode, delay_steps=args.delay_steps, k=args.k, query_every=args.query_every,
                  device=args.device)
