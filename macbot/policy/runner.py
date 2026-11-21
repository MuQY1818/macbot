"""Run a trained policy on the robot, driven from a local web page.

Shares cameras, follower settings and the safe shutdown with ``macbot collect``. Observations match
training: images resized to the recorded size and passed through JPEG, state = measured joints (deg).

Safety: starts paused; arm joints move at most ``max_vel x speed`` deg/s; joint limits; gripper force
clamp; a camera that stalls for 0.5 s pauses the run; pause holds the current command; "home" moves
to the start pose at ``home_vel`` (all arm joints arrive together, with a short ramp); quitting returns to zero in two stages before disabling.

Each trial is saved as ``<data_root>/rollouts/<task>/rollout_XXX.npz``: qpos, cmd, t, chunk_id per
frame, chunk_t and infer_ms per chunk, executor extras, JPEG frames per camera, and meta (result).
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import re
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np

from .. import web
from ..cameras import open_cameras
from ..cameras.framing import check_framing, framing, training_framing
from ..config import Config
from ..data.motor_health import diagnostic_path, motor_arrays, record_motor_meta
from ..i18n import tr
from ..safety import MotorWatch, timed_events
from ..system import ensure_dir, expand, give_back, start_log
from ..viz.rollouts import Store, serve_rollouts
from .backends import select_backend
from .executors import AsyncChunk, Ensemble, Step, SyncChunk
from .gripper import BinaryGripper, find_gripper_config, takeover_gripper_alignment
from .models import check_joint_dimensions, load_generic, load_policy, policy_backend, prep_image, use_generic

CAM_STALL_S = 0.5
HOME_RAMP_S = 0.6                                         # homing speeds up over this time
MOTION_COMMANDS = ("start", "home", "quit")
TAKEOVER_MODES = {"aligning": 2, "following": 1}
LEADER_STALE_S = 0.25                                     # older leader samples hold the follower
CHUNK_LOG_KEYS = ("ex_chunk", "ex_prev", "ex_q_obs")      # per-switch chunk arrays, kept in float32
CATCHUP_S = 1.0                                           # takeover soft start: home_vel -> max_vel over this time
CATCHUP_DONE_DEG = 2.0                                    # ...or earlier, once every arm joint is this close
LEADER_GRACE_S = 0.5                                      # a shorter leader dropout holds, then keeps following
TRIAL_MODES = ("running", *TAKEOVER_MODES)


class LeaderReader:
    """Keep serial reads off the control loop. A missing or old sample cannot move the follower."""

    def __init__(self, robot, dt, cleanup):
        self.sample = None
        self.stop = threading.Event()
        worker = threading.Thread(target=self._loop, args=(robot, dt), daemon=True)
        worker.start()
        cleanup.callback(self._close, worker)

    def _loop(self, robot, dt):
        while not self.stop.is_set():
            tick = time.monotonic()
            self.sample = (tick, leader_pose(robot))
            self.stop.wait(max(0.0, dt - (time.monotonic() - tick)))

    def get(self):
        sample = self.sample
        return sample[1] if sample and time.monotonic() - sample[0] <= LEADER_STALE_S else None

    def _close(self, worker):
        self.stop.set()
        worker.join(timeout=0.5)


def leader_pose(robot):
    try:
        q = robot.read_leader_fresh()
        if len(q) == len(robot.joints) and all(map(math.isfinite, q)):
            return q
    except Exception:
        pass  # Hold and require alignment again if a device read fails.
    return None


class State:
    def __init__(self, cfg: Config, out: Path, args):
        self.cfg, self.out, self.args = cfg, out, args
        self.sim = not args.fake and cfg["robot"].get("type") == "sim"
        self.takeover = getattr(args, "takeover", False) or cfg["run"]["takeover"]
        self.align_deg = float(cfg["run"]["takeover_align_deg"]) if self.takeover else None
        if self.takeover and not (math.isfinite(self.align_deg) and self.align_deg > 0):
            raise SystemExit("run.takeover_align_deg must be positive and finite")
        self.alignment = {}
        self.gripper_alignment = None
        self.gripper_gate = {}
        self.follow_since = None
        self.cam_names = list(cfg["cameras"])
        self.lock = threading.Lock()
        self.mode = "paused"            # paused | running | homing | aligning | following
        self.want = None                # request from the page
        self.msg = tr("Paused. Set up the scene, make sure it is safe, then press Start.",
                      "已暂停。场景摆好、确认安全后按“开始”。")
        self.joints = []
        self.q, self.cmd = [], []
        self.cams, self.chunks, self.infer_ms = {}, 0, []
        self.quit = False
        self.rec = None
        self.next = self.next_idx()
        self.saved_revision = 0
        self.replay_readers = 0
        self.replay_transition = False
        self.executor = ""
        self.backend_selection = None
        self.gripper, self.robot_info, self.limits = None, {}, {}
        self.binary_gripper = False
        self.watch = MotorWatch(cfg["safety"])

    def next_idx(self):
        ids = [int(m.group(1)) for p in self.out.glob("rollout_*.npz") if (m := re.search(r"_(\d+)\.npz$", p.name))]
        return max(ids, default=-1) + 1

    def replay_ready(self):
        """Call with the state lock held."""
        return self.mode == "paused" and not self.quit and not self.replay_transition and self.want not in MOTION_COMMANDS

    @contextmanager
    def replay_lease(self):
        with self.lock:
            allowed = self.replay_ready()
            if allowed:
                self.replay_readers += 1
        try:
            yield allowed
        finally:
            if allowed:
                with self.lock:
                    self.replay_readers -= 1

    def take_command(self):
        with self.lock:
            if self.want in MOTION_COMMANDS and self.replay_readers:
                # Keep ticking while the last paused read finishes. Pending motion blocks new reads.
                return None
            want, self.want = self.want, None
            self.replay_transition = want in MOTION_COMMANDS
            return want

    def request(self, action):
        with self.lock:
            if action == "mark_hand":
                if self.rec is not None and self.want != "quit":
                    self.rec["hand_marks"].append(max(0.0, time.time() - self.rec["t0"]))
                return
            if self.want == "quit":
                return
            if action == "takeover":
                if not self.takeover or self.mode not in TRIAL_MODES or self.want is not None:
                    return
            elif action not in (*MOTION_COMMANDS, "pause", "save_ok", "save_fail", "discard"):
                return
            self.want = action

    def set_mode(self, mode):
        """Call with the state lock held. Account for actual time spent following."""
        if self.mode == mode:
            return
        if self.follow_since is not None:
            self.rec["takeover_seconds"] += time.monotonic() - self.follow_since
            self.follow_since = None
        if mode == "following":
            self.follow_since = time.monotonic()
        self.mode = mode
        if mode not in TAKEOVER_MODES:
            self.alignment = {}
            self.gripper_alignment = None

    def log_chunk(self, t_chunk, ms, **extra):
        """Called by executors after every inference."""
        with self.lock:
            if self.watch.trip or (self.takeover and self.mode != "running"):
                return
            self.infer_ms.append(ms)
            if extra.get("switched", 1):
                self.chunks += 1
            r = self.rec
            if r is not None:
                if t_chunk is not None:
                    r["chunk_t"].append(t_chunk - r["t0"])
                    r["infer_ms"].append(ms)
                for k, v in extra.items():
                    r.setdefault(f"ex_{k}", []).append(v)

    def status(self):
        with self.lock:
            now = time.time()
            ages = {c: (round(now - self.cams[c].latest()[1], 2) if c in self.cams else None) for c in self.cam_names}
            r = self.rec
            return {"mode": self.mode, "msg": self.msg, "q": self.q, "cmd": self.cmd, "names": self.joints, "limits": self.limits,
                    "cam_names": self.cam_names, "cam_age": ages,
                    "cam_size": {c: self.cfg["cameras"][c]["save_size"] for c in self.cam_names}, "stall_s": CAM_STALL_S, "chunks": self.chunks,
                    "infer_ms": round(float(np.median(self.infer_ms[-20:])), 1) if self.infer_ms else None,
                    "executor": self.executor, "speed": self.args.speed, "fake": self.args.fake, "sim": self.sim,
                    "policy": "/".join(Path(self.args.policy).parts[-2:]), "task": self.cfg.task,
                    "rec_s": round(len(r["t"]) / self.cfg["record"]["fps"], 1) if r else 0,
                    "recording": r is not None, "hand_marks": len(r["hand_marks"]) if r else 0,
                    "takeover_enabled": self.takeover, "align_deg": self.align_deg, "alignment": self.alignment,
                    "gripper_alignment": self.gripper_alignment, "gripper": self.gripper,
                    "saved_revision": self.saved_revision, "replay_ready": self.replay_ready(), "next": self.next,
                    "motor_health": self.watch.status()}

    def new_record(self, now):
        return {"q": [], "cmd": [], "t": [], "chunk": [], "chunk_t": [], "infer_ms": [], "gp": [], "gp_ahead": [],
                "t0": now, "takeover": [], "leader": [], "takeover_count": 0, "takeover_seconds": 0.0,
                "hand_marks": [], "motor": [], "watchdog_events": [], **{f"{c}_jpg": [] for c in self.cam_names}}


def stack_chunks(rows) -> np.ndarray:
    """Stack per-switch arrays into float32, padding chunks of different lengths with NaN."""
    rows = [np.asarray(r, np.float32) for r in rows]
    if not rows:
        return np.zeros((0,), np.float32)
    shape = tuple(max(r.shape[i] for r in rows) for i in range(rows[0].ndim))
    out = np.full((len(rows), *shape), np.nan, np.float32)
    for i, r in enumerate(rows):
        out[(i, *(slice(0, n) for n in r.shape))] = r
    return out


def save_rollout(st: State, rec: dict, result: str):
    p = (diagnostic_path(st.out, "rollout") if result == "watchdog"
         else st.out / f"rollout_{st.next_idx():03d}.npz")
    arrs = motor_arrays(rec["motor"], len(st.joints))
    for c in st.cam_names:
        js = rec[f"{c}_jpg"]
        arrs[f"{c}_offsets"] = np.cumsum([0] + [len(j) for j in js]).astype(np.int64)
        arrs[f"{c}_jpeg"] = np.concatenate(js) if js else np.zeros(0, np.uint8)
    a = st.args
    fps = st.cfg["record"]["fps"]
    meta = {"task": st.cfg.task, "instruction": st.cfg["task"]["instruction"], "policy": str(a.policy),
            "executor": st.executor, "mode": a.mode, "k": a.k, "te_m": a.te_m, "rtc_beta": a.rtc_beta,
            "rtc_horizon": getattr(a, "rtc_horizon", None), "rtc_schedule": getattr(a, "rtc_schedule", "linear"),
            "query_every": getattr(a, "query_every", 0),
            "grip_debounce": a.grip_debounce, "grip_lead": getattr(a, "grip_lead", 0.0), "speed": a.speed, "max_vel": st.cfg["run"]["max_vel"], "result": result,
            "fps": fps, "frames": len(rec["t"]), "joint_names": st.joints, "gripper": st.gripper, "robot": st.robot_info, "cameras": st.cam_names,
            "camera_framing": framing(st.cfg["cameras"], fps),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "fake": a.fake, "writer": "macbot"}
    meta.update(hand_marks=rec["hand_marks"], takeover_count=rec["takeover_count"],
                takeover_seconds=round(rec["takeover_seconds"], 3), binary_gripper=st.binary_gripper,
                safety=st.cfg["safety"])
    if policy_hub := getattr(a, "policy_hub", None):
        meta["policy_hub"] = policy_hub
    if (selection := st.backend_selection) is not None:
        meta.update(backend=selection.backend, backend_requested=selection.requested, backend_reason=selection.reason)
    record_motor_meta(rec, meta)
    if "grasp_miss" in st.cfg["robot"]:
        meta["grasp_miss_profile"] = st.cfg["robot"]["grasp_miss"]
    if st.takeover:
        meta["takeover_alignment"] = {"arm_deg": st.align_deg, **st.gripper_gate}
        arrs["takeover"] = np.asarray(rec["takeover"], np.uint8)
        arrs["leader"] = np.asarray(rec["leader"], np.float32)
    ex = {k: stack_chunks(v) if k in CHUNK_LOG_KEYS else np.asarray(v, np.float64) for k, v in rec.items() if k.startswith("ex_")}
    tmp = p.with_suffix(".tmp.npz")
    shape = (len(rec["t"]), len(st.joints))
    np.savez(tmp, qpos=np.asarray(rec["q"], np.float32).reshape(shape), cmd=np.asarray(rec["cmd"], np.float32).reshape(shape),
             t=np.asarray(rec["t"], np.float64), chunk_id=np.asarray(rec["chunk"], np.int32),
             chunk_t=np.asarray(rec["chunk_t"], np.float64), infer_ms=np.asarray(rec["infer_ms"], np.float32),
             gp=np.asarray(rec["gp"], np.float32), gp_ahead=np.asarray(rec.get("gp_ahead", []), np.float32), meta=json.dumps(meta, ensure_ascii=False), **ex, **arrs)
    tmp.rename(p)
    give_back(p)
    return p.name, meta


def make_handler(st: State, trace_worker=None):
    from ..data.grasp import settings

    preview = {k: tuple(st.cfg["cameras"][k]["save_size"]) for k in st.cam_names}
    store = Store(st.out, trace_worker=trace_worker, grasp_cfg=settings(st.cfg))

    class H(web.Handler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self.send(200, web.page("run", task=st.cfg.task), "text/html; charset=utf-8")
            if self.static(path) or serve_rollouts(self, store, path, lease=st.replay_lease):
                return
            if path == "/api/status":
                return self.json(st.status())
            m = re.fullmatch(r"/stream/(\w+)\.mjpg", path)
            if m and m.group(1) in st.cams:
                return self.mjpeg(st.cams[m.group(1)], preview[m.group(1)], lambda: st.quit)
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if serve_rollouts(self, store, self.path, method="POST", lease=st.replay_lease):
                return
            if self.path != "/api/cmd":
                return self.send(404, b"", "text/plain")
            try:
                st.request(self.body_json().get("action"))
            except Exception as e:
                with st.lock:
                    st.msg = tr(f"Command failed: {e}", f"操作失败：{e}")
            self.json(st.status())

    return H


def homing_steps(cmd, home, gi, step_max):
    """Per-joint step limits for a straight line in joint space: the farthest arm joint moves ``step_max`` per
    cycle and every other arm joint proportionally less, so all arrive together."""
    dist = [abs(h - c) for h, c in zip(home, cmd, strict=True)]
    far = max((d for j, d in enumerate(dist) if j != gi), default=0.0)
    return [step_max * (d / far if far > 0 else 1.0) for d in dist]


def run(cfg: Config, args):
    with web.bind_console(cfg, "run", web.Handler, args.port) as srv, ExitStack() as cleanup:
        _run(cfg, args, srv, cleanup)


def start_trace_worker():
    if hasattr(os, "nice"):
        os.nice(5)


def _run(cfg, args, srv, cleanup):
    args.policy = expand(args.policy)
    current = framing(cfg["cameras"], cfg["record"]["fps"])
    try:
        expected = training_framing(args.policy, cfg.root)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Cannot read training camera framing: {exc}") from None
    check_framing(expected, current, override=args.allow_framing_mismatch)
    start_log(cfg.log_dir, "run")
    rc = cfg["run"]
    args.speed = args.speed if args.speed is not None else rc["speed"]
    if not 0.1 <= args.speed <= 1.0:
        sys.exit("--speed must be between 0.1 and 1.0")
    if args.autostart and not (args.fake or cfg["robot"].get("type") == "sim"):
        sys.exit("--autostart only works with --fake or robot.type: sim")
    gcfg = find_gripper_config(args.policy, args.gripper)
    gripper = BinaryGripper(gcfg, args.grip_debounce) if gcfg else None
    lead = float(getattr(args, "grip_lead", 0.0) or 0.0)
    if not (math.isfinite(lead) and 0 <= lead <= 2):
        sys.exit("--grip-lead must be between 0 and 2 seconds")
    if int(getattr(args, "query_every", 0) or 0) < 0:
        sys.exit("--query-every must be 0 or positive")
    if lead and gripper is None:
        print("--grip-lead applies to binary grippers only; ignored", flush=True)

    fps = int(cfg["record"]["fps"])
    out = ensure_dir(cfg.rollout_dir(cfg.task + ("_fake" if args.fake else "")))
    st = State(cfg, out, args)
    trace_worker = cleanup.enter_context(ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=start_trace_worker,
    ))
    # Spawn before connecting the robot, so opening the first replay does not start a process.
    trace_worker.submit(os.getpid).result()
    srv.RequestHandlerClass = make_handler(st, trace_worker)
    _control(st, srv, fps, gripper, cleanup)


def _control(st, srv, fps, gripper, cleanup):
    cfg, args, rc = st.cfg, st.args, st.cfg["run"]
    from ..robot import open_robot

    st.binary_gripper = {"open_deg": gripper.open_deg, "close_deg": gripper.close_deg} if gripper else False
    if st.takeover:
        if st.binary_gripper:
            close = float(cfg["export"]["gripper_close_above"])
            opened = float(cfg["export"]["gripper_open_below"])
            if not (math.isfinite(opened) and math.isfinite(close) and opened < close):
                sys.exit("takeover needs finite export gripper thresholds with open_below less than close_above")
            st.gripper_gate = {"close_above": close, "open_below": opened}
        else:
            tolerance = float(rc["takeover_gripper_align_deg"])
            if not (math.isfinite(tolerance) and tolerance > 0):
                sys.exit("run.takeover_gripper_align_deg must be positive and finite")
            st.gripper_gate = {"tolerance": tolerance}
    robot = (open_robot(cfg["robot"], fake=args.fake, leader=True, fresh_leader=True) if st.takeover
             else open_robot(cfg["robot"], fake=args.fake))
    started = False
    cleanup.callback(lambda: robot.shutdown(moved=started))
    names, n_j = robot.joints, len(robot.joints)
    with st.lock:
        st.joints, st.gripper = names, robot.gripper
        st.watch.set_joints(names, robot.gripper)
        st.q = st.cmd = [0.0] * n_j
    gi = robot.gripper_index
    metadata = json.loads((args.policy / "config.json").read_text())
    selection = select_backend(policy_backend(cfg, args), args.policy, metadata, args)
    st.backend_selection = selection
    backend = selection.backend
    selection.announce()
    generic = use_generic(metadata.get("type", "act"), getattr(args, "policy_impl", "auto"), backend)
    if generic:
        if not (args.fake or st.sim) and cfg.get("policy", {}).get("action_space") != "joint_position":
            sys.exit("Generic policies require policy.action_space: joint_position for hardware. "
                     "First verify absolute joint order and units with --fake.")
    else:
        check_joint_dimensions(args.policy, names)
    if st.sim:
        robot.check_rate(rc["hz"], "run.hz")
    if gripper is not None and gi is None:
        sys.exit("gripper.json found but the robot has no joint named gripper; use --gripper raw")
    print(f"loading policy {args.policy} on {args.device}", flush=True)
    if generic:
        pol = load_generic(args.policy, cfg, args, n_j)
        mode = "step" if pol.stepwise else "chunk"
        if args.mode not in {"auto", mode}:
            sys.exit(f"Generic {pol.kind} requires --mode {mode} (or auto)")
        args.mode = mode
        if mode == "chunk" and args.k < 1:
            sys.exit("--k must be positive")
    elif backend == "mlx":
        pol = load_policy(args.policy, args.device, cfg["task"]["instruction"], args.steps, backend="mlx",
                          mlx_bits=args.mlx_bits, mlx_quantize=args.mlx_quantize, mlx_time_grid=args.mlx_time_grid)
    else:
        pol = load_policy(args.policy, args.device, cfg["task"]["instruction"], args.steps)
    if not generic and pol.kind == "pi05" and not args.no_compile:
        pol.enable_compile()
        compiler = "mx.compile" if backend == "mlx" else "torch.compile"
        print(f"pi0.5: {compiler} on; the first inference compiles.", flush=True)
    if args.mode == "auto":
        args.mode = ("async" if selection.requested == "mlx" else "rtc") if pol.kind == "pi05" else "ensemble"
    if getattr(args, "query_every", 0) and args.mode not in ("async", "rtc"):
        sys.exit("--query-every applies to --mode async or rtc only")
    if args.mode == "step" and not generic:
        sys.exit("--mode step requires a generic policy that uses select_action")
    if args.mode == "rtc" and not pol.supports_rtc:
        sys.exit("--mode rtc needs pi0.5")
    sizes = {c: tuple(cfg["cameras"][c]["save_size"]) for c in st.cam_names}
    if pol.shapes is not None and set(pol.shapes) != set(st.cam_names):
        sys.exit(f"policy cameras {sorted(pol.shapes)} do not match config cameras {sorted(st.cam_names)}")
    for c in (st.cam_names if pol.shapes is not None else []):
        want = (3, sizes[c][1], sizes[c][0])
        if pol.shapes.get(c) != want:
            sys.exit(f"policy input {c} is {pol.shapes.get(c)}, but the camera config gives {want}")
    jq = int(cfg["record"]["jpeg_quality"])
    dummy = {c: np.zeros((sizes[c][1], sizes[c][0], 3), np.uint8) for c in st.cam_names}
    for _ in range(3):                                   # warm-up (MPS is slow at first; compile happens here)
        t = time.perf_counter()
        try:
            a = pol(dummy, [0.0] * n_j)
        except Exception as exc:
            if not generic:
                raise
            sys.exit(f"{pol.kind} warm-up failed on {args.device}: {exc}")
    if generic:
        pol.reset()
    if a.shape[-1] != n_j:
        sys.exit(f"policy outputs {a.shape[-1]} values per step, but {robot.kind} has {n_j} joints ({', '.join(names)})")
    execution = "step" if generic and pol.stepwise else f"chunk {pol.H}"
    print(f"policy ready ({pol.kind}, {execution}): {(time.perf_counter() - t) * 1e3:.0f} ms per inference", flush=True)
    if args.mode == "rtc":
        pol.enable_rtc(args.rtc_beta, getattr(args, "rtc_schedule", "linear"))
        pol(dummy, [0.0] * n_j)                          # unguided call (first chunk) compiles separately
        lo = np.zeros((pol.H, n_j), np.float32)
        pol(dummy, [0.0] * n_j, leftover=lo, delay=8)
        t = time.perf_counter()
        pol(dummy, [0.0] * n_j, leftover=lo, delay=8)
        print(f"rtc warm-up: {(time.perf_counter() - t) * 1e3:.0f} ms per guided inference", flush=True)

    st.cams = open_cameras(cfg["cameras"], fps, args.fake, robot)
    cams = st.cams
    for cam in cams.values():
        cleanup.callback(cam.close)
    robot.connect(leader=st.takeover)
    st.robot_info = robot.info()
    st.limits = {n: list(v) for n, v in robot.limits.items()}
    home = robot.home_pose()
    cmd = robot.read()                                    # start from the measured pose
    sent_v = list(cmd)
    st.q, st.cmd = list(cmd), list(cmd)
    hz = rc["hz"]
    dt = 1.0 / hz
    leader_reader = LeaderReader(robot, dt, cleanup) if st.takeover and not (args.fake or st.sim) else None

    def observe():
        with st.lock:
            q = list(st.q)
            frames = {c: cams[c].latest()[0] for c in st.cam_names}
        return {c: prep_image(frames[c], sizes[c], jq)[0] for c in st.cam_names}, q

    rate = fps * args.speed
    stop = lambda: st.quit  # noqa: E731
    if args.mode == "step":
        ex = Step(pol, observe, rate, st.log_chunk)
    elif args.mode == "chunk":
        ex = SyncChunk(pol, observe, rate, st.log_chunk, min(args.k, pol.H) if generic else args.k)
    elif args.mode == "ensemble":
        ex = Ensemble(pol, observe, rate, st.log_chunk, args.te_m, args.te_hz, stop)
    else:
        ex = AsyncChunk(pol, observe, rate, st.log_chunk, guide=args.mode == "rtc", horizon=args.rtc_horizon,
                        every=getattr(args, "query_every", 0), stop=stop)
    ex.start()
    st.executor = ex.describe()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cleanup.callback(srv.shutdown)
    print(f"executor: {st.executor}\nweb page: {srv.url}   rollouts: {st.out}", flush=True)
    if not getattr(args, "no_browser", True) and not args.autostart:
        web.show(srv)

    vmax = rc["max_vel"] * args.speed * dt               # max arm joint step per control cycle (robot units)
    grip_lead = float(getattr(args, "grip_lead", 0.0) or 0.0)   # seconds the close decision looks ahead
    grip_step = 300.0 * args.speed * dt                  # gripper step while following or homing
    home_vmax = rc.get("home_vel", 25.0) * dt             # homing is slow regardless of speed
    home_t0 = 0.0
    catch_t0 = None                                       # set while the follower catches up with the leader
    leader_lost = None                                    # when the leader signal dropped while following
    writer = cleanup.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix="watchdog"))
    diagnostic = None

    def say(en, zh, mode=None):
        with st.lock:
            st.msg = tr(en, zh)
            if mode:
                st.set_mode(mode)

    def restart_policy():
        if generic and args.mode == "chunk":
            pol.reset()
        ex.reset(True)

    try:
        started = True
        chunk_id, gp_last, gp_ahead = -1, float("nan"), float("nan")
        next_rec = time.time()
        auto = [(0.0, "start"), (args.autostart, "save_fail"), (args.autostart + 0.5, "quit")] if args.autostart else []
        auto_t0 = time.time()
        print("ready (paused)", flush=True)
        while True:
            tick = time.perf_counter()
            now = time.time()
            if auto and now - auto_t0 >= auto[0][0]:
                with st.lock:
                    st.want = auto.pop(0)[1]
            want = st.take_command()
            if want == "quit" and st.watch.trip:
                break
            q = robot.read()
            motors = robot.motor_telemetry()
            trip_record = None
            with st.lock:
                st.q = q
                was_tripped = st.watch.trip is not None
                events = st.watch.update(time.perf_counter(), q, sent_v, motors, home=home,
                                         holding=st.mode in ("paused", "aligning") or leader_lost is not None)
                if events and st.rec is not None:
                    st.rec["watchdog_events"].extend(timed_events(events, now, st.rec["t0"]))
                if st.watch.trip and not was_tripped:
                    st.set_mode("paused")
                    trip_record = st.rec
                    if trip_record is None:
                        trip_record = st.new_record(now)
                        trip_record["watchdog_events"] = timed_events(events, now, now)
                    trip_record["motor_at_trip"] = motors
                    st.rec = None
                    st.msg = st.watch.blocked()
            if st.watch.trip:
                # Rebase before archival or any operator command can act on the old target.
                try:
                    sent_v = robot.hold(st.watch.pose, q) if not st.sim else list(st.watch.pose)
                finally:
                    if trip_record is not None:
                        print(st.msg["en"], flush=True)
                        diagnostic = writer.submit(save_rollout, st, trip_record, "watchdog")
                with st.lock:
                    st.cmd = sent_v
                if not was_tripped and getattr(ex, "active", True):
                    ex.reset(False)
                if want == "quit":
                    break
                time.sleep(max(0.0, dt - (time.perf_counter() - tick)))
                continue
            if want == "quit":
                break
            if (want == "start" or (want == "takeover" and st.mode in TAKEOVER_MODES)) and st.watch.blocked():
                with st.lock:
                    st.msg = st.watch.blocked()
                want = None
            if st.sim and want in ("start", "home"):
                if st.rec is None:
                    robot.reset_scene()
                    q = cmd = robot.read()
                    motors, sent_v = robot.motor_telemetry(), list(cmd)
                    st.watch.reset_motion()
                    with st.lock:
                        st.q, st.cmd = list(cmd), list(cmd)
                    now = time.time()
                    if want == "home":
                        say("Simulation reset.", "仿真场景已重置。", "paused")
                        want = None
                elif robot.ended or want == "home":
                    say("Save or discard this trial before resetting the simulation.",
                        "请先保存或丢弃这次试验，再重置仿真场景。", "paused")
                    want = None
            stalled = [c for c in st.cam_names if
                       (cams[c].latest()[0] is None if st.sim else now - cams[c].latest()[1] > CAM_STALL_S)]
            if want == "start" and st.mode not in TAKEOVER_MODES:
                if stalled:
                    say(f"No frames from {', '.join(stalled)}; cannot start.", f"{'、'.join(stalled)} 相机没有画面，不能开始。")
                else:
                    say("Policy running. Space pauses.", "策略运行中。空格暂停。", "running")
                    with st.lock:
                        if st.rec is None:
                            chunk_id, gp_last, gp_ahead, next_rec = -1, float("nan"), float("nan"), now
                            st.rec = st.new_record(now)
                        if st.rec["takeover_count"] and gripper is not None:
                            gripper.sync(cmd[gi])
                    restart_policy()
            elif want == "takeover" and st.takeover and st.mode == "running":
                ex.reset(False)
                gp_last, gp_ahead, chunk_id = float("nan"), float("nan"), -1
                with st.lock:
                    st.rec["takeover_count"] += 1
                say("Holding the follower. Align the leader to take control.",
                    "从臂保持当前指令。请对齐领臂以接管。", "aligning")
            elif want == "takeover" and st.mode in TAKEOVER_MODES:
                if stalled or (st.sim and robot.ended):
                    say("Cannot resume the policy. Trial paused.", "暂时无法恢复策略，试验已暂停。", "paused")
                else:
                    if gripper is not None:
                        gripper.sync(cmd[gi])
                    say("Policy running. Space pauses.", "策略运行中。空格暂停。", "running")
                    restart_policy()
            elif want == "pause":
                say("Paused (holding the current command).", "已暂停（保持当前指令）。", "paused")
            elif want == "home":
                say("Moving slowly to the start pose...", "正在慢速回到起始姿态…", "homing")
                home_t0 = now
            elif want in ("save_ok", "save_fail", "discard"):
                with st.lock:
                    st.set_mode("paused")
                    rec, st.rec = st.rec, None
                if rec and want != "discard" and rec["t"]:
                    name, meta = save_rollout(st, rec, "success" if want == "save_ok" else "failure")
                    with st.lock:
                        st.next = int(Path(name).stem.split("_")[-1]) + 1
                        st.saved_revision += 1
                    ok = want == "save_ok"
                    say(f"Saved {name} ({'success' if ok else 'failure'}). Reset the scene and press Start.",
                        f"已保存 {name}（{'成功' if ok else '失败'}）。复位场景后按开始。")
                    print(f"saved {name} {meta['result']} {meta['frames']} frames", flush=True)
                else:
                    say("Trial discarded. Reset the scene and press Start.", "这次试验已丢弃。复位场景后按开始。")
            with st.lock:
                mode = st.mode
            if ex.err and (not st.takeover or mode == "running"):
                say(f"Inference failed, paused: {ex.err}", f"推理出错，已暂停：{ex.err}", "paused")
                print(f"inference error: {ex.err}", flush=True)
                ex.err, mode = None, "paused"
            if mode in TRIAL_MODES and st.sim and robot.ended:
                say("Simulation episode ended. Save or discard the trial.",
                    "仿真回合已结束。请保存或丢弃这次试验。", "paused")
                mode = "paused"
            if mode in TRIAL_MODES and stalled:
                say(f"Camera stalled ({', '.join(stalled)}), paused.", f"相机断流（{'、'.join(stalled)}），已自动暂停。", "paused")
                mode = "paused"
            if mode != "running" and getattr(ex, "active", True):
                ex.reset(False)

            leader = (leader_reader.get() if leader_reader else leader_pose(robot)) if st.takeover and mode in TRIAL_MODES else None
            target = list(cmd)
            if mode == "running":
                a, chunk_id = ex.target(time.time())
                if a is not None:
                    target = [float(v) for v in a]
                    if gi is not None:
                        gp_last = target[gi]
                        if gripper is not None:
                            ahead = ex.peek(now + grip_lead) if grip_lead > 0 else None
                            gp_ahead = float(ahead[gi]) if ahead is not None else float("nan")
                            target[gi] = gripper(gp_last, now, gp_ahead if ahead is not None else None)
            elif mode in TAKEOVER_MODES:
                errors = ({n: a - x for n, a, x in zip(names, leader, q, strict=True) if n != robot.gripper}
                          if leader is not None else {})
                grip = (takeover_gripper_alignment(leader[gi], cmd[gi], st.binary_gripper, st.gripper_gate)
                        if leader is not None and gi is not None else None)
                with st.lock:
                    st.alignment = errors
                    st.gripper_alignment = grip
                if leader is None and mode == "following" and now - (leader_lost or now) < LEADER_GRACE_S:
                    leader_lost = leader_lost or now      # brief dropout: hold the command, keep the takeover
                elif leader is None:
                    leader_lost = None
                    say("Leader signal lost. Holding the follower; align again when the signal returns.",
                        "领臂信号丢失。从臂保持当前位置，信号恢复后请重新对齐。", "aligning")
                    mode = "aligning"
                elif mode == "following" and leader_lost is not None:
                    leader_lost, catch_t0 = None, now     # signal back: soft start again, no re-alignment
                elif mode == "aligning":
                    clamped_errors = robot.alignment_errors(leader, q)
                    if (want != "takeover" and all(abs(d) <= st.align_deg for d in errors.values())
                            and all(abs(d) <= st.align_deg for d in clamped_errors.values())
                            and (grip is None or grip["ready"])):
                        # Nobody can hold six joints within a few degrees by hand: start from a rough match. The
                        # follower starts at home_vel along a straight line and reaches max_vel within CATCHUP_S.
                        say("Following the leader.", "正在跟随领臂。", "following")
                        mode = "following"
                        catch_t0, leader_lost = now, None
                    else:
                        say("Holding the follower. Align the leader to take control.",
                            "从臂保持当前指令。请对齐领臂以接管。")
                if mode == "following":
                    target = leader if leader is not None else list(cmd)
            elif mode == "homing":
                target = list(home)
                if all(abs(cmd[j] - home[j]) < 0.5 for j in range(n_j) if j != gi):
                    say("Back at the start pose. Reset the scene and press Start.", "已回到起始姿态。复位场景后按开始。", "paused")

            if mode == "homing":
                ramp = min(1.0, 0.2 + (now - home_t0) / HOME_RAMP_S)
                steps = homing_steps(cmd, home, gi, home_vmax * ramp)
            elif mode == "following" and catch_t0 is not None:
                ramp = (now - catch_t0) / CATCHUP_S
                if ramp >= 1 or all(abs(target[j] - cmd[j]) <= CATCHUP_DONE_DEG for j in range(n_j) if j != gi):
                    catch_t0 = None                       # caught up: follow at the normal limit
                else:
                    steps = homing_steps(cmd, target, gi, home_vmax + (vmax - home_vmax) * max(0.0, ramp))
            ramping = mode == "following" and catch_t0 is not None
            catching = ramping or (mode == "following" and leader_lost is not None)
            for j, n in enumerate(names):                 # velocity limit (arm joints) + joint limits
                if j == gi:
                    step = grip_step if mode in ("homing", "following") else 1e9
                else:
                    step = steps[j] if mode == "homing" or ramping else vmax
                cmd[j] = robot.clamp(n, cmd[j] + max(-step, min(step, target[j] - cmd[j])))
            with st.lock:
                rec = st.rec if st.mode in TRIAL_MODES else None
            sample = rec is not None and now >= next_rec
            if sample:
                frames = {c: cams[c].latest()[0] for c in st.cam_names}
            if st.sim:
                with st.lock:
                    sent_v = robot.send(cmd, q) if mode in ("running", "following") else list(cmd)
                    st.q = robot.read()
                    st.cmd = sent_v
            else:
                sent_v = robot.send(cmd, q)
                with st.lock:
                    st.cmd = sent_v
            if sample:                                    # record at fps
                next_rec = max(next_rec + 1 / fps, now)
                rec["q"].append(q)
                rec["cmd"].append(sent_v)
                rec["t"].append(now - rec["t0"])
                rec["chunk"].append(chunk_id)
                rec["gp"].append(gp_last)
                rec["gp_ahead"].append(gp_ahead)
                rec["motor"].append(motors)
                if st.takeover:
                    # While the follower catches up or holds through a dropout, the motion is not the operator's.
                    rec["takeover"].append(TAKEOVER_MODES["aligning"] if catching else TAKEOVER_MODES.get(mode, 0))
                    rec["leader"].append(leader if leader is not None else [float("nan")] * n_j)
                for c in st.cam_names:
                    rec[f"{c}_jpg"].append(prep_image(frames[c], sizes[c], jq)[1])
            time.sleep(max(0.0, dt - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        st.quit = True
        if isinstance(ex, threading.Thread):
            ex.join(timeout=2)
        if diagnostic is not None:
            name, _ = diagnostic.result()
            print(f"watchdog diagnostic saved: {st.out / 'watchdog' / name}", flush=True)
        print("done", flush=True)
