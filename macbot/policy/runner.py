"""Run a trained policy on the robot, driven from a local web page.

Shares cameras, follower settings and the safe shutdown with ``macbot collect``. Observations match
training: images resized to the recorded size and passed through JPEG, state = measured joints (deg).

Safety: starts paused; arm joints move at most ``max_vel x speed`` deg/s; joint limits; gripper force
clamp; a camera that stalls for 0.5 s pauses the run; pause holds the current command; "home" moves
slowly to the zero pose; quitting returns to zero in two stages before disabling.

Each trial is saved as ``<data_root>/rollouts/<task>/rollout_XXX.npz``: qpos, cmd, t, chunk_id per
frame, chunk_t and infer_ms per chunk, executor extras, JPEG frames per camera, and meta (result).
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .. import web
from ..cameras import open_cameras
from ..config import Config
from ..i18n import tr
from ..system import ensure_dir, expand, give_back, start_log
from .executors import AsyncChunk, Ensemble, SyncChunk
from .gripper import BinaryGripper, find_gripper_config
from .models import load_policy, prep_image

CAM_STALL_S = 0.5


class State:
    def __init__(self, cfg: Config, out: Path, args):
        self.cfg, self.out, self.args = cfg, out, args
        self.cam_names = list(cfg["cameras"])
        self.lock = threading.Lock()
        self.mode = "paused"            # paused | running | homing
        self.want = None                # request from the page
        self.msg = tr("Paused. Set up the scene, make sure it is safe, then press Start.",
                      "已暂停。场景摆好、确认安全后按“开始”。")
        self.joints = []
        self.q, self.cmd = [], []
        self.cams, self.chunks, self.infer_ms = {}, 0, []
        self.quit = False
        self.rec = None
        self.saved = []
        self.executor = ""
        self.gripper, self.robot_info, self.limits = None, {}, {}

    def next_idx(self):
        ids = [int(m.group(1)) for p in self.out.glob("rollout_*.npz") if (m := re.search(r"_(\d+)\.npz$", p.name))]
        return max(ids, default=-1) + 1

    def log_chunk(self, t_chunk, ms, **extra):
        """Called by executors after every inference."""
        with self.lock:
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
                    "executor": self.executor, "speed": self.args.speed, "fake": self.args.fake,
                    "policy": "/".join(Path(self.args.policy).parts[-2:]), "task": self.cfg.task,
                    "rec_s": round(len(r["t"]) / self.cfg["record"]["fps"], 1) if r else 0,
                    "saved": self.saved[-15:], "next": self.next_idx()}


def save_rollout(st: State, rec: dict, result: str):
    p = st.out / f"rollout_{st.next_idx():03d}.npz"
    arrs = {}
    for c in st.cam_names:
        js = rec[f"{c}_jpg"]
        arrs[f"{c}_offsets"] = np.cumsum([0] + [len(j) for j in js]).astype(np.int64)
        arrs[f"{c}_jpeg"] = np.concatenate(js) if js else np.zeros(0, np.uint8)
    a = st.args
    fps = st.cfg["record"]["fps"]
    meta = {"task": st.cfg.task, "instruction": st.cfg["task"]["instruction"], "policy": str(a.policy),
            "executor": st.executor, "mode": a.mode, "k": a.k, "te_m": a.te_m, "rtc_beta": a.rtc_beta,
            "grip_debounce": a.grip_debounce, "speed": a.speed, "max_vel": st.cfg["run"]["max_vel"], "result": result,
            "fps": fps, "frames": len(rec["t"]), "joint_names": st.joints, "gripper": st.gripper, "robot": st.robot_info, "cameras": st.cam_names,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "fake": a.fake, "writer": "macbot"}
    ex = {k: np.asarray(v, np.float64) for k, v in rec.items() if k.startswith("ex_")}
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, qpos=np.asarray(rec["q"], np.float32), cmd=np.asarray(rec["cmd"], np.float32),
             t=np.asarray(rec["t"], np.float64), chunk_id=np.asarray(rec["chunk"], np.int32),
             chunk_t=np.asarray(rec["chunk_t"], np.float64), infer_ms=np.asarray(rec["infer_ms"], np.float32),
             gp=np.asarray(rec["gp"], np.float32), meta=json.dumps(meta, ensure_ascii=False), **ex, **arrs)
    tmp.rename(p)
    give_back(p)
    return p.name, meta


def make_handler(st: State):
    preview = {k: tuple(st.cfg["cameras"][k]["save_size"]) for k in st.cam_names}

    class H(web.Handler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self.send(200, web.page("run", task=st.cfg.task), "text/html; charset=utf-8")
            if self.static(path):
                return
            if path == "/api/status":
                return self.json(st.status())
            m = re.fullmatch(r"/stream/(\w+)\.mjpg", path)
            if m and m.group(1) in st.cams:
                return self.mjpeg(st.cams[m.group(1)], preview[m.group(1)], lambda: st.quit)
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/cmd":
                return self.send(404, b"", "text/plain")
            try:
                act = self.body_json().get("action")
                if act in ("start", "pause", "home", "save_ok", "save_fail", "discard", "quit"):
                    with st.lock:
                        st.want = act
            except Exception as e:
                with st.lock:
                    st.msg = tr(f"Command failed: {e}", f"操作失败：{e}")
            self.json(st.status())

    return H


def run(cfg: Config, args):
    start_log(cfg.log_dir, "run")
    rc = cfg["run"]
    args.speed = args.speed if args.speed is not None else rc["speed"]
    if not 0.1 <= args.speed <= 1.0:
        sys.exit("--speed must be between 0.1 and 1.0")
    if args.autostart and not args.fake:
        sys.exit("--autostart only works with --fake")
    args.policy = expand(args.policy)
    gcfg = find_gripper_config(args.policy, args.gripper)
    gripper = BinaryGripper(gcfg, args.grip_debounce) if gcfg else None

    fps = int(cfg["record"]["fps"])
    out = ensure_dir(cfg.rollout_dir(cfg.task + ("_fake" if args.fake else "")))
    st = State(cfg, out, args)
    port = args.port or rc["port"]
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(st))
    srv.daemon_threads = True

    from ..robot import open_robot
    robot = open_robot(cfg["robot"], fake=args.fake)
    names, n_j = robot.joints, len(robot.joints)
    with st.lock:
        st.joints, st.gripper = names, robot.gripper
        st.q = st.cmd = [0.0] * n_j
    gi = robot.gripper_index
    if gripper is not None and gi is None:
        sys.exit("gripper.json found but the robot has no joint named gripper; use --gripper raw")
    print(f"loading policy {args.policy} on {args.device}", flush=True)
    pol = load_policy(args.policy, args.device, cfg["task"]["instruction"], args.steps)
    if pol.kind == "pi05" and not args.no_compile:
        pol.enable_compile()
        print("pi0.5: torch.compile on, the first inference compiles (~25 s)...", flush=True)
    if args.mode == "auto":
        args.mode = "rtc" if pol.kind == "pi05" else "ensemble"
    if args.mode == "rtc" and not pol.supports_rtc:
        sys.exit("--mode rtc needs pi0.5")
    sizes = {c: tuple(cfg["cameras"][c]["save_size"]) for c in st.cam_names}
    for c in (st.cam_names if pol.shapes is not None else []):
        want = (3, sizes[c][1], sizes[c][0])
        if pol.shapes.get(c) != want:
            sys.exit(f"policy input {c} is {pol.shapes.get(c)}, but the camera config gives {want}")
    jq = int(cfg["record"]["jpeg_quality"])
    dummy = {c: np.zeros((sizes[c][1], sizes[c][0], 3), np.uint8) for c in st.cam_names}
    for _ in range(3):                                   # warm-up (MPS is slow at first; compile happens here)
        t = time.perf_counter()
        a = pol(dummy, [0.0] * n_j)
    if a.shape[-1] != n_j:
        sys.exit(f"policy outputs {a.shape[-1]} values per step, but {robot.kind} has {n_j} joints ({', '.join(names)})")
    print(f"policy ready ({pol.kind}, chunk {pol.H}): {(time.perf_counter() - t) * 1e3:.0f} ms per inference", flush=True)
    if args.mode == "rtc":
        pol.enable_rtc(args.rtc_beta)
        pol(dummy, [0.0] * n_j)                          # unguided call (first chunk) compiles separately
        lo = np.zeros((pol.H, n_j), np.float32)
        pol(dummy, [0.0] * n_j, leftover=lo, delay=8)
        t = time.perf_counter()
        pol(dummy, [0.0] * n_j, leftover=lo, delay=8)
        print(f"rtc warm-up: {(time.perf_counter() - t) * 1e3:.0f} ms per guided inference", flush=True)

    st.cams = open_cameras(cfg["cameras"], fps, args.fake)
    cams = st.cams

    def observe():
        imgs = {c: prep_image(cams[c].latest()[0], sizes[c], jq)[0] for c in st.cam_names}
        with st.lock:
            return imgs, list(st.q)

    rate = fps * args.speed
    stop = lambda: st.quit  # noqa: E731
    if args.mode == "chunk":
        ex = SyncChunk(pol, observe, rate, st.log_chunk, args.k)
    elif args.mode == "ensemble":
        ex = Ensemble(pol, observe, rate, st.log_chunk, args.te_m, args.te_hz, stop)
    else:
        ex = AsyncChunk(pol, observe, rate, st.log_chunk, guide=args.mode == "rtc", horizon=args.rtc_horizon, stop=stop)
    ex.start()
    st.executor = ex.describe()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"executor: {st.executor}\nweb page: http://127.0.0.1:{port}   rollouts: {out}", flush=True)

    robot.connect()
    st.robot_info = robot.info()
    st.limits = {n: list(v) for n, v in robot.limits.items()}
    home = robot.home_pose()
    started = False
    hz = rc["hz"]
    dt = 1.0 / hz
    vmax = rc["max_vel"] * args.speed * dt               # max arm joint step per control cycle (robot units)

    def say(en, zh, mode=None):
        with st.lock:
            st.msg = tr(en, zh)
            if mode:
                st.mode = mode

    try:
        cmd = robot.read()                                # start from the measured pose (no jump)
        started = True
        chunk_id, gp_last = -1, float("nan")
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
            with st.lock:
                want, st.want = st.want, None
            if want == "quit":
                break
            stalled = [c for c in st.cam_names if now - cams[c].latest()[1] > CAM_STALL_S]
            if want == "start":
                if stalled:
                    say(f"No frames from {', '.join(stalled)}; cannot start.", f"{'、'.join(stalled)} 相机没有画面，不能开始。")
                else:
                    say("Policy running. Space pauses.", "策略运行中。空格暂停。", "running")
                    with st.lock:
                        if st.rec is None:
                            st.rec = {"q": [], "cmd": [], "t": [], "chunk": [], "chunk_t": [], "infer_ms": [], "gp": [],
                                      "t0": now, **{f"{c}_jpg": [] for c in st.cam_names}}
                    ex.reset(True)                        # every start re-plans from the current observation
            elif want == "pause":
                say("Paused (holding the current command).", "已暂停（保持当前指令）。", "paused")
            elif want == "home":
                say("Moving slowly to the start pose...", "正在慢速回到起始姿态…", "homing")
            elif want in ("save_ok", "save_fail", "discard"):
                with st.lock:
                    rec, st.rec, st.mode = st.rec, None, "paused"
                if rec and want != "discard" and rec["t"]:
                    name, meta = save_rollout(st, rec, "success" if want == "save_ok" else "failure")
                    with st.lock:
                        st.saved.append({"name": name, "result": meta["result"], "s": round(meta["frames"] / fps, 1)})
                    ok = want == "save_ok"
                    say(f"Saved {name} ({'success' if ok else 'failure'}). Reset the scene and press Start.",
                        f"已保存 {name}（{'成功' if ok else '失败'}）。复位场景后按开始。")
                    print(f"saved {name} {meta['result']} {meta['frames']} frames", flush=True)
                else:
                    say("Trial discarded. Reset the scene and press Start.", "这次试验已丢弃。复位场景后按开始。")
            with st.lock:
                mode = st.mode
            if ex.err:
                say(f"Inference failed, paused: {ex.err}", f"推理出错，已暂停：{ex.err}", "paused")
                print(f"inference error: {ex.err}", flush=True)
                ex.err, mode = None, "paused"
            if mode != "running" and getattr(ex, "active", True):
                ex.reset(False)
            if mode == "running" and stalled:
                say(f"Camera stalled ({', '.join(stalled)}), paused.", f"相机断流（{'、'.join(stalled)}），已自动暂停。", "paused")
                mode = "paused"

            q = robot.read()
            with st.lock:
                st.q = q
            target = list(cmd)
            if mode == "running":
                a, chunk_id = ex.target(time.time())
                if a is not None:
                    target = [float(v) for v in a]
                    if gi is not None:
                        gp_last = target[gi]
                        if gripper is not None:
                            target[gi] = gripper(gp_last, now)
            elif mode == "homing":
                target = list(home)
                if all(abs(cmd[j] - home[j]) < 0.5 for j in range(n_j) if j != gi):
                    say("Back at the start pose. Reset the scene and press Start.", "已回到起始姿态。复位场景后按开始。", "paused")

            for j, n in enumerate(names):                 # velocity limit (arm joints) + joint limits
                step = (300.0 * args.speed * dt if mode == "homing" else 1e9) if j == gi else vmax
                cmd[j] = robot.clamp(n, cmd[j] + max(-step, min(step, target[j] - cmd[j])))
            sent_v = robot.send(cmd, q)
            with st.lock:
                st.cmd = sent_v
                rec = st.rec if st.mode == "running" else None
            if rec is not None and now >= next_rec:       # record at fps
                next_rec = max(next_rec + 1 / fps, now)
                rec["q"].append(q)
                rec["cmd"].append(sent_v)
                rec["t"].append(now - rec["t0"])
                rec["chunk"].append(chunk_id)
                rec["gp"].append(gp_last)
                for c in st.cam_names:
                    rec[f"{c}_jpg"].append(prep_image(cams[c].latest()[0], sizes[c], jq)[1])
            time.sleep(max(0.0, dt - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        st.quit = True
        try:
            robot.shutdown(moved=started)
        finally:
            for c in cams.values():
                c.close()
            srv.shutdown()
            print("done", flush=True)
