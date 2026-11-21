"""Leader-follower teleoperation with recording, driven from a local web page.

Workflow (same as ``lerobot-record``): record one episode, reset the scene, record the next.
Keys on the page: S start, Enter or right arrow save, left arrow discard. Every saved episode can
be replayed (all cameras plus joint traces) or deleted (moved to ``deleted/``, numbers are not reused).
Each save runs quality checks (frame age, repeated frames, joint jumps, late samples).

Episode file ``<data_root>/raw/<task>/episode_XXX.npz``:
  qpos (N,J)      follower measured joints, in robot units
  action (N,J)    command sent to the follower (reBot gripper: leader value before the force clamp)
  timestamp (N,)  k / fps (fixed grid); wall_time (N,) actual sampling time
  <cam>_jpeg, <cam>_offsets   JPEG stream per camera, frame i = jpeg[off[i]:off[i+1]]
  <cam>_age, <cam>_repeat     age of the frame at sampling time; same frame as the previous sample
  meta            JSON: task, instruction, fps, joint names, camera info, follower feedforward, check result
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time

import cv2
import numpy as np

from .. import web
from ..cameras import camera_info, open_cameras
from ..cameras.framing import framing
from ..config import Config
from ..data.motor_health import diagnostic_path, motor_arrays, record_motor_meta
from ..i18n import text, tr
from ..safety import MotorWatch, timed_events
from ..system import ensure_dir, expand, give_back, start_log
from .checks import check_episode

CAM_STALL_S = 0.5      # a camera without a new frame for this long counts as disconnected


class State:
    def __init__(self, cfg: Config, out, fake: bool):
        self.cfg, self.out, self.fake = cfg, out, fake
        self.sim = not fake and cfg["robot"].get("type") == "sim"
        self.sim_input = cfg["teleop"].get("type", "scripted") if self.sim else None
        self.sim_step_recording = self.sim_input in {"keyboard", "so101_leader"}
        self.sim_outcome = None
        self.key_sample = (0.0, frozenset())
        self.start_requested = False
        self.cam_names = list(cfg["cameras"])
        self.fps = int(cfg["record"]["fps"])
        self.lock = threading.Lock()
        self.phase = "idle"                 # idle | recording | stopped
        self.t0 = 0.0
        self.instruction = cfg["task"]["instruction"]
        self.msg = tr("Loading simulation...", "正在加载仿真…") if self.sim else tr(
            "Align the arms. Teleoperation starts after the alignment check.", "请对齐两条臂。对齐检查通过才开始遥操作。")
        self.joints, self.gripper = [], None
        self.q, self.cmd = [], []
        self.motors = ()
        self.limits, self.ready, self.quit = {}, False, False
        self.buf = None
        self.saving = 0
        self.ff = {}
        self.cam_info = camera_info(cfg["cameras"])
        self.summaries = {}                 # file name -> (mtime, summary)
        self.cams = {}
        self.watch = MotorWatch(cfg["safety"])
        self.refresh()

    def new_buffer(self):
        keys = ["qpos", "action", "wall", "late", "motor", "watchdog_events"]
        keys += [f"{k}_{s}" for k in self.cam_names for s in ("jpg", "age", "repeat")]
        return {k: [] for k in keys}

    def files(self):
        return sorted(self.out.glob("episode_*.npz"))

    def refresh(self):
        keep = {}
        for p in self.files():
            m = p.stat().st_mtime
            old = self.summaries.get(p.name)
            if old and old[0] == m:
                keep[p.name] = old
                continue
            idx = int(p.stem.split("_")[1])
            try:
                with np.load(p) as z:
                    meta = json.loads(str(z["meta"]))
                keep[p.name] = (m, {"idx": idx, "dur": meta["duration_s"], "frames": meta["frames"],
                                    "time": meta["saved_at"][5:16], "instruction": meta["instruction"],
                                    "check": meta.get("check")})
            except Exception as e:          # half-written or damaged
                keep[p.name] = (m, {"idx": idx, "dur": 0, "frames": 0, "time": "", "instruction": "",
                                    "check": {"ok": False, "errors": [tr(f"unreadable: {e}", f"读不出：{e}")], "warnings": []}})
        self.summaries = keep

    def next_idx(self):
        ids = [int(p.stem.split("_")[1]) for p in self.files()]
        deleted = self.out / "deleted"
        if deleted.exists():
            ids += [int(m.group(1)) for p in deleted.glob("episode_*.npz") if (m := re.match(r"episode_(\d+)", p.stem))]
        return max(ids, default=-1) + 1

    def cam_age(self):
        return {k: (round(a, 2) if (a := c.age()) is not None else None) for k, c in self.cams.items()}

    def stalled(self):
        return [k for k, a in self.cam_age().items() if a is None or a > CAM_STALL_S]

    def status(self):
        ages = self.cam_age()
        with self.lock:
            eps = [v[1] for _, v in sorted(self.summaries.items())][::-1]
            rec = self.phase == "recording"
            sim = {"sim_input": self.sim_input, "sim_outcome": self.sim_outcome} if self.sim else {}
            return {"phase": self.phase, "ready": self.ready, "fake": self.fake, "sim": self.sim,
                    "elapsed": time.time() - self.t0 if rec else 0.0,
                    "frames": len(self.buf["qpos"]) if (rec and self.buf) else 0,
                    "msg": self.msg, "instruction": self.instruction, "task": self.cfg.task,
                    "target": self.cfg["task"]["target_episodes"], "count": len(eps), "saving": self.saving,
                    "q": self.q, "cmd": self.cmd, "names": self.joints, "limits": self.limits, "episodes": eps,
                    "fps": self.fps, "cams": self.cam_info, "cam_names": self.cam_names, "out": str(self.out),
                    "cam_age": ages, "stall_s": CAM_STALL_S, "motor_health": self.watch.status(), **sim}

    def set_keys(self, keys):
        from ..robot.sim_cartesian import KEYS

        if self.sim_input == "keyboard" and isinstance(keys, list):
            with self.lock:
                self.key_sample = (time.monotonic(), frozenset(k for k in keys if isinstance(k, str) and k in KEYS))

    def keys(self):
        with self.lock:
            stamp, keys = self.key_sample
        return keys if time.monotonic() - stamp < 0.3 else ()

    def command(self, c, *, scene_ready=False):
        a = c.get("action")
        with self.lock:
            if a == "instruction" and self.phase == "idle":
                text = str(c.get("text", "")).strip()
                if text:
                    self.instruction = text
                    self.msg = tr(f"Instruction set: {text}", f"指令改为：{text}")
            elif a == "start" and self.phase == "idle":
                if blocked := self.watch.blocked():
                    self.msg = blocked
                    return
                if not self.ready:
                    self.msg = tr("Teleoperation has not started yet.", "遥操作还没启动。")
                    return
                if self.sim and not scene_ready:
                    self.start_requested = True
                    return
                bad = self.stalled()
                if bad:
                    self.msg = tr(f"No frames from {', '.join(bad)} (cable loose?). Fix it and restart.",
                                   f"{'、'.join(bad)} 相机没有画面（线松了？），插好后重启采集程序。")
                    return
                self.buf = self.new_buffer()
                self.phase, self.t0 = "recording", time.time()
                self.msg = tr("Recording. Enter saves, left arrow discards.", "录制中。回车保存，← 丢弃。")
            elif a == "discard" and self.phase == "recording":
                self.phase, self.buf = "idle", None
                self.msg = tr("Discarded. Reset the scene and press S.", "已丢弃。复位场景后按 S 重录。")
            elif a == "save" and self.phase == "recording":
                buf, self.buf, self.phase = self.buf, None, "idle"
                n = len(buf["qpos"])
                if n / self.fps < self.cfg["record"]["min_episode_s"]:
                    self.msg = tr(f"Not saved: only {n / self.fps:.1f} s.", f"没保存：只有 {n / self.fps:.1f} 秒。")
                    return
                idx = self.next_idx() + self.saving
                self.saving += 1
                self.msg = tr(f"Saving episode {idx:03d} ({n / self.fps:.1f} s)...", f"正在保存第 {idx:03d} 条（{n / self.fps:.1f} 秒）…")
                threading.Thread(target=self._save, args=(idx, buf, self.instruction), daemon=True).start()
            elif a == "delete" and self.phase == "idle":
                p = self.out / f"episode_{int(c.get('idx', -1)):03d}.npz"
                if p.exists():
                    d = ensure_dir(self.out / "deleted")
                    shutil.move(str(p), str(d / p.name))
                    self.refresh()
                    self.msg = tr(f"Episode {p.stem[-3:]} moved to deleted/.", f"第 {p.stem[-3:]} 条已移到 deleted/。")
            elif a == "quit":
                self.quit = True
                if self.sim:
                    self.msg = tr("Closing simulation.", "正在关闭仿真。")
                elif self.watch.trip is None:
                    self.msg = tr("Shutting down: the follower returns home, then disables.",
                                  "正在结束：从臂慢速回到起始位后失能。")

    def _save(self, idx, buf, instruction, *, diagnostic=False, batch=False):
        try:
            arr = {k: np.asarray(v) for k, v in buf.items() if not k.endswith("_jpg")
                   and k not in ("motor", "watchdog_events", "motor_at_trip", "simulation")}
            rc = self.cfg["record"]
            chk = ({"ok": False, "errors": [self.watch.blocked()], "warnings": []} if diagnostic else
                   check_episode(arr, self.cam_names, self.fps, self.cfg["checks"], rc["min_episode_s"],
                                 joints=self.joints, gripper=self.gripper))
            n = len(buf["qpos"])
            meta = {"task": self.cfg.task, "instruction": instruction, "fps": self.fps, "frames": n,
                    "duration_s": round(n / self.fps, 2), "joint_names": self.joints, "gripper": self.gripper,
                    "units": "robot (degrees for most arm joints)",
                    "state": "follower measured position", "action": "command sent to follower (gripper: leader value)",
                    "cameras": self.cam_info, "camera_framing": framing(self.cfg["cameras"], self.fps),
                    "robot": self.ff, "fake": self.fake, "safety": self.cfg["safety"],
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "check": chk, "writer": "macbot"}
            record_motor_meta(buf, meta)
            if "simulation" in buf:
                meta["simulation"] = buf["simulation"]
            if diagnostic:
                meta["result"] = "watchdog"
            extra = motor_arrays(buf["motor"], len(self.joints))
            for k in self.cam_names:
                extra[f"{k}_jpeg"] = np.concatenate(buf[f"{k}_jpg"]) if n else np.empty(0, np.uint8)
                extra[f"{k}_offsets"] = np.cumsum([0] + [len(b) for b in buf[f"{k}_jpg"]]).astype(np.int64)
                extra[f"{k}_age"] = arr[f"{k}_age"].astype(np.float32)
                extra[f"{k}_repeat"] = arr[f"{k}_repeat"].astype(bool)
            path = diagnostic_path(self.out, "episode") if diagnostic else self.out / f"episode_{idx:03d}.npz"
            tmp = path.with_suffix(".tmp.npz")
            shape = (n, len(self.joints))
            np.savez(tmp, qpos=arr["qpos"].astype(np.float32).reshape(shape), action=arr["action"].astype(np.float32).reshape(shape),
                     timestamp=np.arange(n) / self.fps, wall_time=arr["wall"],
                     meta=np.array(json.dumps(meta, ensure_ascii=False)), **extra)
            tmp.rename(path)
            give_back(path)
            if diagnostic:
                print(f"watchdog diagnostic saved: {path}", flush=True)
                return
            if batch:
                return path
            items = chk["errors"] + chk["warnings"]
            en, zh = "; ".join(text(x, "en") for x in items), "；".join(text(x, "zh") for x in items)
            with self.lock:
                self.refresh()
                k = len(self.summaries)
                if self.watch.trip is None:
                    self.msg = tr(f"Saved episode {idx:03d} ({n / self.fps:.1f} s)" + (f", check: {en}" if en else ", checks passed")
                                  + f". {k} in total. Reset the scene and press S.",
                                  f"已保存第 {idx:03d} 条（{n / self.fps:.1f} 秒）" + (f"，注意：{zh}" if zh else "，检查通过")
                                  + f"。共 {k} 条。复位场景后按 S 录下一条。")
        except Exception as e:
            if batch:
                raise
            if diagnostic:
                print(f"saving watchdog diagnostic failed: {e}", flush=True)
            with self.lock:
                if self.watch.trip is None:
                    self.msg = tr(f"Saving episode {idx:03d} failed: {e}", f"保存第 {idx:03d} 条失败：{e}")
        finally:
            if not (diagnostic or batch):
                with self.lock:
                    self.saving -= 1


class Recorder:
    """Samples joints and cameras on a fixed 1/fps grid while recording."""

    def __init__(self, st: State, *, threaded=True):
        self.st = st
        self.size = {k: tuple(st.cfg["cameras"][k]["save_size"]) for k in st.cam_names}
        self.q = int(st.cfg["record"]["jpeg_quality"])
        if threaded:
            threading.Thread(target=self._loop, daemon=True).start()

    def frame(self, q, motors):
        """Read the observation before applying the action for this frame."""
        now = time.time()
        row = {"qpos": q, "motor": motors}
        for c in self.st.cam_names:
            img, stamp = self.st.cams[c].latest()
            row[f"{c}_jpg"] = self.jpeg(c, img)
            row[f"{c}_age"], row[f"{c}_repeat"] = max(0.0, now - stamp), False
        return row

    def jpeg(self, c, img):
        _, jpg = cv2.imencode(".jpg", cv2.resize(img, self.size[c], interpolation=cv2.INTER_AREA),
                              [cv2.IMWRITE_JPEG_QUALITY, self.q])
        return jpg.reshape(-1)

    def append_step(self, buf, row, cmd, elapsed, *, realtime):
        previous = buf["wall"][-1] if buf["wall"] else 0.0
        late = realtime and elapsed - previous > 2 / self.st.fps
        row.update(action=cmd, wall=elapsed, late=late)
        for key, value in row.items():
            buf[key].append(value)

    def _loop(self):
        st, dt = self.st, 1.0 / self.st.fps
        while not st.quit:
            with st.lock:
                buf, t0 = (st.buf, st.t0) if st.phase == "recording" else (None, 0.0)
            if buf is None:
                time.sleep(0.005)
                continue
            last = {k: None for k in st.cam_names}
            k = 0
            while True:
                target = t0 + k * dt
                wait = target - time.time()
                if wait > 0:
                    time.sleep(wait)
                with st.lock:
                    if st.buf is not buf:
                        break
                    q, cmd, motors = list(st.q), list(st.cmd), st.motors
                now = time.time()
                row = {"qpos": q, "action": cmd, "wall": now - t0, "late": (now - target) > dt, "motor": motors}
                bad = [c for c in st.cam_names if now - st.cams[c].latest()[1] > CAM_STALL_S]
                if bad:
                    with st.lock:
                        if st.buf is buf:
                            st.phase, st.buf = "idle", None
                            st.msg = tr(f"{', '.join(bad)} lost frames for over {CAM_STALL_S} s; this episode was discarded. "
                                        "Check the cable and restart (a dropped camera must be reopened).",
                                        f"{'、'.join(bad)} 相机断流超过 {CAM_STALL_S} 秒，这一条已自动丢弃。检查线后重启采集程序。")
                    print(f"camera stalled: {bad}; episode discarded", flush=True)
                    break
                for c in st.cam_names:
                    img, stamp = st.cams[c].latest()
                    row[f"{c}_jpg"], row[f"{c}_age"], row[f"{c}_repeat"] = self.jpeg(c, img), now - stamp, stamp == last[c]
                    last[c] = stamp
                with st.lock:
                    if st.buf is not buf:
                        break
                    for key, v in row.items():
                        buf[key].append(v)
                k += 1


def make_handler(st: State):
    preview = {k: tuple(st.cfg["cameras"][k]["save_size"]) for k in st.cam_names}

    class H(web.Handler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self.send(200, web.page("collect", task=st.cfg.task), "text/html; charset=utf-8")
            if self.static(path):
                return
            if path == "/api/status":
                return self.json(st.status())
            m = re.fullmatch(r"/stream/(\w+)\.mjpg", path)
            if m and m.group(1) in st.cams:
                return self.mjpeg(st.cams[m.group(1)], preview[m.group(1)], lambda: st.quit)
            m = re.fullmatch(r"/api/episode/(\d+)", path)
            if m:
                p = st.out / f"episode_{int(m.group(1)):03d}.npz"
                if not p.exists():
                    return self.json({}, 404)
                with np.load(p) as d:
                    return self.json({"qpos": d["qpos"].round(2).tolist(), "action": d["action"].round(2).tolist(),
                                      "ts": d["timestamp"].round(3).tolist(), "meta": json.loads(str(d["meta"]))})
            m = re.fullmatch(r"/api/episode/(\d+)/(\w+)/(\d+)", path)
            if m:
                p = st.out / f"episode_{int(m.group(1)):03d}.npz"
                if not p.exists():
                    return self.send(404, b"", "text/plain")
                with np.load(p) as d:
                    off, k = d[f"{m.group(2)}_offsets"], int(m.group(3))
                    k = max(0, min(k, len(off) - 2))
                    return self.send(200, d[f"{m.group(2)}_jpeg"][off[k]:off[k + 1]].tobytes(), "image/jpeg")
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path == "/api/teleop/keys" and st.sim_input == "keyboard":
                try:
                    keys = self.body_json()["keys"]
                except (ValueError, TypeError, KeyError):
                    return self.json({"error": "Expected a list of pressed keys"}, 400)
                if not isinstance(keys, list):
                    return self.json({"error": "Expected a list of pressed keys"}, 400)
                st.set_keys(keys)
                return self.send(204, b"", "text/plain")
            if self.path != "/api/cmd":
                return self.send(404, b"", "text/plain")
            try:
                st.command(self.body_json())
            except Exception as e:
                with st.lock:
                    st.msg = tr(f"Command failed: {e}", f"操作失败：{e}")
            self.json(st.status())

    return H


def run(cfg: Config, fake: bool = False, port: int | None = None, out=None, *, listen_fd=None, open_browser=False):
    listener = (web.bind_console(cfg, "teleop", web.Handler, port) if listen_fd is None
                else web.ConsoleServer.from_listener(listen_fd, web.Handler))
    with listener as srv:
        start_log(cfg.log_dir, "collect")
        out = ensure_dir(expand(out) if out else cfg.raw_dir(cfg.task + "_fake" if fake else cfg.task))
        st = State(cfg, out, fake)
        srv.RequestHandlerClass = make_handler(st)
        # From init, the setup page already links to this console.
        _control(st, srv, cfg["teleop"], open_browser=open_browser and listen_fd is None)


def _control(st, srv, T, open_browser=False):
    from ..robot import open_robot
    robot = (open_robot(st.cfg["robot"], leader=True, teleop=T) if st.sim
             else open_robot(st.cfg["robot"], fake=st.fake, leader=True))
    names = robot.joints
    with st.lock:
        st.joints, st.gripper = names, robot.gripper
        st.watch.set_joints(names, robot.gripper)
        st.q = st.cmd = [0.0] * len(names)
    started = serving = False
    diagnostic_writer = None
    try:
        if st.sim:
            robot.check_rate(T["hz"], "teleop.hz")
        if st.sim_step_recording:
            robot.check_rate(st.fps, "record.fps")
        st.cams = open_cameras(st.cfg["cameras"], st.fps, st.fake, robot)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        serving = True
        print(f"web page: {srv.url}   data: {st.out}", flush=True)
        if open_browser:
            web.show(srv)
        recorder = Recorder(st, threaded=not st.sim_step_recording)
        robot.connect(leader=True)
        st.ff = robot.info()
        a0, q0 = robot.read_leader(), robot.read()
        cmd, home = q0, robot.home_pose()
        motors = robot.motor_telemetry()
        errors = robot.alignment_errors(a0, q0)
        bad = [f"{n} leader {a:.0f} / follower {q:.0f}" for n, a, q in zip(names, a0, q0, strict=False)
               if abs(errors.get(n, 0)) > T["align_max_deg"]]
        if bad and not (st.fake or st.sim):
            with st.lock:
                st.msg = tr(f"Arms not aligned (more than {T['align_max_deg']}): " + "; ".join(bad) + ". Align and restart.",
                            f"两臂没对齐（差超过 {T['align_max_deg']}）：" + "；".join(bad) + "。对齐后重启。")
            print(st.msg["en"], flush=True)
            time.sleep(3)                   # let the page show the message
            raise SystemExit(2)
        with st.lock:
            st.q = st.cmd = q0
            st.watch.samples = st.motors = motors
            st.limits = {n: list(robot.limits[n]) for n in names if n in robot.limits}
            st.msg = tr(f"Teleoperation running. Set up the scene and press S. {len(st.summaries)} episodes so far.",
                        f"遥操作已启动。摆好场景后按 S 开始录制。已有 {len(st.summaries)} 条。")
        started = True
        print(f"teleoperation running ({robot.kind}: {', '.join(names)})", flush=True)
        hz, soft = T["hz"], max(0.0, T["soft_start_s"])
        seed = T.get("seed") if st.sim_step_recording else None
        t_start = time.perf_counter()
        deadline = t_start
        while not st.quit:
            tick = time.perf_counter()
            if st.sim:
                with st.lock:
                    reset, st.start_requested = st.start_requested, False
                if reset:
                    robot.reset_scene() if seed is None else robot.reset_scene(seed)
                    q0, t_start = robot.read(), time.perf_counter()
                    cmd = q0
                    with st.lock:
                        st.q = st.cmd = q0
                        st.motors = robot.motor_telemetry()
                        st.watch.reset_motion()
                        st.sim_outcome = None
                        st.key_sample = (0.0, frozenset())
                    st.command({"action": "start"}, scene_ready=True)
                    tick = deadline = t_start
                if robot.ended:
                    # Save the terminal frame sequence once; the next Start resets the scene.
                    st.command({"action": "save"})
                    time.sleep(1 / hz)
                    continue
                if st.sim_step_recording:
                    with st.lock:
                        buf = st.buf
                        st.ready = True
                    if buf is not None:
                        q = robot.read()
                        elapsed = time.perf_counter() - t_start
                        row = recorder.frame(q, robot.motor_telemetry())
                        act = robot.read_leader(st.keys())
                        if robot.leader is not None and soft:
                            ramp = min(1.0, elapsed / soft)
                            act = [x + (a - x) * ramp for x, a in zip(q0, act, strict=True)]
                        cmd = robot.send(act, q)
                        with st.lock:
                            st.q, st.cmd = robot.read(), cmd
                            if buf is st.buf:
                                recorder.append_step(buf, row, cmd, elapsed, realtime=True)
                                st.sim_outcome = robot.outcome()
                                buf["simulation"] = {"source": st.sim_input, "seed": seed, **st.sim_outcome}
                    now = time.perf_counter()
                    deadline = max(deadline + 1 / hz, now)
                    time.sleep(max(0.0, deadline - now))
                    continue
            q = robot.read()
            motors = robot.motor_telemetry()
            now = time.time()
            trip_buffer = None
            with st.lock:
                was_tripped = st.watch.trip is not None
                # The leader can hold a fixed target between episodes or while recording.
                events = st.watch.update(time.perf_counter(), q, cmd, motors, holding=True, home=home)
                if events and st.buf is not None:
                    st.buf["watchdog_events"].extend(timed_events(events, now, st.t0))
                if st.watch.trip and not was_tripped:
                    trip_buffer = st.buf
                    if trip_buffer is None:
                        trip_buffer = st.new_buffer()
                        trip_buffer["watchdog_events"] = timed_events(events, now, now)
                    trip_buffer["motor_at_trip"] = motors
                    st.phase, st.buf = "stopped", None
                    st.msg = st.watch.blocked()
                # Don't accept Start until the first sample has gone through the watchdog.
                st.ready = st.watch.trip is None
            if st.watch.trip:
                try:
                    cmd = robot.hold(st.watch.pose, q) if not st.sim else list(st.watch.pose)
                finally:
                    if trip_buffer is not None:
                        print(st.msg["en"], flush=True)
                        diagnostic_writer = threading.Thread(target=st._save, args=(None, trip_buffer, st.instruction),
                                                             kwargs={"diagnostic": True}, daemon=True)
                        diagnostic_writer.start()
            else:
                act = robot.read_leader()
                r = min(1.0, (tick - t_start) / soft) if soft else 1.0
                act = [x * (1 - r) + a * r for x, a in zip(q0, act, strict=False)]
                cmd = robot.send(act, q)
            with st.lock:
                st.q, st.cmd, st.motors = q, cmd, motors
            time.sleep(max(0.0, 1 / hz - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        st.quit = True
        try:
            robot.shutdown(moved=started)
        finally:
            for c in st.cams.values():
                c.close()
            t = time.time()
            while st.saving and time.time() - t < 10:      # wait for background saves
                time.sleep(0.1)
            if diagnostic_writer is not None:
                diagnostic_writer.join()
            if serving:
                srv.shutdown()
            print("done", flush=True)
