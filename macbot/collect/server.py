"""Leader-follower teleoperation with recording, driven from a local web page.

Workflow (same as ``lerobot-record``): record one episode, reset the scene, record the next.
Keys on the page: S start, Enter or right arrow save, left arrow discard. Every saved episode can
be replayed (all cameras plus joint traces) or deleted (moved to ``deleted/``, numbers are not reused).
Each save runs quality checks (frame age, repeated frames, joint jumps, late samples).

Episode file ``<data_root>/raw/<task>/episode_XXX.npz``:
  qpos (N,7)      follower measured joints, degrees
  action (N,7)    command sent to the follower (gripper: leader value before the force clamp)
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
from http.server import ThreadingHTTPServer

import cv2
import numpy as np

from .. import web
from ..cameras import camera_info, open_cameras
from ..config import ARM, JOINTS, Config
from ..system import ensure_dir, give_back, start_log
from ..i18n import text, tr
from .checks import check_episode

CAM_STALL_S = 0.5      # a camera without a new frame for this long counts as disconnected


class State:
    def __init__(self, cfg: Config, out, fake: bool):
        self.cfg, self.out, self.fake = cfg, out, fake
        self.cam_names = list(cfg["cameras"])
        self.fps = int(cfg["record"]["fps"])
        self.lock = threading.Lock()
        self.phase = "idle"                 # idle | recording
        self.t0 = 0.0
        self.instruction = cfg["task"]["instruction"]
        self.msg = tr("Put both arms at the zero pose. Teleoperation starts after the alignment check.",
                      "两臂摆到零位后启动；对齐检查通过才开始遥操作。")
        self.q, self.cmd = [0.0] * 7, [0.0] * 7
        self.limits, self.ready, self.quit = {}, False, False
        self.buf = None
        self.saving = 0
        self.ff = {}
        self.cam_info = camera_info(cfg["cameras"])
        self.summaries = {}                 # file name -> (mtime, summary)
        self.cams = {}
        self.refresh()

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
            return {"phase": self.phase, "ready": self.ready, "fake": self.fake,
                    "elapsed": time.time() - self.t0 if rec else 0.0,
                    "frames": len(self.buf["qpos"]) if (rec and self.buf) else 0,
                    "msg": self.msg, "instruction": self.instruction, "task": self.cfg.task,
                    "target": self.cfg["task"]["target_episodes"], "count": len(eps), "saving": self.saving,
                    "q": self.q, "cmd": self.cmd, "names": JOINTS, "limits": self.limits, "episodes": eps,
                    "fps": self.fps, "cams": self.cam_info, "cam_names": self.cam_names, "out": str(self.out),
                    "cam_age": ages, "stall_s": CAM_STALL_S}

    def command(self, c):
        a = c.get("action")
        with self.lock:
            if a == "instruction" and self.phase == "idle":
                text = str(c.get("text", "")).strip()
                if text:
                    self.instruction = text
                    self.msg = tr(f"Instruction set: {text}", f"指令改为：{text}")
            elif a == "start" and self.phase == "idle":
                if not self.ready:
                    self.msg = tr("Teleoperation has not started yet.", "遥操作还没启动。")
                    return
                bad = self.stalled()
                if bad:
                    self.msg = tr(f"No frames from {', '.join(bad)} (cable loose?). Fix it and restart.",
                                   f"{'、'.join(bad)} 相机没有画面（线松了？），插好后重启采集程序。")
                    return
                keys = ["qpos", "action", "wall", "late"] + [f"{k}_{s}" for k in self.cam_names for s in ("jpg", "age", "repeat")]
                self.buf = {k: [] for k in keys}
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
                self.msg = tr("Shutting down: the follower returns to zero slowly, then disables.", "正在结束：从臂慢速回零后失能。")

    def _save(self, idx, buf, instruction):
        try:
            arr = {k: np.asarray(v) for k, v in buf.items() if not k.endswith("_jpg")}
            rc = self.cfg["record"]
            chk = check_episode(arr, self.cam_names, self.fps, self.cfg["checks"], rc["min_episode_s"])
            n = len(buf["qpos"])
            meta = {"task": self.cfg.task, "instruction": instruction, "fps": self.fps, "frames": n,
                    "duration_s": round(n / self.fps, 2), "joint_names": JOINTS, "units": "deg",
                    "state": "follower measured position", "action": "command sent to follower (gripper: leader value)",
                    "cameras": self.cam_info, "follower_ff": self.ff, "fake": self.fake,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "check": chk, "writer": "macbot"}
            extra = {}
            for k in self.cam_names:
                extra[f"{k}_jpeg"] = np.concatenate(buf[f"{k}_jpg"])
                extra[f"{k}_offsets"] = np.cumsum([0] + [len(b) for b in buf[f"{k}_jpg"]]).astype(np.int64)
                extra[f"{k}_age"] = arr[f"{k}_age"].astype(np.float32)
                extra[f"{k}_repeat"] = arr[f"{k}_repeat"].astype(bool)
            path = self.out / f"episode_{idx:03d}.npz"
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, qpos=arr["qpos"].astype(np.float32), action=arr["action"].astype(np.float32),
                     timestamp=np.arange(n) / self.fps, wall_time=arr["wall"],
                     meta=np.array(json.dumps(meta, ensure_ascii=False)), **extra)
            tmp.rename(path)
            give_back(path)
            items = chk["errors"] + chk["warnings"]
            en, zh = "; ".join(text(x, "en") for x in items), "；".join(text(x, "zh") for x in items)
            with self.lock:
                self.saving -= 1
                self.refresh()
                k = len(self.summaries)
                self.msg = tr(f"Saved episode {idx:03d} ({n / self.fps:.1f} s)" + (f", check: {en}" if en else ", checks passed")
                              + f". {k} in total. Reset the scene and press S.",
                              f"已保存第 {idx:03d} 条（{n / self.fps:.1f} 秒）" + (f"，注意：{zh}" if zh else "，检查通过")
                              + f"。共 {k} 条。复位场景后按 S 录下一条。")
        except Exception as e:
            with self.lock:
                self.saving -= 1
                self.msg = tr(f"Saving episode {idx:03d} failed: {e}", f"保存第 {idx:03d} 条失败：{e}")


class Recorder:
    """Samples joints and cameras on a fixed 1/fps grid while recording."""

    def __init__(self, st: State):
        self.st = st
        self.size = {k: tuple(st.cfg["cameras"][k]["save_size"]) for k in st.cam_names}
        self.q = int(st.cfg["record"]["jpeg_quality"])
        threading.Thread(target=self._loop, daemon=True).start()

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
                    q, cmd = list(st.q), list(st.cmd)
                now = time.time()
                row = {"qpos": q, "action": cmd, "wall": now - t0, "late": (now - target) > dt}
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
                    _, jpg = cv2.imencode(".jpg", cv2.resize(img, self.size[c], interpolation=cv2.INTER_AREA),
                                          [cv2.IMWRITE_JPEG_QUALITY, self.q])
                    row[f"{c}_jpg"], row[f"{c}_age"], row[f"{c}_repeat"] = jpg.reshape(-1), now - stamp, stamp == last[c]
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
            if self.path != "/api/cmd":
                return self.send(404, b"", "text/plain")
            try:
                st.command(self.body_json())
            except Exception as e:
                with st.lock:
                    st.msg = tr(f"Command failed: {e}", f"操作失败：{e}")
            self.json(st.status())

    return H


def run(cfg: Config, fake: bool = False, port: int | None = None, out=None):
    start_log(cfg.log_dir, "collect")
    out = ensure_dir(out or (cfg.raw_dir() if not fake else cfg.raw_dir(cfg.task + "_fake")))
    st = State(cfg, out, fake)
    T = cfg["teleop"]
    port = port or T["port"]
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(st))   # bind first: fails fast if the port is taken
    srv.daemon_threads = True
    st.cams = open_cameras(cfg["cameras"], st.fps, fake)
    Recorder(st)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"web page: http://127.0.0.1:{port}   data: {out}", flush=True)

    from ..robot.rebot import clip_gripper
    rcfg = cfg["robot"]
    if fake:
        from ..robot import FakeFollower, FakeLeader
        robot, leader = FakeFollower(), FakeLeader()
    else:
        from ..robot.rebot import make_follower, make_leader
        robot, leader = make_follower(rcfg), make_leader(leader_id=rcfg.get("leader_id", "rebot_arm_102_leader"))
    st.ff = dict(robot.feedforward, gripper_mode=rcfg.get("gripper_mode"), gripper_lead_deg=rcfg["gripper_lead_deg"])
    leader.connect(calibrate=False)
    robot.connect(calibrate=False)
    lim = robot.config.joint_limits
    started = False
    try:
        a0, o0 = leader.get_action(), robot.get_observation()
        bad = [f"{n} leader {a0[f'{n}.pos']:.0f} / follower {o0[f'{n}.pos']:.0f}" for n in ARM
               if abs(min(max(a0[f"{n}.pos"], lim[n][0]), lim[n][1]) - o0[f"{n}.pos"]) > T["align_max_deg"]]
        if bad and not fake:
            with st.lock:
                st.msg = tr(f"Arms not aligned (more than {T['align_max_deg']} deg): " + "; ".join(bad) + ". Align and restart.",
                            f"两臂没对齐（差超过 {T['align_max_deg']}°）：" + "；".join(bad) + "。对齐后重启。")
            print(st.msg["en"], flush=True)
            time.sleep(3)                   # let the page show the message
            raise SystemExit(2)
        with st.lock:
            st.limits = {n: list(lim[n]) for n in JOINTS}
            st.ready = True
            st.msg = tr(f"Teleoperation running. Set up the scene and press S. {len(st.summaries)} episodes so far.",
                        f"遥操作已启动。摆好场景后按 S 开始录制。已有 {len(st.summaries)} 条。")
        started = True
        print("teleoperation running", flush=True)
        hz, soft, lead = T["hz"], T["soft_start_s"], float(rcfg["gripper_lead_deg"])
        t_start = time.perf_counter()
        while not st.quit:
            tick = time.perf_counter()
            act = leader.get_action()
            r = min(1.0, (tick - t_start) / soft)              # soft start from the follower's pose
            act = {key: o0.get(key, v) * (1 - r) + v * r for key, v in act.items()}
            obs = robot.get_observation()
            sent = robot.send_action(clip_gripper(act, float(obs["gripper.pos"]), lead))
            q = [float(obs[f"{n}.pos"]) for n in JOINTS]
            cmd = [float(sent.get(f"{n}.pos", act[f"{n}.pos"])) for n in ARM] + [float(act["gripper.pos"])]
            with st.lock:
                st.q, st.cmd = q, cmd
            time.sleep(max(0.0, 1 / hz - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        st.quit = True
        try:
            if not fake:
                from ..robot.rebot import safe_disconnect
                try:
                    leader.disconnect()
                finally:
                    safe_disconnect(robot, go_zero=started)
        finally:
            for c in st.cams.values():
                c.close()
            t = time.time()
            while st.saving and time.time() - t < 10:      # wait for background saves
                time.sleep(0.1)
            srv.shutdown()
            print("done", flush=True)
