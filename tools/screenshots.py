"""Render the web pages offline with mocked data and take screenshots with headless Chrome.

    python tools/screenshots.py --episode ~/macbot-data/raw/<task>/episode_000.npz --out docs/img

No server and no network are needed: the page is written to a temporary file, ``fetch`` and camera
image URLs are answered from mock data, and Chrome renders it at a fixed size. Use it to check the
look of the pages and to refresh the README screenshots.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from macbot import web  # noqa: E402
from macbot.robot.rebot import JOINTS  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CAPTURE = True


def frame_uri(d, cam, k):
    off = d[f"{cam}_offsets"]
    k = max(0, min(k, len(off) - 2))
    return "data:image/jpeg;base64," + base64.b64encode(d[f"{cam}_jpeg"][off[k]:off[k + 1]].tobytes()).decode()


def mock_script(routes: dict, images: dict, lang: str) -> str:
    return f"""<script>
try {{ localStorage.setItem("macbot.lang", {json.dumps(lang)}); }} catch (e) {{}}
const __R = {json.dumps(routes)}, __I = {json.dumps(images)}, nativeFetch = window.fetch.bind(window);
window.fetch = async (url, opt) => {{
  const p = String(url).split("?")[0];
  if (__I[p]) return nativeFetch(__I[p]);
  const key = Object.keys(__R).find(k => p === k) ?? (p === "/api/cmd" ? "/api/status" : null);
  return {{ ok: true, json: async () => JSON.parse(JSON.stringify(__R[key] ?? {{}})) }};
}};
const __src = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, "src");
Object.defineProperty(HTMLImageElement.prototype, "src", {{
  set(v) {{ const p = String(v).split("?")[0]; const hit = Object.keys(__I).find(k => p.startsWith(k));
           __src.set.call(this, hit ? __I[hit] : v); }},
  get() {{ return __src.get.call(this); }} }});
</script>"""


def render(name: str, routes: dict, images: dict, lang: str, out: Path, size=(1440, 1180), after: str = "",
           chrome_args=(), **vars):
    html = web.page(name, **vars).decode()
    html = html.replace('href="/static/', f'href="{web.STATIC.as_uri()}/').replace('src="/static/', f'src="{web.STATIC.as_uri()}/')
    html = html.replace('url("/static/', f'url("{web.STATIC.as_uri()}/')
    html = html.replace("<head>", "<head>" + mock_script(routes, images, lang), 1)
    if after:
        html = html.replace("</body>", f"<script>{after}</script></body>")
    css = (web.STATIC / "theme.css").read_text().replace('url("/static/', f'url("{web.STATIC.as_uri()}/')
    html = html.replace(f'<link rel="stylesheet" href="{web.STATIC.as_uri()}/theme.css">', f"<style>{css}</style>")
    if name == "init":
        css = (web.STATIC / "init.css").read_text()
        html = html.replace(f'<link rel="stylesheet" href="{web.STATIC.as_uri()}/init.css">', f"<style>{css}</style>")
    out.parent.mkdir(parents=True, exist_ok=True)
    page_file = out.with_suffix(".html")
    page_file.write_text(html)
    if not CAPTURE:
        print("wrote", page_file)
        return
    with tempfile.TemporaryDirectory(prefix="macbot_chrome_") as profile:
        subprocess.run([CHROME, "--headless=new", "--hide-scrollbars", "--allow-file-access-from-files",
                        "--disable-background-networking", "--host-resolver-rules=MAP * ~NOTFOUND",
                        f"--user-data-dir={profile}", "--no-first-run", *chrome_args,
                        f"--window-size={size[0]},{size[1]}", "--virtual-time-budget=2500", f"--screenshot={out}",
                        page_file.as_uri()], check=True, capture_output=True, timeout=25)
    print("wrote", out)


def collect_mock(d, meta):
    n = len(d["qpos"])
    k = n // 3
    eps = [{"idx": i, "dur": 30.1 + (i % 5) * 1.7, "frames": 903 + i * 11, "time": f"11-19 14:{10 + i:02d}",
            "instruction": meta["instruction"],
            "check": {"ok": True, "errors": [], "warnings": [] if i != 46 else [{"en": "wrist frame age p95 112 ms", "zh": "wrist 画面延迟 p95 112 ms"}]}}
           for i in range(49, 37, -1)]
    status = {"phase": "recording", "ready": True, "fake": False, "elapsed": 12.4, "frames": 372,
              "msg": {"en": "Recording. Enter saves, left arrow discards.", "zh": "录制中。回车保存，← 丢弃。"},
              "instruction": meta["instruction"], "task": meta["task"], "target": 60, "count": 50, "saving": 0,
              "q": d["qpos"][k].tolist(), "cmd": d["action"][k].tolist(), "names": JOINTS,
              "limits": {"shoulder_pan": [-145, 145], "shoulder_lift": [-170, 0], "elbow_flex": [-200, 0], "wrist_flex": [-90, 80],
                         "wrist_yaw": [-90, 90], "wrist_roll": [-90, 90], "gripper": [-270, 0]},
              "episodes": eps, "fps": 30, "cam_names": ["top", "wrist"],
              "cams": {"top": {"size": [492, 470], "crop": [625, 140, 1610, 1080]}, "wrist": {"size": [640, 360], "crop": None}},
              "out": "~/macbot-data/raw/" + meta["task"], "cam_age": {"top": 0.031, "wrist": 0.012}, "stall_s": 0.5}
    images = {"/stream/top": frame_uri(d, "top", k), "/stream/wrist": frame_uri(d, "wrist", k)}
    return {"/api/status": status}, images


LIMITS = {"shoulder_pan": [-145, 145], "shoulder_lift": [-170, 0], "elbow_flex": [-200, 0], "wrist_flex": [-90, 80],
          "wrist_yaw": [-90, 90], "wrist_roll": [-90, 90], "gripper": [-270, 0]}
POLICY = "pi05_cubes_to_box/004000"
EXECUTOR = "real-time chunking (guided)"


def run_mock(r, meta):
    k = len(r["t"]) // 2
    status = {"mode": "running", "msg": {"en": "Policy running. Space pauses.", "zh": "策略运行中。空格暂停。"},
              "q": r["qpos"][k].tolist(), "cmd": r["cmd"][k].tolist(), "names": JOINTS, "cam_names": ["top", "wrist"],
              "cam_age": {"top": 0.028, "wrist": 0.014}, "stall_s": 0.5, "cam_size": {"top": [492, 470], "wrist": [640, 360]},
              "chunks": int((r["chunk_t"] < r["t"][k]).sum()), "infer_ms": 284.0, "executor": EXECUTOR, "speed": 0.5,
              "fake": False, "policy": POLICY, "task": meta["task"], "rec_s": round(float(r["t"][k]), 1), "next": 3,
              "saved": [{"name": "rollout_000.npz", "result": "success", "s": 93.6},
                        {"name": "rollout_001.npz", "result": "success", "s": 115.4},
                        {"name": "rollout_002.npz", "result": "failure", "s": 61.2}]}
    idx = np.linspace(max(0, k - 450), k, 75).astype(int)        # last 15 s, as the page keeps it
    hist = {"q": r["qpos"][idx].round(2).tolist(), "c": r["cmd"][idx].round(2).tolist()}
    after = f"setTimeout(() => {{ HQ = {json.dumps(hist['q'])}; HC = {json.dumps(hist['c'])}; HQ.pop(); HC.pop(); }}, 50);"
    return {"/api/status": status}, {"/stream/top": frame_uri(r, "top", k), "/stream/wrist": frame_uri(r, "wrist", k)}, after


def replay_mock(r, meta, name):
    from macbot.viz.replay import summary
    s = summary(r)
    m = s["meta"]
    for key in list(m):
        if key not in ("task", "instruction", "fps", "frames", "joint_names", "cameras", "result", "speed", "saved_at"):
            m.pop(key)
    m.update(policy=POLICY, executor=EXECUTOR)
    k = int(len(s["t"]) * 0.42)
    items = [{"name": name, "result": "success", "s": round(s["t"][-1], 1)},
             {"name": "rollout_000", "result": "success", "s": 93.6}]
    routes = {"/api/rollouts": {"folder": "~/macbot-data/rollouts/" + meta["task"], "items": items},
              f"/api/rollout/{name}": s}
    images = {f"/api/rollout/{name}/top": frame_uri(r, "top", k), f"/api/rollout/{name}/wrist": frame_uri(r, "wrist", k)}
    return routes, images, k


def init_mock():
    """A fully synthetic inventory and two deterministic preview frames."""
    from dataclasses import asdict

    from macbot.cameras.discovery import Camera
    from macbot.init_wizard import ROLES
    from macbot.robot.usb import FAMILIES, Adapter

    devices = [Adapter(f"/dev/cu.demo-{role}", 0x1A86, 0x7523, f"DEMO-{role}", "USB servo board") for role in ("A", "B")]
    cams = [Camera("uvc", "USB Camera", "DEMO-UVC", (1280, 720, 30)),
            Camera("orbbec", "Orbbec Gemini 2", "DEMO-GEMINI", (1280, 720, 30))]
    inventory = {
        "revision": 1, "token": "demo", "families": FAMILIES, "roles": ROLES,
        "devices": [{**asdict(d), "adapter": d.hint[0], "suggestions": d.hint[1]} for d in devices],
        "cameras": [asdict(c) for c in cams], "notes": [], "exists": False,
        "remembered": {"family": "so101", "leader": 0, "follower": 1, "cameras": {"top": 0, "wrist": 1}},
    }
    images = {}
    for i in range(2):
        frame = np.full((180, 320, 3), (204, 208, 211) if i == 0 else (145, 151, 157), dtype=np.uint8)
        cv2.rectangle(frame, (51, 26), (270, 153), (156, 163, 168) if i == 0 else (99, 108, 118), -1)
        cv2.rectangle(frame, (62, 35), (258, 143), (182, 189, 194) if i == 0 else (126, 134, 144), -1)
        for x, y, color in [(100, 75, (52, 126, 225)), (173, 105, (132, 152, 63)), (217, 65, (184, 117, 62))]:
            cv2.rectangle(frame, (x, y), (x + 30, y + 26), color, -1)
        if i:
            cv2.line(frame, (12, 178), (104, 86), (36, 40, 46), 20, cv2.LINE_AA)
        _, jpeg = cv2.imencode(".jpg", frame)
        images[f"/api/preview/{i}"] = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    after = ('window.addEventListener("load", () => setTimeout(() => {'
             '$("next").click(); setTimeout(() => document.activeElement.blur(), 100); }, 150));')
    return {"/api/setup": inventory, "/api/calibration": {"files": []}}, images, after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", help="a recorded episode_XXX.npz (needed for collect, run and replay)")
    ap.add_argument("--rollout", help="a rollout_XXX.npz for the run and replay pages")
    ap.add_argument("--out", default=str(ROOT / "docs" / "img"))
    ap.add_argument("--pages", default="collect,run,replay")
    ap.add_argument("--html-only", action="store_true", help="only write the mocked pages; capture them with any browser")
    ap.add_argument("--mobile-dark", action="store_true", help="render setup at a narrow viewport with a dark theme")
    args = ap.parse_args()
    global CAPTURE
    CAPTURE = not args.html_only
    out = Path(args.out).expanduser()
    pages = args.pages.split(",")
    if any(p in {"collect", "run", "replay"} for p in pages):
        if not args.episode:
            ap.error("--episode is required for collect, run and replay")
        d = np.load(Path(args.episode).expanduser(), allow_pickle=True)
        meta = json.loads(str(d["meta"]))
    if args.rollout:
        r = np.load(Path(args.rollout).expanduser(), allow_pickle=True)
        rmeta = json.loads(str(r["meta"]))
    for lang in ("en", "zh"):
        if "init" in pages:
            routes, images, after = init_mock()
            mobile = args.mobile_dark
            render("init", routes, images, lang, out / f"init_{lang}{'_mobile_dark' if mobile else ''}.png", after=after,
                   size=(500, 1230) if mobile else (1140, 815),
                   chrome_args=("--force-dark-mode",) if mobile else ())
        if "collect" in pages:
            routes, images = collect_mock(d, meta)
            render("collect", routes, images, lang, out / f"collect_{lang}.png", task=meta["task"])
        if args.rollout and "run" in pages:
            routes, images, after = run_mock(r, rmeta)
            render("run", routes, images, lang, out / f"run_{lang}.png", after=after, task=rmeta["task"])
        if args.rollout and "replay" in pages:
            routes, images, _ = replay_mock(r, rmeta, Path(args.rollout).stem)
            render("replay", routes, images, lang, out / f"replay_{lang}.png", task=rmeta["task"], folder="")


if __name__ == "__main__":
    main()
