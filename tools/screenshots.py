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
const __R = {json.dumps(routes)}, __I = {json.dumps(images)};
window.fetch = async (url, opt) => {{
  const p = String(url).split("?")[0];
  const key = Object.keys(__R).find(k => p === k) ?? (p === "/api/cmd" ? "/api/status" : null);
  return {{ ok: true, json: async () => JSON.parse(JSON.stringify(__R[key] ?? {{}})) }};
}};
const __src = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, "src");
Object.defineProperty(HTMLImageElement.prototype, "src", {{
  set(v) {{ const p = String(v).split("?")[0]; const hit = Object.keys(__I).find(k => p.startsWith(k));
           __src.set.call(this, hit ? __I[hit] : v); }},
  get() {{ return __src.get.call(this); }} }});
</script>"""


def render(name: str, routes: dict, images: dict, lang: str, out: Path, size=(1440, 1180), after: str = "", **vars):
    html = web.page(name, **vars).decode()
    html = html.replace('href="/static/', f'href="{web.STATIC.as_uri()}/').replace('src="/static/', f'src="{web.STATIC.as_uri()}/')
    html = html.replace('url("/static/', f'url("{web.STATIC.as_uri()}/')
    html = html.replace("<head>", "<head>" + mock_script(routes, images, lang), 1)
    if after:
        html = html.replace("</body>", f"<script>{after}</script></body>")
    css = (web.STATIC / "theme.css").read_text().replace('url("/static/', f'url("{web.STATIC.as_uri()}/')
    html = html.replace(f'<link rel="stylesheet" href="{web.STATIC.as_uri()}/theme.css">', f"<style>{css}</style>")
    out.parent.mkdir(parents=True, exist_ok=True)
    page_file = out.with_suffix(".html")
    page_file.write_text(html)
    if not CAPTURE:
        print("wrote", page_file)
        return
    subprocess.run([CHROME, "--headless=new", "--hide-scrollbars", "--allow-file-access-from-files",
                    f"--user-data-dir={tempfile.mkdtemp(prefix='macbot_chrome_')}", "--no-first-run",
                    f"--window-size={size[0]},{size[1]}", "--virtual-time-budget=2500", f"--screenshot={out}",
                    page_file.as_uri()], check=True, capture_output=True)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, help="a recorded episode_XXX.npz (frames and joints for the mock)")
    ap.add_argument("--rollout", help="a rollout_XXX.npz for the run and replay pages")
    ap.add_argument("--out", default=str(ROOT / "docs" / "img"))
    ap.add_argument("--pages", default="collect,run,replay")
    ap.add_argument("--html-only", action="store_true", help="only write the mocked pages; capture them with any browser")
    args = ap.parse_args()
    global CAPTURE
    CAPTURE = not args.html_only
    out = Path(args.out).expanduser()
    d = np.load(Path(args.episode).expanduser(), allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    pages = args.pages.split(",")
    if args.rollout:
        r = np.load(Path(args.rollout).expanduser(), allow_pickle=True)
        rmeta = json.loads(str(r["meta"]))
    for lang in ("en", "zh"):
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
