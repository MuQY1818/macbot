"""Browse recorded rollouts (``rollout_XXX.npz`` from ``macbot run``) in the browser.

Frames are served straight from the npz files; nothing is converted. For each rollout the page shows
all cameras in sync, the command and measured joint traces with a tick at every new action chunk, and
the run settings (policy, executor, speed, inference time).
"""
from __future__ import annotations

import json
import re
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .. import web
from ..config import Config
from ..system import expand


class Store:
    """Keeps the most recently opened rollout in memory, so scrubbing does not re-read the file."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.lock = threading.Lock()
        self.cur = (None, None)

    def files(self):
        return sorted(self.folder.glob("rollout_*.npz"))

    def index(self):
        out = []
        for p in self.files():
            try:
                with np.load(p) as d:
                    m = json.loads(str(d["meta"]))
            except Exception:
                continue
            out.append({"name": p.stem, "result": m.get("result"), "s": round(m.get("frames", 0) / m.get("fps", 30), 1),
                        "saved_at": m.get("saved_at", ""), "mode": m.get("mode", "")})
        return out[::-1]

    def get(self, name: str):
        with self.lock:
            if self.cur[0] != name:
                p = self.folder / f"{name}.npz"
                if not re.fullmatch(r"rollout_\d+", name) or not p.exists():
                    return None
                with np.load(p) as d:
                    self.cur = (name, {k: d[k] for k in d.files})
            return self.cur[1]


def summary(d) -> dict:
    meta = json.loads(str(d["meta"]))
    cams = meta.get("cameras") or [k[:-5] for k in d if k.endswith("_jpeg")]
    sizes = {}
    import cv2
    for c in cams:
        off = d[f"{c}_offsets"]
        if len(off) > 1:
            im = cv2.imdecode(d[f"{c}_jpeg"][off[0]:off[1]], cv2.IMREAD_COLOR)
            sizes[c] = [int(im.shape[1]), int(im.shape[0])]
    t = d["t"].astype(float)
    ct = d["chunk_t"].astype(float) if "chunk_t" in d else np.zeros(0)
    ims = d["infer_ms"].astype(float) if "infer_ms" in d else np.zeros(0)
    return {"meta": meta, "cams": cams, "sizes": sizes, "t": np.round(t, 3).tolist(),
            "q": np.round(d["qpos"].astype(float), 2).tolist(), "cmd": np.round(d["cmd"].astype(float), 2).tolist(),
            "chunk_t": np.round(ct, 3).tolist(), "infer_ms": round(float(np.median(ims)), 1) if len(ims) else None,
            "chunks": int(len(ct))}


def make_handler(store: Store, task: str):
    class H(web.Handler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self.send(200, web.page("replay", task=task, folder=str(store.folder)), "text/html; charset=utf-8")
            if self.static(path):
                return
            if path == "/api/rollouts":
                return self.json({"folder": str(store.folder), "items": store.index()})
            m = re.fullmatch(r"/api/rollout/(\w+)", path)
            if m:
                d = store.get(m.group(1))
                return self.json(summary(d)) if d is not None else self.json({}, 404)
            m = re.fullmatch(r"/api/rollout/(\w+)/(\w+)/(\d+)", path)
            if m:
                d = store.get(m.group(1))
                key = f"{m.group(2)}_offsets"
                if d is None or key not in d:
                    return self.send(404, b"", "text/plain")
                off, k = d[key], int(m.group(3))
                k = max(0, min(k, len(off) - 2))
                return self.send(200, d[f"{m.group(2)}_jpeg"][off[k]:off[k + 1]].tobytes(), "image/jpeg")
            self.send(404, b"not found", "text/plain")

    return H


def run(cfg: Config, args):
    folder = expand(args.dir) if args.dir else cfg.rollout_dir()
    if not folder.is_dir():
        raise SystemExit(f"{folder} does not exist")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(Store(folder), cfg.task))
    srv.daemon_threads = True
    print(f"rollouts: {folder}\nopen http://127.0.0.1:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
