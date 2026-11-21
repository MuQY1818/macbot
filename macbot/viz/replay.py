"""Browse recorded rollouts in the standalone console."""
from __future__ import annotations

from html import escape

from .. import web
from ..config import Config
from ..data.grasp import settings
from ..data.lerobot_v3 import recognize
from ..system import expand
from .datasets import DatasetStore, open_dataset
from .rollouts import Store, serve_rollouts


def make_handler(store: Store, task: str):
    readonly = getattr(store, "readonly", False)

    class H(web.Handler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self.send(200, web.page("replay", task=escape(task), source="dataset" if readonly else "rollouts"),
                                 "text/html; charset=utf-8")
            if self.static(path) or serve_rollouts(self, store, path):
                return
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if readonly:
                self.json({"error": "dataset_readonly"}, 405)
            elif not serve_rollouts(self, store, self.path, method="POST"):
                self.send(404, b"not found", "text/plain")

    return H


def run(cfg: Config, args):
    dataset = getattr(args, "dataset", None)
    if dataset is not None or recognize(args.dir):
        dataset = open_dataset(dataset or args.dir)
        store = DatasetStore(dataset)
        try:
            serve(cfg, args, store, dataset.label, f"LeRobot v3.0: {dataset.label}")
        finally:
            store.close()
        return
    folder = expand(args.dir) if args.dir else cfg.rollout_dir()
    if not folder.is_dir():
        raise SystemExit(f"{folder} does not exist")
    serve(cfg, args, Store(folder, grasp_cfg=settings(cfg)), cfg.task, f"rollouts: {folder}")


def serve(cfg, args, store, label, source):
    with web.bind_console(cfg, "replay", make_handler(store, label), args.port) as srv:
        print(f"{source}\nopen {srv.url}  (Ctrl-C to stop)", flush=True)
        if not getattr(args, "no_browser", True):
            web.show(srv)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
