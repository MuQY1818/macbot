"""Browse recorded rollouts in the standalone console."""
from __future__ import annotations

from .. import web
from ..config import Config
from ..system import expand
from .rollouts import Store, serve_rollouts


def make_handler(store: Store, task: str):
    class H(web.Handler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self.send(200, web.page("replay", task=task), "text/html; charset=utf-8")
            if self.static(path) or serve_rollouts(self, store, path):
                return
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if not serve_rollouts(self, store, self.path, method="POST"):
                self.send(404, b"not found", "text/plain")

    return H


def run(cfg: Config, args):
    folder = expand(args.dir) if args.dir else cfg.rollout_dir()
    if not folder.is_dir():
        raise SystemExit(f"{folder} does not exist")
    with web.bind_console(cfg, "replay", make_handler(Store(folder), cfg.task), args.port) as srv:
        print(f"rollouts: {folder}\nopen {srv.url}  (Ctrl-C to stop)", flush=True)
        if not getattr(args, "no_browser", True):
            web.show(srv)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
