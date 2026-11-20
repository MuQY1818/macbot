"""Local setup console. Only an explicit action can request a read or hand off to collect."""
from __future__ import annotations

import json
import re
import secrets
import threading
import time
from dataclasses import asdict

import cv2

from .. import web
from ..config import load
from ..init_wizard import ROLES
from ..robot.usb import FAMILIES


class State:
    def __init__(self, setup):
        self.setup = setup
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(24)
        self.revision = 0
        self.previews = {}
        self.preview_errors = {}
        self.preview_keep = None
        self.result = None
        self.launch = threading.Event()
        self.collect_server = None

    def refresh(self):
        self.close_previews()
        self.result = None
        self.setup.refresh()
        self.revision += 1

    def inventory(self):
        source = self.setup.source_framing()
        matches = self.setup.camera_matches()
        return {
            "revision": self.revision, "token": self.token, "families": FAMILIES,
            "roles": list(dict.fromkeys([*ROLES, *(source or {})])),
            "devices": [{**asdict(d), "adapter": d.hint[0], "suggestions": d.hint[1]} for d in self.setup.devices],
            "cameras": [asdict(c) for c in self.setup.cameras], "notes": self.setup.notes,
            "remembered": self.setup.remembered(), "exists": self.setup.path.is_file(),
            "source_framing": source, "camera_matches": matches,
        }

    def photo(self, index, keep=True):
        if self.launch.is_set() or not 0 <= index < len(self.setup.cameras):
            raise ValueError("Camera not available.")
        if self.preview_keep != keep:
            self.close_previews()
            self.preview_keep = keep
        if index in self.preview_errors:
            raise ValueError(self.preview_errors[index])
        if index not in self.previews:
            try:
                self.previews[index] = self.setup.preview(*self.setup.preview_settings(index, keep))
            except Exception as exc:
                self.preview_errors[index] = str(exc)
                raise
        image, stamp = self.previews[index].latest()
        if image is None or time.time() - stamp > 2:
            raise ValueError("No recent camera frame. Check the cable and rescan.")
        h, w = image.shape[:2]
        small = cv2.resize(image, (320, max(1, round(320 * h / w))), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if not ok:
            raise ValueError("Cannot encode the preview.")
        return jpeg.tobytes()

    def close_previews(self):
        for cam in self.previews.values():
            try:
                cam.close()
            except Exception as exc:
                print(f"Preview close: {exc}")
        self.previews.clear()
        self.preview_errors.clear()

    def action(self, path, body):
        if self.launch.is_set():
            raise ValueError("Collection is starting.")
        if path == "/api/refresh":
            self.refresh()
            return self.inventory()
        if body.get("revision") != self.revision:
            raise ValueError("The device list changed. Reload the setup page.")
        if path == "/api/move":
            return {"leader": self.setup.move(body)}
        if path == "/api/calibration":
            return {"files": self.setup.calibration(self.setup.robot_config(body))}
        if path == "/api/save":
            self.result = None
            profile = self.setup.profile(body)
            code, text = self.setup.write(profile, overwrite=body.get("overwrite") is True)
            self.result = self.setup.path.read_bytes() if code == 0 else None
            return {"ok": code == 0, "output": text, "path": str(self.setup.path)}
        if path == "/api/collect":
            if self.result is None or self.setup.path.read_bytes() != self.result:
                raise ValueError("Save the configuration and pass doctor before starting collection.")
            self.collect_server = web.bind_console(load(self.setup.path), "teleop", web.Handler)
            return {"port": self.collect_server.server_port}
        raise ValueError("Unknown action.")


def make_handler(state, host):
    class H(web.Handler):
        def do_GET(self):
            if self.headers.get("Host") != host:
                return self.json({"error": "Open the local setup URL printed in the terminal."}, 403)
            if self.path == "/":
                return self.send(200, web.page("init"), "text/html; charset=utf-8")
            if self.static(self.path):
                return
            with state.lock:
                if self.path == "/api/setup":
                    return self.json(state.inventory())
                if match := re.fullmatch(r"/api/preview/(\d+)\?revision=(\d+)(?:&keep=([01]))?", self.path):
                    if int(match[2]) != state.revision:
                        return self.json({"error": "Camera inventory changed."}, 409)
                    try:
                        photo = state.photo(int(match[1]), keep=match[3] != "0")
                    except Exception as exc:
                        return self.json({"error": str(exc)}, 422)
                    return self.send(200, photo, "image/jpeg")
            self.json({"error": "Not found."}, 404)

        def do_POST(self):
            # Prevent cross-site form submissions and DNS rebinding on the loopback service.
            if self.headers.get("Host") != host or self.headers.get("X-Macbot-Setup") != state.token:
                return self.json({"error": "Open the local setup page again."}, 403)
            try:
                n = int(self.headers.get("Content-Length", 0))
                if not 0 < n <= 16_384:
                    raise ValueError("Invalid request size.")
                body = json.loads(self.rfile.read(n))
                if not isinstance(body, dict):
                    raise ValueError("Expected setup selections.")
                with state.lock:
                    result = state.action(self.path, body)
                    try:
                        self.json(result)
                    finally:
                        if self.path == "/api/collect":
                            state.launch.set()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (Exception, SystemExit) as exc:
                self.json({"error": str(exc)}, 422)

    return H


def serve(setup, port=None, open_browser=True):
    state = State(setup)
    cfg = load(setup.path if setup.path.is_file() else None)
    with web.bind_console(cfg, "init", web.Handler, port) as server:
        state.refresh()
        host = f"{web.HOST}:{server.server_port}"
        server.RequestHandlerClass = make_handler(state, host)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = server.url
        print(f"Setup: {url}", flush=True)
        try:
            if open_browser:
                web.open_browser(url)
            state.launch.wait()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
            thread.join()
            with state.lock:
                state.close_previews()
    return state.collect_server
