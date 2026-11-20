"""Tiny HTTP layer shared by the collect, run and replay pages (standard library only)."""
from __future__ import annotations

import json
import mimetypes
import re
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
TEMPLATES = HERE / "templates"
BRAND = ('<a class="brand" href="/"><img class="logo-light" src="/static/logo.svg" alt="macbot">'
         '<img class="logo-dark" src="/static/logo-dark.svg" alt="macbot"></a>')


def page(name: str, **vars) -> bytes:
    """Read templates/<name>.html and fill the shared header and {{var}} placeholders."""
    html = (TEMPLATES / f"{name}.html").read_text(encoding="utf-8")
    head = (TEMPLATES / "_head.html").read_text(encoding="utf-8")
    html = html.replace("{{head}}", head).replace("{{brand}}", BRAND)
    for k, v in vars.items():
        html = html.replace("{{" + k + "}}", str(v))
    return html.encode()


class Handler(BaseHTTPRequestHandler):
    """Base handler: JSON, static files and MJPEG camera streams. Subclasses add routes."""

    def log_message(self, *a):
        pass

    def send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code: int = 200):
        self.send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def body_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def static(self, path: str) -> bool:
        m = re.fullmatch(r"/static/([\w./-]+)", path)
        if not m or ".." in m.group(1):
            return False
        f = STATIC / m.group(1)
        if not f.is_file():
            self.send(404, b"", "text/plain")
            return True
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        self.send_response(200)
        data = f.read_bytes()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(data)
        return True

    def mjpeg(self, cam, size, stop, fps: float = 15.0):
        """Stream the latest frame of ``cam`` (resized to ``size``) until ``stop()`` is true."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while not stop():
                img, _ = cam.latest()
                if img is not None:
                    _, b = cv2.imencode(".jpg", cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA),
                                        [cv2.IMWRITE_JPEG_QUALITY, 75])
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(b)).encode() + b"\r\n\r\n" + b.tobytes() + b"\r\n")
                time.sleep(1 / fps)
        except (BrokenPipeError, ConnectionResetError):
            pass
