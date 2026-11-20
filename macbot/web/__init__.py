"""Tiny HTTP layer shared by the collect, run and replay pages (standard library only)."""
from __future__ import annotations

import errno
import json
import mimetypes
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from ..config import CONSOLES, ConsolePortError

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
TEMPLATES = HERE / "templates"
BRAND = ('<a class="brand" href="/"><img class="logo-light" src="/static/logo.svg" alt="macbot">'
         '<img class="logo-dark" src="/static/logo-dark.svg" alt="macbot"></a>')
HOST = "127.0.0.1"
PORT_ATTEMPTS = 10
# Identifies this process to open console pages; a page that sees a new value reloads itself.
BOOT = secrets.token_hex(6)
# An open console page polls every 200 ms and retries while the console is down. Background tabs poll about once
# a second, so a page that has not reached a new console within this time is assumed to be closed.
PAGE_WAIT = 2.5


def open_browser(url: str) -> bool:
    """Open a console in the user's default browser, also when macbot runs under sudo.

    Set MACBOT_NO_BROWSER=1 to only print URLs (tests, scripts, remote sessions).
    """
    if os.environ.get("MACBOT_NO_BROWSER"):
        return False
    uid, user = os.environ.get("SUDO_UID"), os.environ.get("SUDO_USER")
    try:
        if sys.platform == "darwin" and os.geteuid() == 0 and uid and user:
            # A root process opens pages in no browser or in root's; hand the URL to the user who ran sudo.
            subprocess.Popen(["launchctl", "asuser", uid, "sudo", "-u", user, "/usr/bin/open", url],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        return webbrowser.open(url)
    except (OSError, webbrowser.Error):
        return False


def show(server: ConsoleServer, wait: float = PAGE_WAIT) -> threading.Thread:
    """Open the console unless a page left open from an earlier session reconnects (and reloads) first."""
    def run():
        if not server.page_seen.wait(wait):
            open_browser(server.url)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


class ConsoleServer(ThreadingHTTPServer):
    def __init__(self, *args, **kwargs):
        self.page_seen = threading.Event()
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        self.page_seen.set()
        super().process_request(request, client_address)

    @property
    def url(self):
        return f"http://{HOST}:{self.server_port}"

    @classmethod
    def from_listener(cls, fd, handler):
        """Adopt the wizard's already-listening socket after exec."""
        with socket.socket(fileno=fd) as listener:
            if listener.family != socket.AF_INET or listener.type != socket.SOCK_STREAM:
                raise ValueError("The collection listener must be an IPv4 TCP socket.")
            host, port = listener.getsockname()
            if host != HOST or not port:
                raise ValueError("The collection listener must be bound to IPv4 loopback.")
            server = cls((host, port), handler, bind_and_activate=False)
            server.socket.close()
            # Re-wrap the descriptor so the temporary socket does not own it.
            server.socket = socket.socket(fileno=listener.detach())
            server.socket.set_inheritable(False)
            server.server_name, server.server_port = server.server_address
            return server


def bind_console(cfg, section, handler, port=None):
    """Bind the actual listener in one step; retry only implicit ports in use."""
    explicit = port is not None or section in cfg.explicit_ports
    requested = cfg[section]["port"] if port is None else port
    command = "collect" if section == "teleop" else section
    hint = f"choose another with macbot {command} --port or change {section}.port in your config"
    if type(requested) is not int or not 0 <= requested <= 65535:
        raise ConsolePortError(f"Invalid port {requested!r}; {hint}.")
    last = requested if explicit else min(65535, requested + PORT_ATTEMPTS - 1)
    # Fall forward past the other consoles' defaults so one busy port does not push every console along.
    siblings = {cfg[other]["port"] for other in CONSOLES if other != section and other in cfg} - {requested}
    for candidate in range(requested, last + 1):
        if candidate in siblings:
            continue
        try:
            return ConsoleServer((HOST, candidate), handler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            if explicit:
                raise ConsolePortError(f"Port {requested} is already in use; {hint}.") from None
    raise ConsolePortError(f"Ports {requested}-{last} are already in use; {hint}.")


def page(name: str, **vars) -> bytes:
    """Read templates/<name>.html and fill the shared header and {{var}} placeholders."""
    html = (TEMPLATES / f"{name}.html").read_text(encoding="utf-8")
    head = (TEMPLATES / "_head.html").read_text(encoding="utf-8")
    html = html.replace("{{head}}", head).replace("{{brand}}", BRAND)
    for part in ("rollout_player", "rollout_details"):
        if "{{" + part + "}}" in html:
            html = html.replace("{{" + part + "}}", (TEMPLATES / f"_{part}.html").read_text(encoding="utf-8"))
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
        self.send_header("X-Macbot-Boot", BOOT)
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
        # Scripts and styles change with macbot; a cached copy from an older version breaks the new pages.
        self.send_header("Cache-Control", "no-cache" if f.suffix in {".js", ".css"} else "max-age=3600")
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
