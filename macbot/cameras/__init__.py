"""Cameras for macOS.

* ``UVCCamera``: any UVC webcam, opened by its AVFoundation uniqueID through a tiny Swift helper
  (compiled on first use). Opening by name or index can grab an iPhone Continuity Camera.
  The helper crops, encodes JPEG and stamps the capture time.
* ``OrbbecColor``: color stream of an Orbbec Gemini 2 through ``pyorbbecsdk`` (needs sudo on macOS).
* ``FakeCamera``: moving test pattern for ``--fake`` runs.
* ``RobotCamera``: a rendered frame published by the simulation control loop.

All share one interface: ``latest() -> (BGR image, capture time)`` and ``close()``.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from ..system import give_back, user_home

SWIFT = Path(__file__).resolve().parent / "swift" / "uvc_stream.swift"


class _Latest:
    def __init__(self):
        self.frame, self.stamp, self.lock, self.running = None, 0.0, threading.Lock(), True

    def _put(self, img, stamp):
        with self.lock:
            self.frame, self.stamp = img, stamp

    def latest(self):
        with self.lock:
            return self.frame, self.stamp

    def age(self) -> float | None:
        img, t = self.latest()
        return None if img is None else time.time() - t

    def _wait_first(self, name, alive=lambda: True, timeout=5.0):
        t0 = time.time()
        while self.frame is None:
            if not alive() or time.time() - t0 > timeout:
                raise RuntimeError(f"{name}: no frame within {timeout:.0f} s")
            time.sleep(0.05)
        time.sleep(1.0)              # let auto exposure settle


def uvc_helper() -> Path:
    """Compile the Swift helper once per source version into ~/.cache/macbot/bin."""
    src = SWIFT.read_bytes()
    out = user_home() / ".cache" / "macbot" / "bin" / f"uvc_stream_{hashlib.sha1(src).hexdigest()[:10]}"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"compiling camera helper -> {out}", flush=True)
        subprocess.run(["swiftc", "-O", str(SWIFT), "-o", str(out)], check=True)
        for p in (out, out.parent, out.parent.parent):
            give_back(p)
    return out


def list_cameras() -> list[dict]:
    """Video devices known to macOS: name, uniqueID, model."""
    out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True,
                         timeout=15).stdout
    return [{"name": c.get("_name", ""), "unique_id": c.get("spcamera_unique-id", ""),
             "model": c.get("spcamera_model-id", "")} for c in json.loads(out or "{}").get("SPCameraDataType", [])]


def resolve_uid(unique_id: str) -> str:
    """A USB camera's uniqueID is location + vendor + product, so it changes with the USB port.
    If the configured ID is absent, match on the last 8 hex digits (vendor and product)."""
    try:
        ids = [c["unique_id"] for c in list_cameras()]
    except Exception:
        return unique_id
    if unique_id in ids:
        return unique_id
    tail = unique_id.lower()[-8:]
    hit = [i for i in ids if i.lower().startswith("0x") and i.lower().endswith(tail)]
    if len(hit) == 1:
        print(f"camera {unique_id} not found; using {hit[0]} (same vendor and product, different USB port)", flush=True)
        return hit[0]
    return unique_id


class UVCCamera(_Latest):
    def __init__(self, unique_id: str, crop=None, fps: int = 30, name: str = "uvc", *, mode=None, exact=False):
        super().__init__()
        crop = crop or [0, 0, 0, 0]
        uid = unique_id if exact else resolve_uid(unique_id)
        argv = [str(uvc_helper()), uid, *(str(int(v)) for v in crop), "0.92", str(fps),
                *(str(int(v)) for v in (mode or ()))]
        self.p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        try:
            self._wait_first(name, alive=lambda: self.p.poll() is None)
        except RuntimeError as e:
            err = self.p.stderr.read().decode(errors="replace").strip() if self.p.poll() is not None else ""
            self.close()
            raise RuntimeError(f"{e} {err}") from None

    def _read(self, n):
        b = b""
        while len(b) < n:
            c = self.p.stdout.read(n - len(b))
            if not c:
                raise EOFError
            b += c
        return b

    def _loop(self):
        try:
            while self.running:
                t, n = struct.unpack("<dI", self._read(12))
                img = cv2.imdecode(np.frombuffer(self._read(n), np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    self._put(img, t)
        except (EOFError, ValueError, OSError):
            pass

    def close(self):
        self.running = False
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.th.join(timeout=2)
        self.p.stdout.close()
        self.p.stderr.close()


def _orbbec_to_bgr(f):
    from pyorbbecsdk import OBFormat
    w, h, fmt = f.get_width(), f.get_height(), f.get_format()
    buf = np.asanyarray(f.get_data())
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return buf.reshape(h, w, 3)
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(buf.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    if fmt == OBFormat.UYVY:
        return cv2.cvtColor(buf.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    raise RuntimeError(f"unhandled color format {fmt}")


def orbbec_devices():
    from pyorbbecsdk import Context

    context = Context()
    devices = context.query_devices()
    return context, devices


def orbbec_serials(devices):
    return [devices.get_device_serial_number_by_index(i) for i in range(devices.get_count())]


def orbbec_profiles(pipe):
    from pyorbbecsdk import OBFormat, OBSensorType

    profiles = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    formats = {OBFormat.MJPG, OBFormat.RGB, OBFormat.BGR, OBFormat.YUYV, OBFormat.UYVY}
    for i in range(profiles.get_count()):
        profile = profiles.get_stream_profile_by_index(i).as_video_stream_profile()
        if profile.get_format() in formats:
            yield profile


class OrbbecColor(_Latest):
    def __init__(self, expect_size=(1280, 720), name: str = "orbbec", *, serial=None, fps=None):
        super().__init__()
        from pyorbbecsdk import Config, OBSensorType, Pipeline
        self.context = None
        if serial:
            self.context, devices = orbbec_devices()
            matches = [i for i, value in enumerate(orbbec_serials(devices)) if value == serial]
            if len(matches) != 1:
                raise RuntimeError(f"{name}: expected exactly one Orbbec with serial {serial}")
            self.pipe = Pipeline(devices.get_device_by_index(matches[0]))
        else:
            self.pipe = Pipeline()
        if serial and fps is not None:
            prof = next((p for p in orbbec_profiles(self.pipe)
                         if (p.get_width(), p.get_height(), p.get_fps()) == (*expect_size, fps)), None)
            if prof is None:
                raise RuntimeError(f"{name}: Orbbec color mode {expect_size} at {fps} fps is unavailable")
        else:
            prof = self.pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
        if (prof.get_width(), prof.get_height()) != tuple(expect_size):
            raise RuntimeError(f"{name}: color stream is {prof.get_width()}x{prof.get_height()}, config expects {expect_size}")
        cfg = Config()
        cfg.enable_stream(prof)
        self.pipe.start(cfg)
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        try:
            self._wait_first(name)
        except RuntimeError:
            self.close()
            raise

    def _loop(self):
        while self.running:
            fs = self.pipe.wait_for_frames(200)
            c = fs.get_color_frame() if fs is not None else None
            img = _orbbec_to_bgr(c) if c is not None else None
            if img is not None:
                self._put(img, time.time())

    def close(self):
        self.running = False
        self.th.join(timeout=1)
        self.pipe.stop()


class FakeCamera(_Latest):
    """30 fps test pattern: a dark tray and an orange ball moving left and right."""

    def __init__(self, kind="top", size=(985, 940)):
        super().__init__()
        self.kind, self.size = kind, tuple(size)
        threading.Thread(target=self._loop, daemon=True).start()
        self._wait_first(f"fake {kind}", timeout=2)

    def _loop(self):
        w, h = self.size
        while self.running:
            t = time.time()
            img = np.full((h, w, 3), (220, 244, 255) if self.kind == "top" else (76, 42, 43), np.uint8)
            cv2.rectangle(img, (w // 10, h // 6), (w * 9 // 10, h * 2 // 3), (46, 26, 27), -1)
            cv2.circle(img, (int(w / 2 + w / 4 * math.sin(t)), h // 2), h // 12, (26, 122, 255), -1)
            cv2.putText(img, f"fake {self.kind}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 201, 255), 3)
            self._put(img, t)
            time.sleep(1 / 30)

    def close(self):
        self.running = False


def _source_size(c: dict) -> tuple[int, int]:
    """Size of the frames a camera delivers: the crop box, or the full frame ([0, 0, 0, 0] = no crop)."""
    if crop := _crop_box(c):
        x0, y0, x1, y1 = crop
        return x1 - x0, y1 - y0
    if c["type"] == "robot":
        return tuple(c["save_size"])
    return tuple(c.get("expect_size") or ((1920, 1080) if c.get("type") == "uvc" else (1280, 720)))


def _crop_box(c):
    x0, y0, x1, y1 = c.get("crop") or [0, 0, 0, 0]
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


class RobotCamera:
    def __init__(self, robot, name, cfg):
        self.robot, self.key, self.crop = robot, cfg.get("key", name), _crop_box(cfg)
        shapes = getattr(robot, "camera_shapes", {})
        if self.key not in shapes:
            raise ValueError(f"camera {name}: robot observation lacks {self.key!r}; available: {', '.join(shapes)}")
        h, w, channels = shapes[self.key]
        if channels != 3:
            raise ValueError(f"camera {name}: expected an HWC RGB observation")
        if self.crop:
            x0, y0, x1, y1 = self.crop
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError(f"camera {name}: crop {self.crop} lies outside {w}x{h}")

    def latest(self):
        img, stamp = self.robot.camera_frame(self.key)
        if img is not None and self.crop:
            x0, y0, x1, y1 = self.crop
            img = img[y0:y1, x0:x1]
        return img, stamp

    def age(self):
        img, stamp = self.latest()
        return None if img is None else time.time() - stamp

    def close(self):
        pass


def open_cameras(cams_cfg: dict, fps: int, fake: bool, robot=None) -> dict:
    """Open every camera in the ``cameras`` config section. Closes what was opened if one fails."""
    out = {}
    if getattr(robot, "kind", None) == "sim" and (
            not cams_cfg or any(c["type"] != "robot" for c in cams_cfg.values())):
        raise ValueError("simulation requires cameras with type: robot")
    try:
        for name, c in cams_cfg.items():
            if fake:
                out[name] = FakeCamera(name, _source_size(c))
            elif c["type"] == "uvc":
                out[name] = UVCCamera(c["unique_id"], c.get("crop"), c.get("fps", fps), name,
                                      mode=c.get("expect_size"), exact=c.get("exact_id", False))
            elif c["type"] == "orbbec":
                out[name] = OrbbecColor(tuple(c.get("expect_size", (1280, 720))), name,
                                        serial=c.get("serial"), fps=c.get("fps"))
            elif c["type"] == "robot":
                out[name] = RobotCamera(robot, name, c)
            else:
                raise ValueError(f"camera {name}: unknown type {c['type']}")
    except Exception:
        for cam in out.values():
            cam.close()
        raise
    return out


def camera_info(cams_cfg: dict) -> dict:
    return {k: {"type": c["type"], "unique_id": c.get("unique_id"), "crop": _crop_box(c), "size": c["save_size"],
                **({"key": c.get("key", k)} if c["type"] == "robot" else {})}
            for k, c in cams_cfg.items()}
