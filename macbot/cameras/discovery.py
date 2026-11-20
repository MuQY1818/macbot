"""Camera inventory and preview adapter for the setup wizard."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from . import open_cameras, orbbec_devices, orbbec_profiles, orbbec_serials, uvc_helper


@dataclass(frozen=True)
class Camera:
    kind: str
    name: str
    uid: str
    mode: tuple[int, int, int] | None
    builtin: bool = False
    issue: str = ""

    def config(self):
        if not self.mode:
            raise ValueError(f"{self.name}: {self.issue or 'no supported video mode'}")
        w, h, fps = self.mode
        ratio = min(1, 640 / max(w, h))
        address = {"unique_id": self.uid, "exact_id": True} if self.kind == "uvc" else {"serial": self.uid}
        return {"type": self.kind, **address, "expect_size": [w, h], "fps": fps,
                "save_size": [max(2, round(w * ratio / 2) * 2), max(2, round(h * ratio / 2) * 2)]}


def best_mode(modes):
    # Aim for 720p/30; use a smaller or slower actual mode when the camera needs it.
    return min((tuple(m) for m in modes if all(x > 0 for x in m)), default=None, key=lambda m: (
        m[2] < 24, abs(min(m[2], 30) - 30), m[2] > 30, abs(m[0] * m[1] - 1280 * 720), m))


def uvc_inventory():
    raw = subprocess.check_output([str(uvc_helper()), "--list"], timeout=10, stderr=subprocess.PIPE)
    return json.loads(raw)


def orbbec_inventory():
    from pyorbbecsdk import Pipeline

    context, devices = orbbec_devices()
    cameras = []
    for i, serial in enumerate(orbbec_serials(devices)):
        try:
            dev = devices.get_device_by_index(i)
            name = dev.get_device_info().get_name()
            modes = [(p.get_width(), p.get_height(), p.get_fps()) for p in orbbec_profiles(Pipeline(dev))]
            cameras.append(Camera("orbbec", name, serial, best_mode(modes) if serial else None))
        except Exception as exc:
            cameras.append(Camera("orbbec", "Orbbec", serial, None, issue=str(exc)))
    del devices, context
    return cameras


def discover_cameras(uvc=uvc_inventory, orbbec=orbbec_inventory):
    cameras, notes = [], []
    try:
        for c in uvc():
            label = f"{c['name']} {c.get('model', '')}".lower()
            if not c.get("continuity") and "iphone" not in label and "continuity" not in label:
                cameras.append(Camera("uvc", c["name"], c["unique_id"], best_mode(c["modes"]), c.get("builtin", False)))
    except Exception as exc:
        notes.append(f"AVFoundation: {exc}. Install Xcode Command Line Tools (xcode-select --install).")
    try:
        cameras.extend(orbbec())
    except ImportError:
        notes.append("For Orbbec cameras, install the Orbbec SDK as described in the quickstart.")
    except Exception as exc:
        notes.append(f"Orbbec: {exc}. The macOS SDK may need sudo; see the setup guide.")
    if len({(c.kind, c.uid) for c in cameras}) != len(cameras):
        notes.append("Cameras with duplicate identifiers cannot be selected.")
        cameras = [c for c in cameras if sum((o.kind, o.uid) == (c.kind, c.uid) for o in cameras) == 1]
    return cameras, notes


def open_preview(camera):
    return open_cameras({"preview": camera.config()}, camera.mode[2], fake=False)["preview"]
