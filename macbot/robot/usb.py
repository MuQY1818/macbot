"""USB adapter hints and port resolution. Enumeration never opens a motor bus."""
from __future__ import annotations

import hashlib
import json
import plistlib
import subprocess
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import monotonic, sleep

# Adapter IDs are hints about the transport, not proof of the attached arm model.
ADAPTERS = {
    (0x1A86, 0x7523): ("WCH CH340", ("so100", "so101")),
    (0x1A86, 0x7522): ("WCH CH340K", ("so100", "so101")),
    (0x1A86, 0x55D3): ("WCH CH34x", ("so100", "so101")),
    (0x0403, 0x6001): ("FTDI USB serial (e.g. U2D2)", ("koch", "omx")),
    (0x0C72, 0x000C): ("PEAK PCAN-USB", ("rebot_b601",)),
}
FAMILIES = {"so100": "SO-100", "so101": "SO-101", "koch": "Koch", "omx": "OMX", "rebot_b601": "reBot B601-RS"}
PCAN = (0x0C72, 0x000C)
VIDEO_CLASS = 14  # USB interface class for video (UVC)


@dataclass(frozen=True)
class Adapter:
    port: str | None
    vid: int | None
    pid: int | None
    serial: str | None
    name: str

    @property
    def hint(self):
        return ADAPTERS.get((self.vid, self.pid), ("Unknown USB adapter", ()))

    @property
    def identity(self):
        if self.serial and self.vid is not None and self.pid is not None:
            return {"vid": self.vid, "pid": self.pid, "serial": self.serial}
        return None


def _usb_nodes(nodes):
    # `ioreg -a` returns the root entry as a dict; tests and older callers pass a list.
    if isinstance(nodes, dict):
        nodes = [nodes]
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if "idVendor" in node and "idProduct" in node:
            yield node
        yield from _usb_nodes(node.get("IORegistryEntryChildren", []))


def _video_ids():
    """USB IDs of devices with a video interface. Some UVC cameras also expose a CDC serial port."""
    xml = subprocess.check_output(["ioreg", "-a", "-r", "-c", "IOUSBHostInterface", "-l"], timeout=5)
    found = plistlib.loads(xml) if xml.strip() else []
    return {(i.get("idVendor"), i.get("idProduct")) for i in (found if isinstance(found, list) else [found])
            if isinstance(i, dict) and i.get("bInterfaceClass") == VIDEO_CLASS}


def scan_adapters(ports=None, usb_nodes=None, video_ids=None) -> list[Adapter]:
    """On macOS, PCAN does not expose a serial port; enumerate it separately."""
    if ports is None:
        from serial.tools.list_ports import comports
        ports = comports()
    ports = list(ports)
    paths = {p.device for p in ports}
    if usb_nodes is None:
        usb_nodes = []
        if sys.platform == "darwin":
            xml = subprocess.check_output(["ioreg", "-a", "-p", "IOUSB", "-l"], timeout=5)
            usb_nodes = plistlib.loads(xml)
    if video_ids is None:
        video_ids = _video_ids() if sys.platform == "darwin" else set()
    devices = []
    for p in sorted(ports, key=lambda p: p.device):
        if p.vid is None or p.pid is None or (p.vid, p.pid) == PCAN:
            continue
        # A camera's serial port is never an arm, and sampling it would send motor packets to the camera.
        if (p.vid, p.pid) in video_ids and (p.vid, p.pid) not in ADAPTERS:
            continue
        # The /dev/tty.* node is an alias for the same hardware as /dev/cu.*.
        if p.device.startswith("/dev/tty.") and "/dev/cu." + p.device.removeprefix("/dev/tty.") in paths:
            continue
        devices.append(Adapter(p.device, p.vid, p.pid, p.serial_number or None,
                               p.product or p.description or p.device))
    for node in _usb_nodes(usb_nodes):
        vid, pid = node.get("idVendor"), node.get("idProduct")
        if (vid, pid) == PCAN:
            devices.append(Adapter(None, vid, pid, node.get("USB Serial Number"), "PEAK PCAN-USB"))
    return devices


def resolve_adapter(identity: dict, devices: Iterable[Adapter], *, serial_port=True) -> Adapter:
    matches = [d for d in devices if bool(d.port) == serial_port and (
        d.identity == identity if identity.get("serial") else
        (d.vid, d.pid, d.port) == (identity["vid"], identity["pid"], identity.get("port"))
    )]
    if len(matches) != 1:
        raise ValueError(f"Saved USB adapter is {'missing' if not matches else 'ambiguous'}; reconnect it or run macbot init.")
    return matches[0]


def resolve_robot(robot: dict, scan=scan_adapters, *, leader=True) -> dict:
    """Refresh saved port paths. Never substitute a different serial-numbered adapter."""
    if not (bindings := robot.get("usb")):
        return robot
    devices = scan()
    out = dict(robot)
    for role, identity in bindings.items():
        if role == "leader" and not leader:
            continue
        if role == "can":
            resolve_adapter(identity, devices, serial_port=False)
            if sum(d.port is None for d in devices) != 1:
                raise ValueError("reBot setup needs exactly one PCAN-USB adapter for can0.")
        else:
            adapter = resolve_adapter(identity, devices)
            if robot["type"] == "rebot_b601" and role == "leader":
                out["leader_port"] = adapter.port
            else:
                out[role] = {**out[role], "port": adapter.port}
    return out


def calibration_id(binding: dict, role: str) -> str:
    key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()[:12]
    return f"macbot-{role}-{key}"


def sample_positions(port: str, family: str, seconds=3.0) -> float:
    """Read only Present_Position; bypass device connect, handshake and torque handling."""
    from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

    model = "sts3215" if family in {"so100", "so101"} else "xl330-m288"
    ids = range(1, 7) if family != "omx" else (*range(1, 7), *range(11, 17))
    motors = {str(i): Motor(i, model, MotorNormMode.DEGREES) for i in ids}
    bus = SerialMotorsBus(port, motors)

    def read():
        values = {}
        for key in motors:
            try:
                values[key] = bus.read("Present_Position", key, normalize=False)
            except ConnectionError:
                if family != "omx":
                    raise
                # OMX leader and follower have disjoint ID ranges.
        return values

    bus.connect(handshake=False)
    try:
        return position_span(read, seconds)
    finally:
        bus.disconnect(disable_torque=False)


def position_span(read: Callable[[], dict], seconds) -> float:
    """Largest excursion of any joint, accounting for the 4096-count encoder wrap."""
    samples, offsets, spans = {}, {}, {}
    deadline = monotonic() + seconds
    while monotonic() < deadline:
        for key, raw in read().items():
            previous = samples.get(key, raw)
            value = offsets.get(key, 0) + (raw - previous + 2048) % 4096 - 2048
            samples[key], offsets[key] = raw, value
            lo, hi = spans.get(key, (0, 0))
            spans[key] = min(lo, value), max(hi, value)
        sleep(0.04)
    if not samples:
        raise ConnectionError("No position replies.")
    return max(hi - lo for lo, hi in spans.values())


def detect_leader(devices, family, read=sample_positions):
    """Only accept one moving arm and one stable, responding arm."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        samples = list(pool.map(lambda d: read(d.port, family), devices))
    moving = [i for i, span in enumerate(samples) if span >= 32]
    if len(moving) != 1 or any(span > 12 for i, span in enumerate(samples) if i not in moving):
        return None
    return moving[0]
