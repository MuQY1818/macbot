import sys
from types import SimpleNamespace

import pytest

from macbot.cameras.discovery import Camera, discover_cameras
from macbot.robot.usb import (
    Adapter,
    calibration_id,
    detect_leader,
    position_span,
    resolve_robot,
    sample_positions,
    scan_adapters,
)


def usb(port, *, vid=0x1A86, pid=0x7523, serial="A"):
    return SimpleNamespace(device=port, vid=vid, pid=pid, serial_number=serial, product="Board", description="")


def test_usb_hints_and_aliases():
    a, b, c = scan_adapters(
        [usb("/dev/tty.a"), usb("/dev/cu.a"), usb("/dev/cu.b", pid=1, serial=None)],
        [{"idVendor": 0x0C72, "idProduct": 0x000C, "USB Serial Number": "CAN"}],
    )
    assert a.hint[1] == ("so100", "so101")
    assert b.hint[0] == "Unknown USB adapter"
    assert c.port is None and c.hint[1] == ("rebot_b601",)


def test_switch_ports_without_switching_devices():
    a, b = scan_adapters([usb("/dev/cu.first"), usb("/dev/cu.second", serial="B")], [])
    robot = {"type": "lerobot", "follower": {"port": b.port}, "leader": {"port": a.port},
             "usb": {"leader": a.identity, "follower": b.identity}}
    first, second = scan_adapters([usb("/dev/cu.first", serial="B"), usb("/dev/cu.second")], [])
    resolved = resolve_robot(robot, lambda: [first, second])
    assert resolved["leader"]["port"] == second.port
    assert calibration_id(a.identity, "leader") == calibration_id(second.identity, "leader")
    with pytest.raises(ValueError, match="missing"):
        resolve_robot(robot, lambda: [first])
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_robot(robot, lambda: [first, second, second])


def test_motion_selects_one_arm_only(monkeypatch):
    import macbot.robot.usb as module

    monkeypatch.setattr(module, "monotonic", iter([0, 0, 1, 2, 3]).__next__)
    monkeypatch.setattr(module, "sleep", lambda _: None)
    readings = iter([4090, 8, 48])
    assert position_span(lambda: {"1": next(readings)}, 3) == 54
    a, b = [Adapter(p, 0x1A86, 0x7523, p, "Identical board") for p in ("a", "b")]
    assert detect_leader([a, b], "so101", lambda p, _: {"a": 4, "b": 54}[p]) == 1
    assert detect_leader([a, b], "so101", lambda *_: 54) is None
    assert detect_leader([a, b], "so101", lambda *_: 4) is None


def test_position_probe_only_reads_and_closes_without_torque(monkeypatch):
    import macbot.robot.usb as module

    events = []

    class Bus:
        def __init__(self, port, motors):
            assert port == "fake" and len(motors) == 6

        def connect(self, *, handshake):
            assert handshake is False
            events.append("open")

        def read(self, name, key, *, normalize):
            assert name == "Present_Position" and normalize is False
            events.append("read")
            raise ConnectionError("unplugged")

        def disconnect(self, *, disable_torque):
            assert disable_torque is False
            events.append("close")

    monkeypatch.setattr(module, "monotonic", iter([0, 0]).__next__)
    monkeypatch.setitem(sys.modules, "lerobot.motors", SimpleNamespace(
        Motor=lambda *_: None, MotorNormMode=SimpleNamespace(DEGREES=0), SerialMotorsBus=Bus,
    ))
    with pytest.raises(ConnectionError, match="unplugged"):
        sample_positions("fake", "so101")
    assert events == ["open", "read", "close"]


def test_camera_discovery_ignores_iphone_and_uses_reported_modes():
    devices = [
        {"name": "USB Camera", "unique_id": "uvc", "modes": [(1920, 1080, 15), (1280, 720, 30)]},
        {"name": "Personal Phone", "unique_id": "phone", "continuity": True, "modes": [(1920, 1080, 30)]},
        {"name": "iPhone", "unique_id": "old-os", "modes": [(1920, 1080, 30)]},
    ]
    cams, notes = discover_cameras(lambda: devices, lambda: [Camera("orbbec", "Gemini", "G1", (640, 480, 15))])
    assert not notes and [c.uid for c in cams] == ["uvc", "G1"]
    assert cams[0].config()["expect_size"] == [1280, 720]
    assert cams[0].config()["save_size"] == [640, 360]
    assert cams[1].config()["serial"] == "G1"


def test_orbbec_inventory_uses_sdk_serials_and_color_modes(monkeypatch):
    from macbot.cameras import discovery

    devices = SimpleNamespace(get_count=lambda: 2, get_device_serial_number_by_index=lambda i: ("G1", "G2")[i],
                              get_device_by_index=lambda i: SimpleNamespace(
                                  get_device_info=lambda: SimpleNamespace(get_name=lambda: f"Gemini {i + 1}")))
    profile = SimpleNamespace(get_width=lambda: 1280, get_height=lambda: 720, get_fps=lambda: 30)
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", SimpleNamespace(Pipeline=lambda device: device))
    monkeypatch.setattr(discovery, "orbbec_devices", lambda: (object(), devices))
    monkeypatch.setattr(discovery, "orbbec_profiles", lambda _: [profile])
    a, b = discovery.orbbec_inventory()
    assert (a.uid, b.uid) == ("G1", "G2")
    assert a.mode == b.mode == (1280, 720, 30)
