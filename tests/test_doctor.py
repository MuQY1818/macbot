"""Exercise doctor without opening a serial port, camera, or simulator."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from macbot import doctor
from macbot.cli import main
from macbot.config import load
from macbot.robot import lerobot_generic, rebot, sim


@pytest.fixture
def imports(monkeypatch):
    imported = []
    missing = set()

    def import_module(name):
        imported.append(name)
        if name in missing:
            raise ImportError(f"cannot import {name}")
        return SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))

    monkeypatch.setattr(doctor, "importlib", SimpleNamespace(import_module=import_module))
    return imported, missing


def test_rebot_doctor_checks_pcbusb(imports, monkeypatch, tmp_path, capsys):
    cfg = load()
    cfg["cameras"] = {}
    monkeypatch.setattr(doctor, "user_home", lambda: tmp_path)
    monkeypatch.setattr(doctor, "PCBUSB", tmp_path / "libPCBUSB.dylib")
    monkeypatch.setattr(rebot, "find_leader_port", lambda: "/dev/cu.usbserial-TEST")

    assert doctor.run(cfg) == 1
    out = capsys.readouterr().out
    assert "robot (rebot_b601)" in out
    assert "x   PCAN-USB driver" in out
    assert "ok  leader arm (USB serial)" in out
    assert "gravity feedforward" in out


@pytest.mark.parametrize("fault", [None, "port", "calibration", "driver"])
def test_lerobot_doctor(imports, monkeypatch, tmp_path, capsys, fault):
    def unexpected():
        pytest.fail("reBot USB detection should not run for a generic arm")

    monkeypatch.setattr(rebot, "find_leader_port", unexpected)
    constants = ModuleType("lerobot.utils.constants")
    constants.HF_LEROBOT_CALIBRATION = tmp_path / "default"
    constants.ROBOTS = "robots"
    constants.TELEOPERATORS = "teleoperators"
    monkeypatch.setitem(sys.modules, constants.__name__, constants)

    follower_port = tmp_path / "follower-port"
    leader_port = tmp_path / "leader-port"
    if fault != "port":
        follower_port.touch()
    leader_port.touch()
    custom = tmp_path / "custom"
    custom.mkdir()
    if fault != "calibration":
        (custom / "follower.json").touch()
    leader_dir = constants.HF_LEROBOT_CALIBRATION / "teleoperators" / "so_leader"
    leader_dir.mkdir(parents=True)
    (leader_dir / "leader.json").touch()

    def device_config(spec, *, leader):
        if fault == "driver" and not leader:
            raise SystemExit("unknown or unavailable device")
        return SimpleNamespace(id=spec["id"], port=spec["port"],
                               calibration_dir=spec.get("calibration_dir"))

    monkeypatch.setattr(lerobot_generic, "device_config", device_config)
    monkeypatch.setattr(doctor, "_driver_name", lambda cfg: f"so_{cfg.id}")
    cfg = tmp_path / "robot.yaml"
    cfg.write_text(
        f"cameras: {{}}\nrobot:\n  type: lerobot\n"
        f"  follower: {{type: so101_follower, id: follower, port: '{follower_port}', calibration_dir: '{custom}'}}\n"
        f"  leader: {{type: so101_leader, id: leader, port: '{leader_port}'}}\n"
    )
    assert main(["--config", str(cfg), "doctor"]) == int(fault is not None)
    out = capsys.readouterr().out
    assert "PCAN" not in out and "gravity feedforward" not in out
    assert "ok  leader serial port" in out
    assert "ok  leader calibration" in out and str(leader_dir / "leader.json") in out
    if fault == "driver":
        assert "x   follower driver" in out and "check the device type and its LeRobot extra" in out
    else:
        assert "ok  follower driver" in out
        assert ("x  " if fault == "port" else "ok ") + " follower serial port" in out
        assert ("x  " if fault == "calibration" else "ok ") + " follower calibration" in out


@pytest.mark.parametrize("missing", [False, True])
def test_sim_doctor_checks_mujoco_without_hardware(imports, monkeypatch, capsys, missing):
    imported, unavailable = imports
    if missing:
        unavailable.add("mujoco")
    cfg = load()
    cfg["robot"].update(type="sim", env={"type": "envhub"})
    cfg["cameras"] = {"front": {"type": "robot"}}

    def sim_config(spec, task_id):
        assert spec == {"type": "envhub"}
        return SimpleNamespace(env=SimpleNamespace(type="envhub", hub_path="lerobot/so101-tabletop-mujoco@rev"))

    monkeypatch.setattr(sim, "sim_config", sim_config)
    monkeypatch.setattr(sim, "getproxies", dict)
    monkeypatch.setattr(doctor.shutil, "which", lambda _: pytest.fail("should not probe Swift for robot cameras"))
    assert doctor.run(cfg) == int(missing)
    out = capsys.readouterr().out
    assert "PCAN" not in out and "serial" not in out and "calibration" not in out
    assert "ok  robot cameras" in out and "mujoco" in imported
    if missing:
        assert "install macbot[sim]" in out


def test_doctor_marks_broken_required_import_as_failure(imports, monkeypatch, capsys):
    _, unavailable = imports
    unavailable.add("lerobot")
    cfg = load()
    cfg["robot"]["type"] = "sim"
    cfg["cameras"] = {}
    monkeypatch.setattr(sim, "sim_config", lambda *_: pytest.fail("LeRobot is not installed"))
    assert doctor.run(cfg) == 1
    assert "import failed: cannot import lerobot" in capsys.readouterr().out


@pytest.mark.parametrize("cached", [False, True])
def test_sim_doctor_checks_socks_only_when_download_is_needed(imports, monkeypatch, capsys, cached):
    hub = "lerobot/so101-tabletop-mujoco@rev"
    monkeypatch.setattr(sim, "sim_config", lambda *_: SimpleNamespace(env=SimpleNamespace(type="envhub", hub_path=hub)))
    monkeypatch.setenv("ALL_PROXY", "socks5://localhost:1")
    monkeypatch.setattr(sim, "find_spec", lambda _: None)
    monkeypatch.setattr(sim, "cached_envhub_path", lambda path: "/cached:env.py" if path == hub and cached else None)
    assert doctor._check_sim({"env": {"type": "envhub"}}) == int(not cached)
    output = capsys.readouterr().out
    assert ("no SOCKS client required" if cached else "httpx[socks]") in output


def test_so101_driver_resolution():
    pytest.importorskip("lerobot.robots.so_follower")
    for role, leader in (("follower", False), ("leader", True)):
        cfg = lerobot_generic.device_config({"type": f"so101_{role}", "port": "/dev/not-connected"}, leader=leader)
        assert doctor._driver_name(cfg) == f"so_{role}"
