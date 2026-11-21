import shlex

import pytest

from macbot.cameras.discovery import Camera
from macbot.cli import main
from macbot.config import load
from macbot.init_wizard import Setup
from macbot.lerobot_command import parse
from macbot.robot.usb import Adapter
from macbot.web.init_wizard import State

COMMAND = """lerobot-record \\
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=blue \\
  --teleop.type so100_leader --teleop.port=/dev/cu.leader --teleop.id=red \\
  --robot.cameras='{scene_cam: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \\
  --dataset.fps=25 --dataset.single_task="put the cube in the box" --dataset.repo_id=owner/demo
"""


def fixture_setup(path, imported):
    return Setup(path, imported=imported, scan=lambda: [
        Adapter("/dev/cu.follower", 1, 2, "F", "Board"),
        Adapter("/dev/cu.leader", 1, 2, "L", "Board"),
    ], cameras=lambda: ([Camera("uvc", "Camera", "mac-camera-id", (1280, 720, 30))], []),
        positions=lambda *_: pytest.fail("Import must not sample motors."),
        calibration=lambda _: [], check=lambda _: (0, "doctor passed"))


def test_import_remaps_only_device_addresses(tmp_path):
    setup = fixture_setup(tmp_path / "profile.yaml", parse(COMMAND))
    state = State(setup)
    state.refresh()
    inventory = state.inventory()
    assert inventory["roles"] == ["scene_cam"]
    assert inventory["remembered"] == {"family": "so101", "leader": 1, "cameras": {}}
    assert "--dataset.repo_id" in " ".join(inventory["notes"])
    state.action("/api/save", {**inventory["remembered"], "follower": 0, "cameras": {"scene_cam": 0},
                               "revision": inventory["revision"]})
    cfg = load(setup.path)
    assert cfg["robot"]["follower"] == {"type": "so101_follower", "port": "/dev/cu.follower", "id": "blue"}
    assert cfg["robot"]["leader"]["type"] == "so100_leader" and cfg["robot"]["leader"]["id"] == "red"
    assert cfg["cameras"]["scene_cam"] == {"type": "uvc", "unique_id": "mac-camera-id", "exact_id": True,
                                          "expect_size": [640, 480], "save_size": [640, 480], "fps": 30}
    assert cfg["record"]["fps"] == 25 and cfg["task"]["instruction"] == "put the cube in the box"


def test_import_cli_headless_checks_matches_and_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr("macbot.init_wizard.Setup", fixture_setup)
    path = tmp_path / "profile.yaml"
    args = ["--config", str(path), "init", "--yes", "--from-lerobot"]
    with pytest.raises(SystemExit, match="Omit --yes"):
        main([*args, COMMAND])
    assert not path.exists()
    on_mac = COMMAND.replace("/dev/ttyACM0", "/dev/cu.follower").replace(
        "index_or_path: 0", "index_or_path: mac-camera-id")
    assert main([*args, on_mac]) == 0
    with pytest.raises(SystemExit, match="already exists"):
        main([*args, on_mac])


def test_import_flags_only_and_unsupported_camera():
    tokens = shlex.split(COMMAND.replace("\\\n", ""))
    assert parse(shlex.join(tokens[1:])).profile == parse(COMMAND).profile
    with pytest.raises(ValueError, match="realsense.*cannot be imported"):
        parse(COMMAND.replace("opencv", "realsense"))
