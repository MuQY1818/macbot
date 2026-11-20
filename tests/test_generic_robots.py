"""Regression checks for non-reference layouts and observation-backed cameras."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from macbot.cameras import camera_info, open_cameras
from macbot.cli import main
from macbot.data.export import robot_type
from macbot.data.health import episode_stats
from macbot.policy import bench
from macbot.robot import lerobot_generic, open_robot, sim
from macbot.robot.fake import SO_JOINTS


def test_generic_calibration_checked_only_after_connect(monkeypatch, tmp_path):
    class Follower:
        action_features = observation_features = {"joint_a.pos": float}
        connected = False
        calibration_fpath = tmp_path / "calibration.json"
        calibration = {"joint_a": object()}

        @property
        def is_calibrated(self):
            assert self.connected
            return True

        def connect(self, *, calibrate):
            assert calibrate is False
            self.connected = True

        def get_observation(self):
            return {"joint_a.pos": 2.0}

    dev = Follower()
    monkeypatch.setattr(lerobot_generic, "make_follower", lambda spec: dev)
    cfg = {"type": "lerobot", "follower": {"type": "test_device"}}
    with pytest.raises(SystemExit, match="missing calibration"):
        open_robot(cfg)
    dev.calibration_fpath.write_text("{}")
    robot = open_robot(cfg)
    robot.connect()
    assert robot.home_pose() == [2.0]


def test_sim_camera_and_reset(monkeypatch):
    class Env:
        descriptor = SimpleNamespace(fps=30, control_mode="position", semantics="joint-position")
        action_features = {"joint_a.pos": float, "gripper.pos": float}
        observation_features = dict(action_features, front=(4, 6, 3))
        cameras = {"front": "observation.images.front"}
        robot_type = "simulated_device"
        episode_ended = False

        def connect(self, *, calibrate):
            assert not calibrate
            self.start_episode()

        def start_episode(self):
            self.value = 0.0
            self.episode_ended = False

        def send_action(self, action):
            self.value = action["joint_a.pos"]
            self.episode_ended = True
            return action

        def get_observation(self):
            rgb = np.full((4, 6, 3), [240, 30, 5], np.uint8)
            return {"joint_a.pos": self.value, "gripper.pos": 10.0, "front": rgb}

        def disconnect(self):
            self.closed = True

    dev = Env()
    monkeypatch.setattr(sim, "_make_sim", lambda spec, task_id: dev)
    robot = open_robot({"type": "sim"})
    with pytest.raises(ValueError, match="requires cameras with type: robot"):
        open_cameras({"top": {"type": "uvc"}}, 30, False, robot)
    cfg = {"top": {"type": "robot", "key": "front", "crop": [1, 0, 4, 2], "save_size": [3, 2]}}
    cam = open_cameras(cfg, 30, False, robot)["top"]
    assert cam.latest() == (None, 0.0)
    robot.connect(leader=True)
    img, stamp = cam.latest()
    assert img.shape == (2, 3, 3)
    assert img[0, 0].tolist() == [5, 30, 240]
    robot.send([10.0, 10.0])
    assert robot.ended and robot.read() == [10.0, 10.0]
    assert cam.latest()[1] >= stamp
    robot.reset_scene()
    assert not robot.ended and robot.read() == robot.home_pose() == [0.0, 10.0]
    assert robot_type({"robot": robot.info()}) == dev.robot_type
    robot.shutdown()
    assert dev.closed


@pytest.fixture
def six_joint_episode(tmp_path):
    fake = open_robot({"type": "lerobot", "follower": {"type": "so101_follower"}}, fake=True)
    cams = camera_info({"top": {"type": "uvc", "crop": [0, 0, 0, 0], "save_size": [16, 12]}})
    assert cams["top"]["crop"] is None
    meta = {"robot": fake.info(), "fake": True, "joint_names": SO_JOINTS, "gripper": "gripper",
            "cameras": cams, "fps": 30, "instruction": "test the data pipeline"}
    assert robot_type(meta) == "so101_follower"
    assert robot_type({}) == "rebot_b601_follower"
    assert robot_type({"robot": {"kind": "lerobot", "follower": {"type": "koch_follower"}}}) == "koch_follower"
    q = np.zeros((4, 6), dtype=np.float32)
    _, jpg = cv2.imencode(".jpg", np.full((12, 16, 3), (4, 30, 180), dtype=np.uint8))
    raw = tmp_path / "raw"
    raw.mkdir()
    episode = raw / "episode_000.npz"
    np.savez(episode, qpos=q, action=q, timestamp=np.arange(len(q)) / 30, meta=json.dumps(meta),
             top_jpeg=np.tile(jpg, len(q)), top_offsets=np.arange(len(q) + 1) * len(jpg))
    return episode


def test_six_joint_episode_health(six_joint_episode, capsys):
    episode = six_joint_episode
    with np.load(episode) as data:
        stats = episode_stats(data)
    assert stats["lag"] == 0 and stats["p95"].tolist() == [0.0] * 5
    main(["health", "--data", str(episode.parent)])
    assert "000" in capsys.readouterr().out


def test_six_joint_episode_export(six_joint_episode, tmp_path):
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    out = tmp_path / "dataset"
    main(["export", "--images", "--data", str(six_joint_episode.parent), "--out", str(out)])
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["robot_type"] == "so101_follower"
    assert info["total_frames"] == 4
    for key in ("observation.state", "action"):
        assert info["features"][key]["shape"] == [6]
        assert info["features"][key]["names"] == [f"{n}.pos" for n in SO_JOINTS]
    assert info["features"]["observation.images.top"]["shape"] == [12, 16, 3]


def test_bench_uses_checkpoint_state_dimension(monkeypatch, tmp_path, capsys):
    class Policy:
        kind = "act"
        H = 10

        def __call__(self, images, state):
            assert state == [0.0] * 6

    (tmp_path / "config.json").write_text(json.dumps({"input_features": {"observation.state": {"shape": [6]}}}))
    monkeypatch.setattr(bench, "load_policy", lambda *_: Policy())
    assert main(["bench", "--policy", str(tmp_path), "--device", "cpu", "-n", "2"]) is None
    assert "act on cpu, chunk 10: median" in capsys.readouterr().out


def test_pi05_bench_uses_raw_normalizer_dimension(monkeypatch, tmp_path):
    safetensors = pytest.importorskip("safetensors.numpy")

    class Policy:
        kind = "pi05"
        H = 50

        def __call__(self, images, state):
            assert state == [0.0] * 7

    (tmp_path / "config.json").write_text(json.dumps({
        "type": "pi05", "max_state_dim": 32,
        "input_features": {"observation.state": {"shape": [32]}},
        "output_features": {"action": {"shape": [7]}},
    }))
    (tmp_path / "policy_preprocessor.json").write_text(json.dumps({
        "steps": [{"registry_name": "normalizer_processor", "state_file": "stats.safetensors"}],
    }))
    safetensors.save_file({
        "observation.state.q01": np.zeros(7, np.float32),
        "observation.state.count": np.array([1000]),
    }, str(tmp_path / "stats.safetensors"))
    monkeypatch.setattr(bench, "load_policy", lambda *_: Policy())
    assert main(["bench", "--policy", str(tmp_path), "--no-compile", "-n", "2"]) is None
    from macbot.policy.models import check_joint_dimensions

    check_joint_dimensions(tmp_path, ["joint"] * 7)
    with pytest.raises(SystemExit, match="state dimension mismatch"):
        check_joint_dimensions(tmp_path, ["joint"] * 8)
