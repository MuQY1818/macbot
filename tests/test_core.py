"""Fast tests that need no robot, camera or model."""
import threading
import time

import numpy as np
import pytest

from macbot.collect.checks import check_episode
from macbot.config import load
from macbot.robot.rebot import JOINTS
from macbot.data.export import binarize
from macbot.i18n import text, tr
from macbot.policy.executors import AsyncChunk, Ensemble, SyncChunk, _interp
from macbot.policy.gripper import BinaryGripper


def test_config_defaults_and_overlay(tmp_path):
    cfg = load()
    assert cfg.task and set(cfg["cameras"]) == {"top", "wrist"}
    p = tmp_path / "c.yaml"
    p.write_text(f"data_root: {tmp_path}\ntask: {{name: pick}}\nrecord: {{fps: 15}}\n")
    cfg = load(p)
    assert cfg.task == "pick" and cfg["record"]["fps"] == 15
    assert cfg["record"]["jpeg_quality"] == load()["record"]["jpeg_quality"]     # untouched keys survive the merge
    assert cfg.raw_dir() == tmp_path / "raw" / "pick"


def test_tr():
    m = tr("hello", "你好")
    assert text(m, "zh") == "你好" and text(m, "en") == "hello" and text("plain", "zh") == "plain"


def test_binarize_hysteresis():
    g = np.array([-150, -50, -30, -60, -70, -90, -100, -35], float)
    assert binarize(g, close_above=-40, open_below=-80).tolist() == [0, 0, 1, 1, 1, 0, 0, 1]


def test_binary_gripper_debounce():
    g = BinaryGripper({"open_deg": -150, "close_deg": 0}, debounce=0.5)
    assert g(0.1, 0.0) == -150
    assert g(0.9, 0.1) == 0
    assert g(0.1, 0.2) == 0          # short "open" is ignored
    assert g(0.9, 0.3) == 0
    assert g(0.1, 0.4) == 0
    assert g(0.1, 0.95) == -150      # open after 0.55 s


def _episode(n, fps=30, age=0.02):
    q = np.zeros((n, 7))
    return {"qpos": q, "late": np.zeros(n), "top_age": np.full(n, age), "top_repeat": np.zeros(n)}


def test_check_episode():
    checks = load()["checks"]
    ok = check_episode(_episode(300), ["top"], 30, checks, min_s=2)
    assert ok["ok"] and not ok["warnings"]
    short = check_episode(_episode(30), ["top"], 30, checks, min_s=2)
    assert not short["ok"]
    slow = check_episode(_episode(300, age=0.5), ["top"], 30, checks, min_s=2)
    assert slow["ok"] and slow["warnings"]


class FakePolicy:
    """Chunk of H steps whose joint values count up from the call index."""
    kind, H = "act", 20

    def __init__(self, delay=0.0):
        self.calls, self.delay = 0, delay

    def __call__(self, imgs, q, **_):
        time.sleep(self.delay)
        self.calls += 1
        return np.tile(np.arange(self.H, dtype=float)[:, None] + 100 * self.calls, (1, 7))


def _observe():
    return {}, [0.0] * 7


def test_interp():
    ch = np.arange(5, dtype=float)[:, None]
    assert _interp(ch, 1.5)[0] == 1.5 and _interp(ch, 10)[0] == 4 and _interp(ch, -1)[0] == 0


def test_sync_chunk_replans_every_k():
    pol, log = FakePolicy(), []
    ex = SyncChunk(pol, _observe, rate=30, log=lambda t, ms, **k: log.append(t), k=5)
    t0 = time.time()
    a, i = ex.target(t0)
    assert i == 0 and a[0] == 100
    a, i = ex.target(ex.t0 + 6 / 30)
    assert i == 1 and pol.calls == 2


@pytest.mark.parametrize("cls", [Ensemble, AsyncChunk])
def test_threaded_executors_produce_targets(cls):
    stop = threading.Event()
    ex = cls(FakePolicy(delay=0.01), _observe, 30, lambda *a, **k: None, stop=stop.is_set)
    ex.start()
    try:
        assert ex.is_alive()
        ex.reset(True)
        deadline = time.time() + 2
        a = None
        while a is None and time.time() < deadline:
            a, _ = ex.target(time.time())
            time.sleep(0.01)
        assert a is not None and a.shape == (len(JOINTS),)
        ex.reset(False)
        assert ex.target(time.time())[0] is None or cls is Ensemble
    finally:
        stop.set()


def test_cameras_section_is_replaced(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("cameras: {top: {type: uvc, unique_id: '0x1', crop: [0, 0, 0, 0], save_size: [640, 360]}}\n")
    assert list(load(p)["cameras"]) == ["top"]


def test_fake_arm_follows_config():
    from macbot.robot.fake import fake_for
    r = fake_for({"type": "rebot_b601"})
    assert r.joints == JOINTS and r.gripper == "gripper" and r.gripper_index == 6
    r = fake_for({"type": "lerobot", "follower": {"type": "no_such_arm"}, "limits": {"gripper": [0, 50]}})
    assert len(r.joints) == 6 and r.limits["gripper"] == (0, 50)
    r.connect(leader=True)
    q = r.read()
    sent = r.send([1.0] * 5 + [80.0], q)
    assert sent[-1] == 50.0                        # clamped to the configured limit
    assert len(r.read_leader()) == 6
