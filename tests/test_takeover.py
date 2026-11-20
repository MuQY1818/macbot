import json
import time
from contextlib import ExitStack
from types import SimpleNamespace

import numpy as np
import pytest

from macbot import robot as robots
from macbot.config import load
from macbot.policy import runner
from macbot.policy.gripper import BinaryGripper
from macbot.robot.fake import FakeArm


@pytest.mark.parametrize("sim", [False, True], ids=["fake", "sim-interface"])
def test_takeover_holds_then_follows_and_replans(tmp_path, monkeypatch, sim):
    cfg = load()
    cfg["run"]["takeover"] = sim
    cfg["run"]["max_vel"] = 60
    cfg["record"]["fps"] = 60
    cfg["robot"]["type"] = "sim" if sim else "rebot_b601"
    cfg["cameras"] = {}
    path = tmp_path / "policy"
    path.mkdir()
    (path / "config.json").write_text('{"type": "act"}')
    args = SimpleNamespace(fake=not sim, takeover=not sim, speed=1, policy=path, mode="chunk", k=30,
                           te_m=.01, te_hz=30, rtc_beta=10, grip_debounce=0, steps=None, device="cpu", autostart=0)
    st = runner.State(cfg, tmp_path, args)
    commands, observations, follow_types, cleanup_calls = [], [], [], []

    class Arm(FakeArm):
        ended = False

        def check_rate(self, hz, setting):
            assert hz == 60

        def reset_scene(self):
            pass

        def connect(self, leader=False):
            assert leader

        def read(self):
            return list(commands[-1]) if commands else [0.0, 0.0, -100.0]

        def read_leader(self):
            if st.mode == "running" or step == 2:
                return [0.0, -45.0, 60.0]  # one misaligned arm joint, gripper must not move
            if step == 3:
                return [1.9, 0.0, 60.0]    # arm is aligned; gripper alone cannot block
            if step == 5:
                raise OSError("disconnected")
            return [80.0, -6.0, -85.0]

        def send(self, target, measured=None):
            commands.append(list(target))
            follow_types.append(st.mode)
            return list(target)

        def shutdown(self, moved=True):
            cleanup_calls.append(moved)

    class Policy:
        H, kind, shapes = 30, "act", None

        def __call__(self, images, q):
            observations.append(q)
            return np.tile([20.0, 0.0, 0.0], (self.H, 1))

    arm = Arm(["a", "b", "gripper"], {"a": (-5, 4), "gripper": (-120, 0)}, "fake")
    monkeypatch.setattr(robots, "open_robot", lambda *a, **kw: arm)
    monkeypatch.setattr(runner, "open_cameras", lambda *a: {})
    monkeypatch.setattr(runner, "load_policy", lambda *a: Policy())
    step = 0

    def sleep(_seconds):
        nonlocal step
        step += 1

    monkeypatch.setattr(runner, "time", SimpleNamespace(
        time=lambda: 100 + step / 60 + step * 1e-8, perf_counter=time.perf_counter,
        monotonic=lambda: step / 60, strftime=time.strftime, sleep=sleep,
    ))
    want = iter([
        "start", None, "takeover", None, "mark_hand", None, "takeover",
        "takeover", "pause", "save_fail", "quit",
    ])
    def tick():
        action = next(want)
        if action:
            st.request(action)
        return real_command()

    real_command = st.take_command
    monkeypatch.setattr(st, "take_command", tick)
    srv = SimpleNamespace(url="offline", serve_forever=lambda: None, shutdown=lambda: None)
    with ExitStack() as stack:
        runner._control(st, srv, cfg["record"]["fps"], BinaryGripper({"open_deg": -100, "close_deg": 0}, 0), stack)

    if not sim:
        assert follow_types[:7] == ["running", "running", "aligning", "following", "following", "aligning", "running"]
        assert commands[2] == commands[1]
        assert commands[5] == commands[4]
    assert observations[4][0] == pytest.approx(2.9)
    assert max(abs(x) for x in np.diff(np.asarray(commands)[:, 0])) <= 1
    assert commands[4 if not sim else 3][2] == -85.0
    assert cleanup_calls == [True]
    with np.load(tmp_path / "rollout_000.npz") as d:
        np.testing.assert_array_equal(d["takeover"], [0, 0, 2, 1, 1, 2, 0, 2])
        assert np.isnan(d["leader"][5]).all()
        meta = json.loads(str(d["meta"]))
        assert meta["hand_marks"] == pytest.approx([4 / 60])
        assert meta["takeover_count"] == 2
        assert meta["takeover_seconds"] == pytest.approx(.033)
