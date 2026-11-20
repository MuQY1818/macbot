import json
import time
from contextlib import ExitStack
from types import SimpleNamespace

import numpy as np
import pytest

from macbot import robot as robots
from macbot.config import load
from macbot.policy import executors, runner
from macbot.policy.gripper import BinaryGripper, takeover_gripper_alignment
from macbot.robot.fake import FakeArm


@pytest.mark.parametrize("sim", [False, True], ids=["fake", "sim-interface"])
@pytest.mark.parametrize("binary", [True, False], ids=["binary-gripper", "raw-gripper"])
def test_takeover_holds_then_follows_and_replans(tmp_path, monkeypatch, sim, binary):
    cfg = load()
    cfg["run"]["takeover"] = sim
    cfg["run"]["max_vel"] = 60
    cfg["run"]["takeover_gripper_align_deg"] = 10
    cfg["record"]["fps"] = 60
    cfg["export"]["gripper_close_above"] = -30
    cfg["robot"]["type"] = "sim" if sim else "rebot_b601"
    cfg["cameras"] = {}
    path = tmp_path / "policy"
    path.mkdir()
    (path / "config.json").write_text('{"type": "act"}')
    args = SimpleNamespace(fake=not sim, takeover=not sim, speed=.5, policy=path, mode="chunk", k=30,
                           te_m=.01, te_hz=30, rtc_beta=10, grip_debounce=0, steps=None, device="cpu", autostart=0)
    st = runner.State(cfg, tmp_path, args)
    commands, observations, sent, cleanup_calls = [], [], {}, []

    class Arm(FakeArm):
        ended = False

        def check_rate(self, hz, setting):
            assert hz == 60

        def reset_scene(self):
            pass

        def connect(self, leader=False):
            assert leader

        def read(self):
            q = list(commands[-1]) if commands else [0.0, 0.0, -100.0]
            if q[2] == 0:  # the gripper is commanded closed but an object blocks it
                q[2] = -55
            return q

        def read_leader(self):
            if step == 2:
                return [0.0, -45.0, 0.0]   # gripper matches but an arm joint does not
            if step == 3:
                return [1.0, 0.0, -120.0]  # arms match but leader gripper is open
            if step == 4:
                return [1.0, 0.0, -35.0 if binary else -12.0]  # fails the configured threshold
            if step == 5:
                return [1.0, 0.0, 0.0]    # match the command, despite the blocked fingers
            if step == 7:
                raise OSError("disconnected")
            if step == 10:
                return list(commands[-1])
            return [80.0, -6.0, -85.0]

        def send(self, target, measured=None):
            commands.append(list(target))
            sent[step] = (st.mode, list(target))
            return list(target)

        def shutdown(self, moved=True):
            cleanup_calls.append(moved)

    class Policy:
        H, kind, shapes = 30, "act", None

        def __call__(self, images, q):
            observations.append(q)
            return np.tile([20.0, 0.0, 1.0 if binary else 0.0], (self.H, 1))

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
    want = {0: "start", 2: "takeover", 4: "mark_hand", 8: "takeover", 9: "takeover",
            11: "home", 12: "pause", 13: "save_fail", 14: "quit"}

    def tick():
        action = want.get(step)
        if action:
            st.request(action)
        return real_command()

    real_command = st.take_command
    monkeypatch.setattr(st, "take_command", tick)
    srv = SimpleNamespace(url="offline", serve_forever=lambda: None, shutdown=lambda: None)
    gripper = BinaryGripper({"open_deg": -100, "close_deg": 0}, 0) if binary else None
    with ExitStack() as stack:
        runner._control(st, srv, cfg["record"]["fps"], gripper, stack)

    if not sim:
        for i in (2, 3, 4):
            assert sent[i] == ("aligning", sent[1][1])
        assert sent[7] == ("following", sent[6][1])  # a brief dropout holds the command but keeps the takeover
        assert sent[11][0] == "homing"
        assert sent[11][1][0] < sent[10][1][0]
    else:
        assert set(sent) == {0, 1, 5, 6, 7, 8, 10}  # no sim steps while aligning or requesting a reset
    assert sent[5] == ("following", sent[1][1])
    assert sent[6][0] == "following"
    assert sent[6][1][2] == -2.5  # 300 deg/s × 0.5 / 60 Hz, despite a leader target of -85
    assert observations[4][0] == pytest.approx(sent[6][1][0])
    assert max(abs(x) for x in np.diff(np.asarray(commands)[:, 0])) <= .5
    assert cleanup_calls == [True]
    with np.load(tmp_path / "rollout_000.npz") as d:
        np.testing.assert_array_equal(d["takeover"], [0, 0, 2, 2, 2, 1, 1, 2, 0, 2, 1])
        assert np.isnan(d["leader"][7]).all()
        meta = json.loads(str(d["meta"]))
        assert meta["hand_marks"] == pytest.approx([4 / 60])
        assert meta["takeover_count"] == 2
        assert meta["takeover_seconds"] == pytest.approx(4 / 60, abs=1e-3)  # the held dropout frame is still takeover time


def test_binary_takeover_requires_an_open_leader_when_follower_is_open():
    binary = {"open_deg": -120, "close_deg": 0}
    gate = {"close_above": -40, "open_below": -80}
    assert takeover_gripper_alignment(0, -120, binary, gate) == {"leader": "closed", "follower": "open", "ready": False}
    assert not takeover_gripper_alignment(-80, -120, binary, gate)["ready"]
    assert takeover_gripper_alignment(-81, -120, binary, gate)["ready"]


def test_takeover_soft_starts_and_labels_human_frames_after(tmp_path, monkeypatch):
    cfg = load()
    cfg["run"]["takeover"] = True
    cfg["run"]["home_vel"] = 30
    cfg["record"]["fps"] = 60
    cfg["robot"]["type"] = "rebot_b601"
    cfg["cameras"] = {}
    path = tmp_path / "policy"
    path.mkdir()
    (path / "config.json").write_text('{"type": "act"}')
    args = SimpleNamespace(fake=True, takeover=True, speed=1.0, policy=path, mode="chunk", k=30, te_m=.01, te_hz=30,
                           rtc_beta=10, grip_debounce=0, steps=None, device="cpu", autostart=0)
    st = runner.State(cfg, tmp_path, args)
    commands = []

    class Arm(FakeArm):
        ended = False

        def check_rate(self, hz, setting):
            pass

        def connect(self, leader=False):
            assert leader

        def read(self):
            return list(commands[-1]) if commands else [0.0, 0.0, -120.0]

        def read_leader(self):
            return [20.0, -10.0, -120.0]  # a rough match: 20 deg and 10 deg off, gripper open like the follower

        def send(self, target, measured=None):
            commands.append(list(target))
            return list(target)

        def shutdown(self, moved=True):
            pass

    class Policy:
        H, kind, shapes = 30, "act", None

        def __call__(self, images, q):
            return np.tile([0.0, 0.0, -120.0], (self.H, 1))

    arm = Arm(["a", "b", "gripper"], {}, "fake")
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
    want = {0: "start", 2: "takeover", 140: "save_ok", 141: "quit"}
    real_command = st.take_command

    def tick():
        if want.get(step):
            st.request(want[step])
        return real_command()

    monkeypatch.setattr(st, "take_command", tick)
    srv = SimpleNamespace(url="offline", serve_forever=lambda: None, shutdown=lambda: None)
    with ExitStack() as stack:
        runner._control(st, srv, cfg["record"]["fps"], None, stack)

    a = np.asarray(commands)[:, 0]
    da = np.abs(np.diff(a))
    moving = da[da > 0]
    assert a[-1] == pytest.approx(20.0)                   # reached the leader
    assert moving[0] <= 30 / 60 + 1e-9                    # starts at home_vel
    assert moving.max() <= cfg["run"]["max_vel"] / 60 + 1e-9
    assert moving[5] > moving[0]                          # speeds up toward max_vel
    b = np.asarray(commands)[:, 1]
    reached_a, reached_b = np.argmax(a >= 20 - 1e-6), np.argmax(b <= -10 + 1e-6)
    assert abs(int(reached_a) - int(reached_b)) <= 1     # both joints arrive together
    with np.load(tmp_path / "rollout_000.npz") as d:
        labels = d["takeover"]
    human = np.flatnonzero(labels == 1)
    assert len(human) and (labels[2:human[0]] == 2).all()   # catch-up frames are not labelled human
    assert (labels[human[0]:] == 1).all()


@pytest.mark.parametrize("lead", [0.0, 0.25])
def test_grip_lead_closes_earlier(tmp_path, monkeypatch, lead):
    cfg = load()
    cfg["record"]["fps"] = 60
    cfg["robot"]["type"] = "rebot_b601"
    cfg["cameras"] = {}
    path = tmp_path / "policy"
    path.mkdir()
    (path / "config.json").write_text('{"type": "act"}')
    args = SimpleNamespace(fake=True, takeover=False, speed=1.0, policy=path, mode="chunk", k=60, te_m=.01, te_hz=30,
                           rtc_beta=10, grip_debounce=0, grip_lead=lead, steps=None, device="cpu", autostart=0)
    st = runner.State(cfg, tmp_path, args)
    commands = []

    class Arm(FakeArm):
        ended = False

        def check_rate(self, hz, setting):
            pass

        def read(self):
            return list(commands[-1]) if commands else [0.0, 0.0, -100.0]

        def send(self, target, measured=None):
            commands.append(list(target))
            return list(target)

        def shutdown(self, moved=True):
            pass

    class Policy:
        H, kind, shapes = 60, "act", None

        def __call__(self, images, q):
            ch = np.zeros((self.H, 3))
            ch[30:, 2] = 1.0                       # the policy plans to close at step 30
            return ch

    monkeypatch.setattr(robots, "open_robot", lambda *a, **kw: Arm(["a", "b", "gripper"], {}, "fake"))
    monkeypatch.setattr(runner, "open_cameras", lambda *a: {})
    monkeypatch.setattr(runner, "load_policy", lambda *a: Policy())
    step = 0

    def sleep(_seconds):
        nonlocal step
        step += 1

    clock = SimpleNamespace(
        time=lambda: 100 + step / 60 + step * 1e-8, perf_counter=time.perf_counter,
        monotonic=lambda: step / 60, strftime=time.strftime, sleep=sleep,
    )
    monkeypatch.setattr(runner, "time", clock)
    monkeypatch.setattr(executors, "time", clock)  # chunk timing must follow the simulated clock too
    want = {0: "start", 100: "save_ok", 101: "quit"}
    real_command = st.take_command

    def tick():
        if want.get(step):
            st.request(want[step])
        return real_command()

    monkeypatch.setattr(st, "take_command", tick)
    srv = SimpleNamespace(url="offline", serve_forever=lambda: None, shutdown=lambda: None)
    with ExitStack() as stack:
        runner._control(st, srv, cfg["record"]["fps"], BinaryGripper({"open_deg": -100, "close_deg": 0}, 0), stack)

    with np.load(tmp_path / "rollout_000.npz") as d:
        cmd, gp, ahead = d["cmd"][:, 2], d["gp"], d["gp_ahead"]
        meta = json.loads(str(d["meta"]))
    first_close = int(np.argmax(cmd > -100))       # the command starts moving toward closed
    first_planned = int(np.argmax(gp > 0.6))
    assert meta["grip_lead"] == lead
    if lead:
        assert first_planned - first_close == pytest.approx(lead * 60, abs=2)
        assert np.isfinite(ahead).all()
    else:
        assert abs(first_close - first_planned) <= 1
        assert np.isnan(ahead).all()


def test_takeover_rides_through_brief_dropouts_and_stops_on_long_ones(tmp_path, monkeypatch):
    cfg = load()
    cfg["run"]["takeover"] = True
    cfg["record"]["fps"] = 60
    cfg["robot"]["type"] = "rebot_b601"
    cfg["cameras"] = {}
    path = tmp_path / "policy"
    path.mkdir()
    (path / "config.json").write_text('{"type": "act"}')
    args = SimpleNamespace(fake=True, takeover=True, speed=1.0, policy=path, mode="chunk", k=30, te_m=.01, te_hz=30,
                           rtc_beta=10, grip_debounce=0, steps=None, device="cpu", autostart=0)
    st = runner.State(cfg, tmp_path, args)
    commands, modes = [], {}

    class Arm(FakeArm):
        ended = False

        def check_rate(self, hz, setting):
            pass

        def connect(self, leader=False):
            pass

        def read(self):
            return list(commands[-1]) if commands else [0.0, 0.0, -120.0]

        def read_leader(self):
            if 40 <= step < 46 or 160 <= step < 200:
                raise OSError("no fresh reading")    # 0.1 s, then 0.67 s
            return [20.0, -10.0, -120.0] if step < 46 else [50.0, -10.0, -120.0]

        def send(self, target, measured=None):
            commands.append(list(target))
            modes[step] = st.mode
            return list(target)

        def shutdown(self, moved=True):
            pass

    class Policy:
        H, kind, shapes = 30, "act", None

        def __call__(self, images, q):
            return np.tile([0.0, 0.0, -120.0], (self.H, 1))

    monkeypatch.setattr(robots, "open_robot", lambda *a, **kw: Arm(["a", "b", "gripper"], {}, "fake"))
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
    want = {0: "start", 2: "takeover", 220: "save_ok", 221: "quit"}
    real_command = st.take_command

    def tick():
        if want.get(step):
            st.request(want[step])
        return real_command()

    monkeypatch.setattr(st, "take_command", tick)
    srv = SimpleNamespace(url="offline", serve_forever=lambda: None, shutdown=lambda: None)
    with ExitStack() as stack:
        runner._control(st, srv, cfg["record"]["fps"], None, stack)

    a = np.asarray(commands)[:, 0]
    assert all(modes[i] == "following" for i in range(40, 46))   # the brief dropout keeps the takeover
    assert np.ptp(a[40:46]) == 0                                 # ...and holds the command meanwhile
    assert a[150] == pytest.approx(50.0)     # 30 deg away after the dropout: soft start again, no re-alignment
    assert modes[195] == "aligning"           # a long dropout stops following...
    assert modes[210] == "following"          # ...until the signal returns with the leader roughly in place
    with np.load(tmp_path / "rollout_000.npz") as d:
        labels = d["takeover"]
    assert (labels[40:46] == 2).all() and labels[140] == 1


def test_fresh_leader_angles_tolerate_single_servo_misses():
    from macbot.robot.rebot import fresh_leader_angles

    def sample(angle, reliable=True):
        return SimpleNamespace(angle_deg=angle, reliable=reliable)

    ids, good = {"a": 1, "b": 2}, {}
    assert fresh_leader_angles(good, ids, {1: sample(10), 2: None}, 0.0) is None          # b never read
    assert fresh_leader_angles(good, ids, {1: sample(10), 2: sample(5)}, 0.01) == {"a": 10, "b": 5}
    # b goes silent for seconds, as reBot 102 servos do: keep its last reliable angle, as collect does
    assert fresh_leader_angles(good, ids, {1: sample(11), 2: sample(99, reliable=False)}, 2.5) == {"a": 11, "b": 5}
    # silent for longer than the limit: no sample
    assert fresh_leader_angles(good, ids, {1: sample(12), 2: sample(99, reliable=False)}, 3.1) is None
    assert fresh_leader_angles(good, ids, {1: sample(12), 2: sample(6)}, 3.2) == {"a": 12, "b": 6}
