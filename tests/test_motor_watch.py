import json
import time
from contextlib import ExitStack
from types import SimpleNamespace

import numpy as np
import pytest

from macbot import robot as robots
from macbot.cli import main
from macbot.collect import server
from macbot.config import load
from macbot.policy import runner
from macbot.robot.fake import FakeArm
from macbot.robot.rebot import RebotArm, _make_feedback_class
from macbot.robot.telemetry import UNKNOWN, MotorSample
from macbot.safety import MotorWatch

JOINTS = ["shoulder_lift", "gripper"]


def watch(**settings):
    watch = MotorWatch(load()["safety"] | settings)
    watch.set_joints(JOINTS, "gripper")
    return watch


def test_tracking_requires_continuous_error_on_one_arm_joint():
    w = watch()
    q, samples = [0, 0], (UNKNOWN,) * 2
    for i in range(60):
        # The gripper can be blocked; a short excursion in the arm is not a trip.
        assert not w.update(i / 60, q, [9 if i % 12 < 10 else 0, 90], samples)
    for i in range(18):
        assert not w.update(1 + i / 60, q, [9, 90], samples)
    events = w.update(1.32, q, [9, 90], samples)
    assert events[0]["kind"] == "tracking"
    assert events[0]["joint"] == "shoulder_lift"
    assert w.pose == (0, 0)


def test_warning_hysteresis_missing_samples_and_hold():
    w = watch(hold_warn_s=0.15)
    assert w.update(0, [-15, 0], [-15, 0], (MotorSample(61), UNKNOWN), holding=True, home=[0, 0])[0][
        "kind"] == "temperature_warning"
    assert w.blocked()
    assert not w.update(.08, [-15, 0], [-15, 0], (UNKNOWN, UNKNOWN), holding=True, home=[0, 0])
    assert w.blocked()
    assert w.update(.16, [-15, 0], [-15, 0], (MotorSample(55), UNKNOWN), holding=True, home=[0, 0])[0][
        "kind"] == "static_hold"
    assert w.blocked() and w.rest
    w.update(.22, [-2, 0], [-2, 0], (MotorSample(54), UNKNOWN), holding=True, home=[0, 0])
    assert not w.blocked() and not w.rest


def test_collect_start_requires_cooling(tmp_path):
    cfg = load()
    cfg["cameras"] = {}
    st = server.State(cfg, tmp_path, True)
    st.ready = True
    st.watch.set_joints(JOINTS, "gripper")
    for i, temp in enumerate((61, 59, 55, 54)):
        st.watch.update(i / 60, [0, 0], [0, 0], (MotorSample(temp), UNKNOWN))
        st.command({"action": "start"})
        assert (st.phase == "recording") == (temp < 55)


@pytest.mark.parametrize("sample", [MotorSample(71), MotorSample(40, status=4), MotorSample(fault=0x10000)])
def test_motor_alarm_is_immediate_and_latched(sample):
    w = watch()
    w.update(1, [4, 5], [4, 5], (UNKNOWN, sample))
    assert w.trip["joint"] == "gripper"
    assert w.pose == (4, 5)
    w.update(1.02, [3, 2], [4, 5], (UNKNOWN, UNKNOWN))
    assert w.blocked() and w.pose == (4, 5)
    off = watch(enabled=False)
    assert not off.update(1, [4, 5], [4, 5], (UNKNOWN, sample))
    assert off.samples[1] == sample and not off.blocked()


def test_rebot_reuses_position_feedback_and_ignores_placeholder_rotor(monkeypatch):
    from macbot.robot import rebot

    state = SimpleNamespace(t_mos=43., t_rotor=0., torq=-2., status_code=4)
    reads = []

    class Upstream:
        def _read_feedback(self):
            reads.append("feedback")
            return {JOINTS[0]: state}

    monkeypatch.setattr(rebot, "_lerobot", lambda: (None, Upstream, None, None))
    follower = _make_feedback_class()[0]()
    follower.motors = {JOINTS[0]: SimpleNamespace(robstride_get_fault_report=lambda: (1, 2))}
    arm = RebotArm.__new__(RebotArm)
    arm.joints, arm.follower = JOINTS, follower
    assert arm.motor_telemetry() == (UNKNOWN, UNKNOWN)
    assert follower._read_feedback()[JOINTS[0]] is state
    assert arm.motor_telemetry() == (MotorSample(43., None, -2., 4, 1, 2), UNKNOWN)
    assert reads == ["feedback"]


def offline_server():
    return SimpleNamespace(url="offline", serve_forever=lambda: None, shutdown=lambda: None)


def test_run_trips_during_alignment_and_archives(tmp_path, monkeypatch, capsys):
    cfg = load()
    cfg["cameras"] = {}
    path = tmp_path / "policy"
    path.mkdir()
    (path / "config.json").write_text('{"type":"act"}')
    args = SimpleNamespace(fake=True, takeover=True, speed=.5, policy=path, mode="chunk", k=30, te_m=.01,
                           rtc_beta=10, grip_debounce=0, steps=None, device="cpu", autostart=0)
    st = runner.State(cfg, tmp_path, args)
    step, commands, shutdown = 0, [], []

    class Arm(FakeArm):
        def read(self):
            return [1., 0.] if step < 5 else [-5. - (step - 5), 0.]

        def read_leader(self):
            return [-80., 0.]

        def send(self, target, measured=None):
            commands.append(list(target))
            return list(target)

        def motor_telemetry(self):
            return (MotorSample(72. if step == 5 else 40., torque_nm=-2.), UNKNOWN)

        def shutdown(self, moved=True):
            shutdown.append(moved)

    class Policy:
        H, kind, shapes = 30, "act", None

        def __call__(self, images, q):
            return np.tile([20., 0.], (self.H, 1))

    arm = Arm(JOINTS, {}, "fake")
    monkeypatch.setattr(robots, "open_robot", lambda *a, **k: arm)
    monkeypatch.setattr(runner, "load_policy", lambda *a: Policy())
    monkeypatch.setattr(runner, "open_cameras", lambda *a: {})

    def sleep(_):
        nonlocal step
        step += 1

    monkeypatch.setattr(runner, "time", SimpleNamespace(
        time=lambda: 100 + step / 60, strftime=time.strftime, sleep=sleep, perf_counter=lambda: step / 60,
    ))
    take = st.take_command

    def request():
        action = {0: "start", 2: "takeover", 6: "start", 7: "home", 8: "save_ok", 9: "quit"}.get(step)
        if action:
            st.request(action)
        return take()

    monkeypatch.setattr(st, "take_command", request)
    with ExitStack() as cleanup:
        runner._control(st, offline_server(), 60, None, cleanup)
    assert commands[4][0] > 1
    assert commands[5:] == [[-5., 0.]] * 4
    assert shutdown == [True]  # quitting after a trip still returns to zero slowly
    assert not list(tmp_path.glob("rollout_*.npz"))
    diag, = (tmp_path / "watchdog").glob("*.npz")
    with np.load(diag) as z:
        meta = json.loads(str(z["meta"]))
        assert meta["result"] == "watchdog"
        assert z["motor_temp_c"].shape == z["qpos"].shape
        assert (z["motor_torque_nm"][:, 0] == -2.).all()
        assert np.isnan(z["motor_rotor_c"]).all()
        assert meta["watchdog_events"][-1]["time_s"] == pytest.approx(5 / 60)
    main(["health", "--data", str(tmp_path)])
    report = capsys.readouterr().out
    assert "shoulder_lift 72.0 °C" in report and "temperature_alarm shoulder_lift" in report


def test_collect_stops_leader_and_archives_interrupted_episode(tmp_path, monkeypatch):
    cfg = load()
    cfg["cameras"] = {}
    st = server.State(cfg, tmp_path, True)
    commands, shutdown = [], []

    class Arm(FakeArm):
        def read(self):
            return [-2. if len(commands) >= 9 else 0., 0.]

        def read_leader(self):
            if not commands and st.ready:
                st.command({"action": "start"})
            assert len(commands) < 10  # the leader is not used after a trip
            return [15., 0.]

        def motor_telemetry(self):
            return (MotorSample(39., torque_nm=1., status=1 if len(commands) == 9 else 0), UNKNOWN)

        def send(self, target, measured=None):
            commands.append(list(target))
            if len(commands) == 12:
                st.command({"action": "save"})
                st.command({"action": "start"})
                st.command({"action": "quit"})
            assert len(commands) <= 12
            return list(target)

        def shutdown(self, moved=True):
            shutdown.append(moved)

    monkeypatch.setattr(robots, "open_robot", lambda *a, **k: Arm(JOINTS, {}, "fake"))
    monkeypatch.setattr(server, "open_cameras", lambda *a: {})
    server._control(st, offline_server(), cfg["teleop"])
    assert shutdown == [True]  # quitting after a trip still returns to zero slowly
    assert commands[9:] == [[-2., 0.]] * 3
    assert not st.files()
    diag, = (tmp_path / "watchdog").glob("*.npz")
    with np.load(diag) as z:
        assert len(z["qpos"]) > 0
        assert z["motor_temp_c"].shape == z["qpos"].shape
        meta = json.loads(str(z["meta"]))
        assert meta["result"] == "watchdog" and not meta["check"]["ok"]
        assert meta["watchdog_events"][0]["status"] == 1


def test_gripper_stall_is_not_a_fault_but_arm_stall_is():
    from macbot.robot.telemetry import MotorSample
    from macbot.safety import MotorWatch

    cfg = {"enabled": True, "track_deg": 8, "track_s": .3, "hold_warn_s": 30,
           "temp_warn_c": 60, "temp_alarm_c": 70, "temp_resume_c": 55}
    watch = MotorWatch(cfg)
    watch.set_joints(["a", "gripper"], "gripper")
    ok, stall = MotorSample(40.0, None, 0.0, 0, 0, 0), MotorSample(40.0, None, 0.0, 1 << 4, 0, 0)
    assert watch.update(0.0, [0, 0], [0, 0], (ok, stall)) == []
    assert watch.trip is None
    events = watch.update(0.01, [0, 0], [0, 0], (stall, ok))
    assert events and events[0]["kind"] == "fault" and events[0]["joint"] == "a"


def test_status_before_the_first_feedback():
    from macbot.safety import MotorWatch

    w = MotorWatch({"enabled": True, "track_deg": 8, "track_s": .3, "hold_warn_s": 30,
                    "temp_warn_c": 60, "temp_alarm_c": 70, "temp_resume_c": 55})
    w.set_joints(["a", "gripper"], "gripper")
    assert w.status()["levels"] == []      # the console polls before the control loop has read the motors
