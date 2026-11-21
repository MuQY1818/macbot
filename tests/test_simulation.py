"""Simulation batch contracts without MuJoCo, a checkpoint download, or a GPU."""
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from macbot.cli import main
from macbot.policy import evaluate
from macbot.robot import sim, sim_render
from macbot.simulation import summary
from macbot.viz.rollouts import Store


def test_wilson_endpoints():
    assert summary([{"success": False, "steps": 5}] * 50, 30)["wilson95"] == pytest.approx([0, 0.07135], abs=1e-5)
    assert summary([{"success": True, "steps": 5}] * 50, 30)["wilson95"] == pytest.approx([0.92865, 1], abs=1e-5)


def test_eval_refuses_hardware_before_resolving_policy():
    with pytest.raises(SystemExit, match="eval requires robot.type: sim"):
        main(["--config", "configs/so101.yaml", "eval", "--policy", "unused"])


def test_cgl_warns_and_tries_software_when_acceleration_is_unavailable(monkeypatch, capsys):
    class CGLError(RuntimeError):
        pass

    attrs = ("OpenGLProfile", "ColorSize", "AlphaSize", "DepthSize", "StencilSize",
             "Multisample", "SampleBuffers", "Sample", "Accelerated", "RendererID")
    api = SimpleNamespace(
        CGLError=CGLError, CGLContextObj=lambda: None,
        CGLPixelFormatAttribute=SimpleNamespace(**{f"CGLPFA{a}": i for i, a in enumerate(attrs, 1)}),
        CGLOpenGLProfile=SimpleNamespace(CGLOGLPVersion_Legacy=100),
    )
    gl = SimpleNamespace(GL_RENDERER=1, glGetString=lambda _: b"Apple Software Renderer")
    monkeypatch.setitem(sys.modules, "mujoco.cgl", SimpleNamespace(cgl=api))
    monkeypatch.setitem(sys.modules, "OpenGL.GL", gl)
    formats = []

    def select(self, values):
        formats.append(values)
        if len(formats) == 1:
            raise CGLError("test hardware failure")

    monkeypatch.setattr(sim_render.CGLContext, "_open", select)
    sim_render.CGLContext(320, 240)
    assert formats[0][-1] == api.CGLPixelFormatAttribute.CGLPFAAccelerated
    assert formats[1][-2:] == [api.CGLPixelFormatAttribute.CGLPFARendererID, 0x00020400]
    out = capsys.readouterr()
    assert "trying slower software rendering" in out.err
    assert "MuJoCo renderer: Apple Software Renderer" in out.out


def test_rtc_switches_on_the_simulated_delivery_step():
    inputs = []

    def policy(obs, q, **kwargs):
        inputs.append(kwargs)
        chunk = (np.arange(8, dtype=np.float32) + 10 * q[0])[:, None]
        return chunk, chunk

    args = SimpleNamespace(mode="rtc", k=30, query_every=3, delay_steps=2, rtc_horizon=5)
    clock = evaluate.StepClock(policy, args, fps=30, joints=1, limit_chunk=False)
    actions = []
    for step in range(6):
        action, chunk_id = clock.target(step, {}, [step], lambda *_a, **_kw: None)
        actions.append((float(action[0]), chunk_id))
        if step == 3:
            assert clock.ahead(step, 4)[0] == 7
    assert actions == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (32, 1)]
    np.testing.assert_array_equal(inputs[1]["leftover"].ravel(), np.arange(3, 8))
    assert inputs[1]["delay"] == 2


@pytest.mark.parametrize("trace_only", [False, True])
def test_eval_saves_seeded_normal_rollouts(tmp_path, monkeypatch, capsys, trace_only):
    class Env:
        descriptor = SimpleNamespace(fps=30, control_mode="position", semantics="test")
        action_features = {"joint.pos": float, "gripper.pos": float}
        observation_features = dict(action_features, front=(4, 6, 3))
        robot_type = "test_robot"
        cameras = {"front": "observation.images.front"}

        def __init__(self):
            self._env = SimpleNamespace(call=lambda name: (2,))
            self.seeds = []

        def connect(self, *, calibrate):
            self.start_episode()

        def start_episode(self, seed=None):
            self.seeds.append(seed)
            self.steps, self.episode_ended, self.q = 0, False, 0

        def send_action(self, action):
            self.steps += 1
            self.q = action["joint.pos"]
            ok = self.seeds[-1] % 2 == 0
            self.episode_ended = ok or self.steps >= 2
            self.latest = SimpleNamespace(reward=[float(ok)], is_success=[ok],
                                          terminated=[ok], truncated=[not ok and self.episode_ended])
            return action

        def get_observation(self):
            return {"joint.pos": self.q, "gripper.pos": 0, "front": np.zeros((4, 6, 3), np.uint8)}

        def disconnect(self):
            self.closed = True

    class Policy:
        H = 2

        def __call__(self, obs, q):
            np.testing.assert_equal(obs["front"].shape, (4, 6, 3))
            return np.array([[1, 2], [3, 4]], np.float32)

    env = Env()
    captured = []
    monkeypatch.setattr(sim, "_make_sim", lambda *_: env)
    monkeypatch.setattr(sim, "use_tabletop_cgl", lambda _: False)
    monkeypatch.setattr(evaluate, "seed_policy", lambda *_: None)
    monkeypatch.setattr(evaluate, "render_for_queries", lambda _: nullcontext(captured.append))

    def prepare(st, *args, **kwargs):
        st.backend_selection = SimpleNamespace(backend="torch", requested="torch", reason="test")
        return Policy(), {"front": (6, 4)}, 92, True

    monkeypatch.setattr(evaluate, "prepare_policy", prepare)
    config = tmp_path / "config.yaml"
    config.write_text("""\
robot: {type: sim, env: {type: envhub, hub_path: lerobot/so101-tabletop-mujoco}}
cameras: {front: {type: robot, save_size: [6, 4]}}
run: {hz: 30}
""")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    out = tmp_path / "eval"
    main(["--config", str(config), "eval", "--policy", str(model), "--device", "cpu", "--mode", "chunk",
          "--episodes", "2", "--seed", "5", "--out", str(out), *(["--trace-only"] if trace_only else [])])
    stats = json.loads((out / "eval_000.json").read_text())
    assert (stats["successes"], stats["mean_steps"], env.seeds, env.closed) == (1, 1.5, [None, 5, 6], True)
    assert [row["truncated"] for row in stats["results"]] == [True, False]
    files = sorted(out.glob("rollout_*.npz"))
    assert len(files) == 2
    with np.load(files[0]) as data:
        np.testing.assert_allclose(data["t"], [0, 1 / 30])
        np.testing.assert_allclose(data["qpos"][:, 0], [0, 1])
        np.testing.assert_allclose(data["cmd"][:, 0], [1, 3])
        meta = json.loads(str(data["meta"]))
        assert meta["simulation"]["seed"] == 5
        if trace_only:
            assert "front_offsets" not in data.files and not meta["cameras"]
            assert captured == [True, False, False, True, False]
        else:
            assert data["front_offsets"].size == 3
    store = Store(out)
    assert store.index()[0]["result"] in {"success", "failure"}
    assert store.summary(files[0].stem)["cams"] == ([] if trace_only else ["front"])
    main(["health", "--data", str(out)])
    assert "1/2 successful trials" in capsys.readouterr().out
