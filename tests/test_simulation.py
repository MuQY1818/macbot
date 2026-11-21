"""Simulation batch contracts without MuJoCo, a checkpoint download, or a GPU."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from macbot.cli import main
from macbot.policy import evaluate
from macbot.robot import sim
from macbot.simulation import summary
from macbot.viz.rollouts import Store


def test_wilson_endpoints():
    assert summary([{"success": False, "steps": 5}] * 50, 30)["wilson95"] == pytest.approx([0, 0.07135], abs=1e-5)
    assert summary([{"success": True, "steps": 5}] * 50, 30)["wilson95"] == pytest.approx([0.92865, 1], abs=1e-5)


def test_eval_refuses_hardware_before_resolving_policy():
    with pytest.raises(SystemExit, match="eval requires robot.type: sim"):
        main(["--config", "configs/so101.yaml", "eval", "--policy", "unused"])


def test_eval_saves_seeded_normal_rollouts(tmp_path, monkeypatch, capsys):
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
    monkeypatch.setattr(sim, "_make_sim", lambda *_: env)
    monkeypatch.setattr(evaluate, "seed_policy", lambda *_: None)

    def prepare(st, *args, **kwargs):
        st.backend_selection = SimpleNamespace(backend="torch", requested="torch", reason="test")
        return Policy(), {"front": (6, 4)}, 92, True

    monkeypatch.setattr(evaluate, "prepare_policy", prepare)
    config = tmp_path / "config.yaml"
    config.write_text("""\
robot: {type: sim, env: {type: fake}}
cameras: {front: {type: robot, save_size: [6, 4]}}
run: {hz: 30}
""")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    out = tmp_path / "eval"
    main(["--config", str(config), "eval", "--policy", str(model), "--device", "cpu", "--mode", "chunk",
          "--episodes", "2", "--seed", "5", "--out", str(out)])
    stats = json.loads((out / "eval_000.json").read_text())
    assert (stats["successes"], stats["mean_steps"], env.seeds, env.closed) == (1, 1.5, [None, 5, 6], True)
    assert [row["truncated"] for row in stats["results"]] == [True, False]
    files = sorted(out.glob("rollout_*.npz"))
    assert len(files) == 2
    with np.load(files[0]) as data:
        np.testing.assert_allclose(data["t"], [0, 1 / 30])
        np.testing.assert_allclose(data["qpos"][:, 0], [0, 1])
        np.testing.assert_allclose(data["cmd"][:, 0], [1, 3])
        assert data["front_offsets"].size == 3
        assert json.loads(str(data["meta"]))["simulation"]["seed"] == 5
    assert Store(out).index()[0]["result"] in {"success", "failure"}
    main(["health", "--data", str(out)])
    assert "1/2 successful trials" in capsys.readouterr().out
