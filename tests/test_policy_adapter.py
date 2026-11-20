import json

import numpy as np
import pytest

from macbot.cli import main
from macbot.policy.executors import Step
from macbot.policy.generic import GenericPolicy, _features
from macbot.policy.models import use_generic
from tools.tiny_policies import build


@pytest.fixture(scope="module", params=["act", "diffusion", "vqbet", "tdmpc"])
def tiny_policy(request, tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("lerobot.policies")
    kind = request.param
    if kind == "diffusion":
        pytest.importorskip("diffusers")
    path = tmp_path_factory.mktemp(kind)
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        build(path, kind)
        yield kind, path
    finally:
        torch.set_num_threads(prior_threads)


def test_generic_checkpoint_and_fake_run(tiny_policy, tmp_path, capsys):
    kind, path = tiny_policy
    camera_names = ("top", "wrist") if kind in {"act", "diffusion"} else ("top",)
    cameras = {name: [64, 64] for name in camera_names}
    camera_map = {"top": "observation.image"} if kind == "tdmpc" else {}
    adapter = GenericPolicy(path, "cpu", "pick up the cube", cameras=cameras, camera_map=camera_map, joint_count=6)
    assert adapter.stepwise == (kind != "act")
    out = adapter({name: np.zeros((64, 64, 3), np.uint8) for name in cameras}, [0.0] * 6)
    assert out.shape == (1 if adapter.stepwise else 3, 6)
    assert np.isfinite(out).all()

    data = tmp_path / "data"
    cfg = tmp_path / "config.json"  # JSON is also valid YAML.
    cfg.write_text(json.dumps({
        "data_root": str(data), "robot": {"type": "lerobot", "follower": {"type": "so101_follower"}},
        "cameras": {name: {"type": "uvc", "save_size": size} for name, size in cameras.items()},
        "policy": {"camera_map": camera_map}, "run": {"port": 0},
    }))
    common = ["--policy", str(path), "--policy-impl", "generic", "--device", "cpu"]
    main(["--config", str(cfg), "bench", *common, "--profile", "-n", "2"])
    assert "Synchronized stages" in capsys.readouterr().out
    main(["--config", str(cfg), "run", *common, "--fake", "--port", "0", "--autostart", "0.5"])
    rollout, = data.glob("rollouts/*/rollout_*.npz")
    with np.load(rollout) as r:
        assert len(r["infer_ms"]) > 0 and r["cmd"].shape[1] == 6
        assert json.loads(str(r["meta"]))["mode"] == ("step" if adapter.stepwise else "chunk")


def test_step_holds_actions_and_resets_history():
    class Policy:
        def reset(self):
            self.seen = []

        def __call__(self, imgs, q):
            self.seen.append(q)
            return np.array([q])

    policy = Policy()
    observations = iter(([1], [2], [3]))
    executor = Step(policy, lambda: ({}, next(observations)), 4, lambda *args: None)
    executor.reset(True)
    assert executor.target(1.0)[0].tolist() == [1]
    assert executor.target(1.2)[0].tolist() == [1]
    assert executor.target(2.0)[0].tolist() == [2]
    assert policy.seen == [[1], [2]]
    executor.reset(False)
    assert executor.target(3.0)[0] is None
    executor.reset(True)
    assert executor.target(3.0)[0].tolist() == [3]
    assert policy.seen == [[3]]


def test_adapter_feature_errors_and_explicit_camera_map(tmp_path):
    config = {
        "input_features": {"observation.images.camera1": {"type": "VISUAL", "shape": [3, 64, 64]}},
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
    }
    assert _features(tmp_path, config, {"top": [64, 64]}, {"top": "camera1"}, 6)[2] == {
        "observation.images.camera1": "top",
    }
    with pytest.raises(ValueError, match="Missing policy cameras"):
        _features(tmp_path, config, {"top": [64, 64]}, {}, 6)
    with pytest.raises(ValueError, match="end-effector"):
        _features(tmp_path, config | {"action_mode": "ee6d"}, {}, {}, 6)
    with pytest.raises(ValueError, match="robot has 7 joints"):
        _features(tmp_path, config, {}, {}, 7)


def test_stock_wrappers_remain_default_and_policy_catalog_works(capsys):
    assert not use_generic("act", "auto", "torch")
    assert not use_generic("pi05", "auto", "torch")
    assert not use_generic("pi05", "auto", "mlx")
    assert use_generic("act", "generic", "torch")
    assert use_generic("diffusion", "auto", "torch")
    main(["policies"])
    assert "pi0_fast" in capsys.readouterr().out
