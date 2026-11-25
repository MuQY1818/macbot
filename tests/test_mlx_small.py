"""MLX ports of the small LeRobot policies against PyTorch on tiny random checkpoints.

Everything runs on the CPU (PyTorch and the MLX CPU stream), so the comparison is exact up to float32 rounding and
the test never touches the GPU. On the GPU, MLX float32 convolutions differ from PyTorch by about 1e-4 relative,
which is why these backends are explicit-only and not chosen by ``--backend auto``.
"""
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
pytest.importorskip("lerobot.policies.factory")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors  # noqa: E402

from macbot.policy.generic import GenericPolicy  # noqa: E402
from macbot.policy.models import load_policy  # noqa: E402

J = 6
TINY = {
    "vqbet": dict(
        n_obs_steps=2, n_action_pred_token=2, action_chunk_size=3, crop_shape=(48, 48),
        pretrained_backbone_weights=None, spatial_softmax_num_keypoints=4, vqvae_n_embed=8,
        vqvae_embedding_dim=16, vqvae_enc_hidden_dim=16, gpt_block_size=32, gpt_input_dim=32,
        gpt_output_dim=32, gpt_n_layer=1, gpt_n_head=4, gpt_hidden_dim=32,
    ),
    "diffusion": dict(
        horizon=8, n_action_steps=3, crop_shape=(48, 48), pretrained_backbone_weights=None,
        spatial_softmax_num_keypoints=4, down_dims=(16, 32), diffusion_step_embed_dim=16,
        noise_scheduler_type="DDIM", num_inference_steps=4,
    ),
    "tdmpc": dict(
        horizon=3, image_encoder_hidden_dim=8, state_encoder_hidden_dim=16, latent_dim=8, q_ensemble_size=2,
        mlp_dim=16, cem_iterations=2, n_gaussian_samples=16, n_pi_samples=2, n_elites=4,
    ),
}
CAMERAS = {
    "vqbet": {"top": "observation.images.top"},
    "diffusion": {"top": "observation.images.top", "wrist": "observation.images.wrist"},
    "tdmpc": {"top": "observation.image"},
}


@pytest.fixture
def cpu_stream():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def tiny_checkpoint(kind, root):
    torch.manual_seed(0)
    inputs = {key: PolicyFeature(FeatureType.VISUAL, (3, 64, 64)) for key in CAMERAS[kind].values()}
    inputs["observation.state"] = PolicyFeature(FeatureType.STATE, (J,))
    config = make_policy_config(kind, device="cpu", **TINY[kind])
    config.input_features = inputs
    config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (J,))}
    policy = get_policy_class(kind)(config)
    unit = {"min": -torch.ones(J), "max": torch.ones(J), "mean": torch.zeros(J), "std": torch.ones(J)}
    image = {name: torch.full((3, 1, 1), value) for name, value in (("min", 0.0), ("max", 1.0), ("mean", 0.0),
                                                                     ("std", 1.0))}
    stats = {"observation.state": unit, "action": unit} | {key: image for key in CAMERAS[kind].values()}
    pre, post = make_pre_post_processors(config, dataset_stats=stats)
    path = root / kind
    for item in (policy, pre, post):
        item.save_pretrained(path)
    return path


def observation(kind, seed=7):
    rng = np.random.default_rng(seed)
    imgs = {name: rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for name in CAMERAS[kind]}
    return imgs, rng.uniform(-0.5, 0.5, J).astype(np.float32).tolist()


def load_pair(kind, tmp_path):
    path = tiny_checkpoint(kind, tmp_path)
    io = dict(cameras={name: [64, 64] for name in CAMERAS[kind]}, camera_map=CAMERAS[kind], joint_count=J)
    return GenericPolicy(path, "cpu", "task", **io), load_policy(path, "mps", "task", backend="mlx", **io)


def torch_chunk(policy, imgs, q, *, argmax=False, **kwargs):
    policy.reset()
    batch = policy.pre(policy._observation(imgs, q))
    with ExitStack() as stack:
        stack.enter_context(torch.inference_mode())
        if argmax:
            stack.enter_context(patch.object(
                torch, "multinomial", lambda probs, num_samples, replacement=False, **_: probs.argmax(-1, keepdim=True)))
        first = policy.pol.select_action(dict(batch), **kwargs)
        rest = list(policy.pol._queues["action"])
    return policy.post(torch.cat([first, *rest]).unsqueeze(0)).float().numpy()[0]


def test_vqbet_argmax_chunk_matches_torch(tmp_path, cpu_stream):
    reference, port = load_pair("vqbet", tmp_path)
    imgs, q = observation("vqbet")
    expected = torch_chunk(reference, imgs, q, argmax=True)
    port.reset()
    actual = np.asarray(port(imgs, q, argmax=True))
    assert actual.shape == expected.shape == (3, J)
    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_diffusion_chunk_matches_torch_with_shared_noise(tmp_path, cpu_stream):
    reference, port = load_pair("diffusion", tmp_path)
    imgs, q = observation("diffusion")
    noise = np.random.default_rng(3).standard_normal((1, reference.pol.config.horizon, J), dtype=np.float32)
    expected = torch_chunk(reference, imgs, q, noise=torch.from_numpy(noise))
    port.reset()
    actual = np.asarray(port(imgs, q, noise=noise))
    assert actual.shape == expected.shape == (3, J)
    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_tdmpc_plans_a_finite_action(tmp_path, cpu_stream):
    # CEM planning samples its own noise, so only the data path and the output contract are checked here.
    _, port = load_pair("tdmpc", tmp_path)
    imgs, q = observation("tdmpc")
    port.reset()
    action = np.asarray(port(imgs, q))
    assert action.shape == (1, J) and np.isfinite(action).all()
