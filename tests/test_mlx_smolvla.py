import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from macbot.policy.mlx_smolvla import _EXPERT, _MODEL, _TEXT, _VISION, SmolNetwork  # noqa: E402


def test_smolvla_caches_cross_attention_and_predicts_a_chunk():
    rng = np.random.default_rng(802)
    weights = {}

    def param(name, shape, bias=False):
        weights[name + ".weight"] = mx.array(rng.normal(0, 0.05, shape).astype(np.float32))
        if bias:
            weights[name + ".bias"] = mx.zeros((shape[0],))

    for name, shape in (
        ("action_in_proj", (6, 3)), ("action_time_mlp_in", (6, 12)), ("action_time_mlp_out", (6, 6)),
        ("action_out_proj", (3, 6)), ("state_proj", (8, 3)),
    ):
        param(name, shape, True)
    param(_TEXT + "embed_tokens", (12, 8))
    param(_EXPERT + "norm", (6,))
    param(_VISION + "embeddings.patch_embedding", (64, 3, 16, 16), True)
    param(_VISION + "embeddings.position_embedding", (1024, 64))
    param(_VISION + "post_layernorm", (64,), True)
    param(_MODEL + "connector.modality_projection.proj", (8, 1024))
    for i in range(12):
        root = f"{_VISION}encoder.layers.{i}."
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            param(root + "self_attn." + name, (64, 64), True)
        for name in ("layer_norm1", "layer_norm2"):
            param(root + name, (64,), True)
        param(root + "mlp.fc1", (8, 64), True)
        param(root + "mlp.fc2", (64, 8), True)
    for stem, dim in ((_TEXT, 8), (_EXPERT, 6)):
        for i in range(16):
            root = f"{stem}layers.{i}."
            for name in ("input_layernorm", "post_attention_layernorm"):
                param(root + name, (dim,))
            width = 64 if stem == _EXPERT and i % 2 else dim
            for name, shape in (("q_proj", (64, dim)), ("k_proj", (64, width)),
                                ("v_proj", (64, width)), ("o_proj", (dim, 64))):
                param(root + "self_attn." + name, shape)
            for name, shape in (("gate_proj", (8, dim)), ("up_proj", (8, dim)), ("down_proj", (dim, 8))):
                param(root + "mlp." + name, shape)
    net = SmolNetwork(weights, mx.float32)
    vision = net.vision(mx.zeros((3, 512, 512, 3)))
    assert vision.shape == (3, 64, 8)
    tokens = mx.array([[1, 5]])
    a = net.prefix(vision, tokens, mx.zeros((1, 3)))
    b = net.prefix(vision, tokens, mx.ones((1, 3)))
    assert len(a) == len(b) == 16
    # The state is visible to the suffix but cannot change prefix image/text keys.
    np.testing.assert_allclose(np.array(a[1][0][:, :, :-1]), np.array(b[1][0][:, :, :-1]))
    assert not np.array_equal(np.array(a[1][0][:, :, -1]), np.array(b[1][0][:, :, -1]))
    actions = net.velocity(mx.zeros((1, 5, 3)), mx.ones((1, 6)), b)
    assert actions.shape == (1, 5, 3) and np.isfinite(np.array(actions)).all()
