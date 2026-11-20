from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from macbot.policy.mlx_pi05 import _EXPERT, _PREFIX, _VISION, _VLM, Geometry, PI05Network  # noqa: E402


@pytest.fixture
def tiny():
    g = Geometry(vision_depth=1, vision_heads=2, patch=2, text_depth=2, heads=2, kv_heads=1, head_dim=4)
    rng = np.random.default_rng(71)
    weights = {}

    def param(name, shape, bias=False):
        weights[name + ".weight"] = mx.array(rng.normal(0, 0.2, shape).astype(np.float32))
        if bias:
            weights[name + ".bias"] = mx.array(rng.normal(0, 0.1, shape[0]).astype(np.float32))

    v, p, a, d, h = 8, 8, 6, 3, 12
    param(_VISION + "embeddings.patch_embedding", (v, 3, 2, 2), True)
    param(_VISION + "embeddings.position_embedding", (4, v))
    param(_VISION + "post_layernorm", (v,), True)
    param(_VLM + "multi_modal_projector.linear", (p, v), True)
    param(_PREFIX + "embed_tokens", (13, p))
    root = _VISION + "encoder.layers.0."
    for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
        param(root + "self_attn." + proj, (v, v), True)
    for name in ("layer_norm1", "layer_norm2"):
        param(root + name, (v,), True)
    param(root + "mlp.fc1", (h, v), True)
    param(root + "mlp.fc2", (v, h), True)
    for stem, dim, adaptive in ((_PREFIX, p, False), (_EXPERT, a, True)):
        norms = [stem + "norm"]
        for i in range(g.text_depth):
            root = f"{stem}layers.{i}."
            for name, shape in (
                ("q_proj", (8, dim)), ("k_proj", (4, dim)), ("v_proj", (4, dim)), ("o_proj", (dim, 8)),
            ):
                param(root + "self_attn." + name, shape)
            for name, shape in (("gate_proj", (h, dim)), ("up_proj", (h, dim)), ("down_proj", (dim, h))):
                param(root + "mlp." + name, shape)
            norms += [root + "input_layernorm", root + "post_attention_layernorm"]
        for name in norms:
            if adaptive:
                param(name + ".dense", (3 * dim, dim), True)
            else:
                param(name, (dim,))
    for name, shape in (
        ("action_in_proj", (a, d)), ("action_out_proj", (d, a)), ("time_mlp_in", (a, a)), ("time_mlp_out", (a, a)),
    ):
        param(name, shape, True)
    images = mx.array(rng.normal(size=(2, 4, 4, 3)).astype(np.float32))
    noise = mx.array(rng.normal(size=(1, 5, d)).astype(np.float32))
    return g, weights, images, noise


def test_padded_tokens_do_not_contribute(tiny, tmp_path):
    g, weights, images, noise = tiny
    path = tmp_path / "tiny.safetensors"
    mx.save_safetensors(str(path), weights)
    net = PI05Network(mx.load(str(path)), g, dtype=mx.float32)
    vision = net.vision(images)
    assert vision.shape == (2, 4, 8)
    cond = net.time_condition(mx.array([1.0]), 0.004, 4.0)

    def predict(ids, mask, padded):
        valid, cache = net.prefix(vision, mx.array([ids]), mx.array([mask]), padded)
        pos, keep = net.expert_inputs(valid, noise.shape[1], masked=padded)
        return net.velocity(noise, cond, cache, pos, keep), cache

    a, padded = predict([1, 2, 9, 8], [True, True, False, False], True)
    b, trimmed = predict([1, 2], [True, True], False)
    np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-5)
    for (k, v), (k_trim, v_trim) in zip(padded, trimmed, strict=True):
        np.testing.assert_allclose(np.array(k[:, :, :10]), np.array(k_trim), atol=1e-5)
        np.testing.assert_allclose(np.array(v[:, :, :10]), np.array(v_trim), atol=1e-5)
    assert a.shape == noise.shape
    assert np.isfinite(np.array(a)).all()
    assert not np.allclose(np.array(a), 0)


def test_compiled_velocity_matches_eager(tiny):
    g, weights, images, noise = tiny
    net = PI05Network(weights, g, dtype=mx.float32)
    vision = net.vision(images)
    valid, cache = net.prefix(vision, mx.array([[1, 2]]), mx.array([[True, True]]), False)
    pos, keep = net.expert_inputs(valid, noise.shape[1], masked=False)
    cond = net.time_condition(mx.array([1.0]), 0.004, 4.0)
    args = (noise, cond, cache, pos, keep)
    assert keep is None
    np.testing.assert_allclose(np.array(mx.compile(net.velocity)(*args)), np.array(net.velocity(*args)), atol=1e-5)


def test_missing_checkpoint_weight_rejected(tiny):
    g, weights, *_ = tiny
    weights.pop("action_in_proj.bias")
    with pytest.raises(ValueError, match="action_in_proj.bias"):
        PI05Network(weights, g)


def test_gemma_embedding_is_scaled(tiny):
    g, weights, *_ = tiny
    net = PI05Network(weights, g, dtype=mx.float32)
    token = mx.array([[3]])
    table = weights[_PREFIX + "embed_tokens.weight"]
    np.testing.assert_allclose(np.array(net.embed_text(token)), np.array(table[3])[None, None] * np.sqrt(8), rtol=1e-6)


def test_packed_projections_and_precomputed_modulation(tiny):
    g, weights, images, noise = tiny
    plain = PI05Network(weights, g, dtype=mx.float32)
    packed = PI05Network(weights, g, dtype=mx.float32, pack_linears=True)
    tokens, mask = mx.array([[1, 2]]), mx.array([[True, True]])
    cond = plain.time_condition(mx.array([0.6]), 0.004, 4.0)

    def run(network, modulation=None):
        vision = network.vision(images)
        valid, cache = network.prefix(vision, tokens, mask, False)
        positions, keep = network.expert_inputs(valid, noise.shape[1], masked=False)
        return np.array(network.velocity(noise, cond, cache, positions, keep, modulation))

    np.testing.assert_allclose(run(packed, packed.adarms(cond)), run(plain), atol=1e-5)
