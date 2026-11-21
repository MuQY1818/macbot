from types import SimpleNamespace

import pytest

from macbot.policy import backends


def test_auto_requires_exact_checkpoint_runtime_and_mode(monkeypatch, tmp_path):
    args = SimpleNamespace(device="mps", mode="rtc")
    metadata = {"type": "pi05"}
    (tmp_path / "config.json").write_text('{"type": "pi05"}')
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(backends.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(backends.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(backends, "find_spec", lambda _: object())
    monkeypatch.setattr(backends, "machine_model", lambda: "test-mac")
    monkeypatch.setattr(backends, "mlx_runtime_available", lambda: True)
    row = backends.Qualification(
        "mlx", backends.checkpoint_digest(tmp_path), "test-mac", frozenset({"rtc"}), True, (),
    )
    monkeypatch.setitem(backends.QUALIFIED, "pi05", (row,))
    assert backends.select_backend("auto", tmp_path, metadata, args).backend == "mlx"
    args.mode = "chunk"
    assert "no qualified chunk" in backends.select_backend("auto", tmp_path, metadata, args).reason
    args.mode = "rtc"
    monkeypatch.setattr(backends, "mlx_runtime_available", lambda: False)
    assert "could not access" in backends.select_backend("auto", tmp_path, metadata, args).reason
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    assert "do not match" in backends.select_backend("auto", tmp_path, metadata, args).reason


@pytest.mark.parametrize("options", [{"mlx_bits": 8}, {"mlx_time_grid": "quadratic"}, {"steps": 3}])
def test_auto_refuses_lossy_options(tmp_path, options):
    args = SimpleNamespace(device="mps", **options)
    with pytest.raises(SystemExit, match="[Ee]xplicit"):
        backends.select_backend("auto", tmp_path, {"type": "pi05", "num_inference_steps": 10}, args)
