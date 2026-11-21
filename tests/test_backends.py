import json
from types import SimpleNamespace

import pytest

from macbot.policy import backends


def test_auto_uses_checkpoint_layout_without_a_machine_whitelist(monkeypatch, tmp_path):
    args = SimpleNamespace(device="mps", mode="chunk")
    metadata = {"type": "smolvla"}
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    pre = ("rename_observations_processor", "to_batch_processor", "smolvla_new_line_processor",
           "tokenizer_processor", "device_processor", "normalizer_processor")
    post = ("unnormalizer_processor", "device_processor")
    for side, layout in (("pre", pre), ("post", post)):
        (tmp_path / f"policy_{side}processor.json").write_text(
            json.dumps({"steps": [{"registry_name": key} for key in layout]}),
        )
    monkeypatch.setattr(backends.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(backends.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(backends, "find_spec", lambda _: object())
    monkeypatch.setattr(backends, "supported_mlx_dependencies", lambda: True)
    monkeypatch.setattr(backends, "mlx_runtime_available", lambda: True)
    decision = backends.select_backend("auto", tmp_path, metadata, args)
    assert decision.backend == "mlx" and "lerobot/smolvla_base" in decision.reason
    args.mode = "async"
    assert "no qualified async" in backends.select_backend("auto", tmp_path, metadata, args).reason
    args.mode = "chunk"
    monkeypatch.setattr(backends, "mlx_runtime_available", lambda: False)
    assert "could not access" in backends.select_backend("auto", tmp_path, metadata, args).reason
    (tmp_path / "policy_preprocessor.json").write_text('{"steps": []}')
    fallback = backends.select_backend("auto", tmp_path, metadata, args)
    assert fallback.backend == "torch" and "checkpoint layout" in fallback.reason


@pytest.mark.parametrize("options", [{"mlx_bits": 8}, {"mlx_time_grid": "quadratic"}, {"steps": 3}])
def test_auto_refuses_lossy_options(tmp_path, options):
    args = SimpleNamespace(device="mps", **options)
    with pytest.raises(SystemExit, match="[Ee]xplicit"):
        backends.select_backend("auto", tmp_path, {"type": "pi05", "num_inference_steps": 10}, args)
