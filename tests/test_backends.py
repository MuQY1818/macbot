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


def test_pi05_auto_maps_cameras_only_for_qualified_modes(monkeypatch, tmp_path):
    (tmp_path / "model.safetensors").touch()
    norm = {"norm_map": {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"}}
    for side, steps in (
        ("pre", [("rename_observations_processor", {}), ("to_batch_processor", {}),
                 ("relative_actions_processor", {"enabled": False}), ("normalizer_processor", norm),
                 ("pi05_prepare_state_tokenizer_processor_step", {}),
                 ("tokenizer_processor", {"padding_side": "right"}), ("device_processor", {})]),
        ("post", [("unnormalizer_processor", norm), ("device_processor", {})]),
    ):
        (tmp_path / f"policy_{side}processor.json").write_text(
            json.dumps({"steps": [{"registry_name": name, "config": config} for name, config in steps]}),
        )
    monkeypatch.setattr(backends.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(backends.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(backends, "find_spec", lambda _: object())
    monkeypatch.setattr(backends, "supported_mlx_dependencies", lambda: True)
    monkeypatch.setattr(backends, "mlx_runtime_available", lambda: True)
    meta = {"type": "pi05", "input_features": {
        f"observation.images.{name}": {"type": "VISUAL"} for name in ("top", "left_wrist_0_rgb")
    }}
    cfg = {"cameras": {"top": {}, "wrist": {}}}
    args = SimpleNamespace(device="mps", mode="chunk")
    for mode in ("chunk", "rtc", "auto"):
        args.mode = mode
        choice = backends.select_backend("auto", tmp_path, meta, args, config=cfg)
        assert choice.backend == "mlx" and choice.camera_map == {"top": "top", "wrist": "left_wrist_0_rgb"}
    assert backends.select_backend("mlx", tmp_path, meta, args, config=cfg).camera_map is None
    choice = backends.select_backend("auto", tmp_path, meta, args, config={"cameras": {"top": {}}})
    assert choice.backend == "torch" and "camera mapping" in choice.reason
    args.mode = "async"
    assert backends.select_backend("auto", tmp_path, meta, args, config=cfg).backend == "torch"


def test_pi0_qualification_requires_its_trained_processor_layout(tmp_path):
    (tmp_path / "model.safetensors").touch()
    norm = {"norm_map": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}}
    pre = [("rename_observations_processor", {}), ("to_batch_processor", {}), ("pi0_new_line_processor", {}),
           ("tokenizer_processor", {"padding_side": "right"}), ("device_processor", {}), ("normalizer_processor", norm)]
    post = [("unnormalizer_processor", norm), ("device_processor", {})]

    def write(side, steps):
        (tmp_path / f"policy_{side}processor.json").write_text(
            json.dumps({"steps": [{"registry_name": name, "config": config} for name, config in steps]}),
        )

    write("pre", pre)
    write("post", post)
    assert backends.compatible_pi(tmp_path, {"type": "pi0"})
    assert not backends.compatible_pi(tmp_path, {"type": "pi0", "use_relative_actions": True})
    post[0] = ("unnormalizer_processor", {"norm_map": {"ACTION": "QUANTILES"}})
    write("post", post)
    assert not backends.compatible_pi(tmp_path, {"type": "pi0"})


def test_pi05_camera_map_allows_masked_views():
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    views = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    metadata = {"input_features": {f"observation.images.{v}": {"type": "VISUAL"} for v in views}}
    two = {"cameras": {"top": {}, "wrist": {}}}
    assert backends.pi05_camera_map(two, metadata) == {"top": "base_0_rgb", "wrist": "left_wrist_0_rgb"}
    with pytest.raises(ValueError):
        backends.pi05_camera_map({"cameras": {"top": {}, "side": {}}}, metadata)
