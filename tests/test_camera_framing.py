import json
from pathlib import Path

import pytest

from macbot.cameras.framing import FRAMING_FILE, dataset_framing, framing, training_framing
from macbot.cli import main
from macbot.config import load


def test_dataset_framing_preserves_known_legacy_values_and_rejects_mixed_modes():
    example = framing(load()["cameras"], 30)
    assert example["top"]["mode"]["preset"] == "1080p-if-available"
    assert example["top"]["crop"] == [625, 140, 1610, 1080]
    first = (Path("episode_000.npz"), {"camera_framing": example})
    legacy = {"cameras": {role: {"size": spec["save_size"], "crop": spec["crop"]} for role, spec in example.items()}}
    assert dataset_framing([first]) == example
    merged = dataset_framing([first, (Path("episode_001.npz"), legacy)])
    assert merged["top"] == {"crop": [625, 140, 1610, 1080], "save_size": [492, 470]}
    changed = example | {"top": example["top"] | {"crop": None}}
    with pytest.raises(SystemExit, match="camera 'top' crop differs"):
        dataset_framing([first, (Path("episode_002.npz"), {"camera_framing": changed})])


def test_run_checks_dataset_or_checkpoint_framing_before_opening_robot(tmp_path, monkeypatch, capsys):
    from macbot.policy import runner

    metadata = tmp_path / "dataset" / "meta"
    metadata.mkdir(parents=True)
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("cameras:\n  top: {type: uvc, unique_id: camera, expect_size: [1280, 720], save_size: [640, 360]}\n")
    expected = {"top": framing(load()["cameras"], 30)["top"]}
    (ckpt / "train_config.json").write_text(json.dumps({"dataset": {"root": "../dataset"}}))
    (metadata / FRAMING_FILE).write_text(json.dumps(expected))
    (ckpt / FRAMING_FILE).write_text(json.dumps({"top": {"crop": None}}))
    assert training_framing(ckpt, tmp_path) == {"top": {"crop": None}}
    (ckpt / FRAMING_FILE).unlink()
    calls = []
    monkeypatch.setattr(runner, "start_log", lambda *_: None)
    monkeypatch.setattr(runner, "ensure_dir", lambda path: path)
    monkeypatch.setattr(runner, "_control", lambda *_: calls.append("control"))
    cmd = ["--config", str(config), "run", "--fake", "--policy", str(ckpt), "--port", "0"]
    with pytest.raises(SystemExit, match=r"Camera 'top' framing mismatch: trained .*current .*--allow-framing-mismatch"):
        main(cmd)
    assert calls == []
    main([*cmd, "--allow-framing-mismatch"])
    assert calls == ["control"] and "Warning: Camera 'top'" in capsys.readouterr().out
    (metadata / FRAMING_FILE).unlink()
    main(cmd)
    assert capsys.readouterr().out.count("No training camera framing recorded; current framing:") == 1
