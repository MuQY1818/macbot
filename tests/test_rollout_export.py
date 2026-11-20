import json

import cv2
import numpy as np
import pytest

from macbot.cli import main
from macbot.config import load
from macbot.data.grasp import detect, settings
from macbot.data.rollouts import metadata, write_annotation
from macbot.viz.rollouts import Store


def test_grasp_miss_needs_a_reopened_and_calibrated_plateau():
    cfg = settings(load())
    meta = {"joint_names": ["gripper"], "fps": 30, "robot": {"kind": "rebot_b601"}}
    t = np.arange(210) / 30
    cmd = np.repeat([0, -120, 0, -120, 0, -120, 0], 30)[:, None]
    q = np.repeat([.1, -120, .1, -120, -9, -120, .1], 30)[:, None]
    report = detect(meta, t, cmd, q, cfg)
    assert report == {"misses": [{"first": 60, "end": 90, "t": 2, "until": 3}], "miss_count": 1, "unknown_closes": 1}
    assert detect(meta, t, cmd, q, {**cfg, "profile": {"held": 0, "empty": .1, "tolerance": .1}})["miss_count"] is None
    assert detect({**meta, "fake": True}, t, cmd, q, cfg)["miss_count"] is None


def fixture_trial(path, *, demo=False, legacy=False, result="failure"):
    n = 24
    t = np.arange(n) / 10
    arm = np.arange(n) * 0.5
    cmd = np.column_stack([arm, np.r_[np.full(6, -120), np.zeros(10), np.full(8, -120)]])
    q = cmd.copy()
    q[6:16, 1] = 0.1
    leader = np.column_stack([arm + 30, np.r_[np.full(8, -120), np.full(3, -30), np.full(13, -65)]])
    takeover = np.r_[np.zeros(6), np.full(2, 2), np.ones(5), np.zeros(11)].astype(np.uint8)
    gp = np.r_[np.zeros(6), np.ones(3), np.zeros(15)]
    gp[13:16] = 0.82
    cameras = {"top": {"size": [32, 24]}} if demo else ["top"]
    meta = {"fps": 10, "joint_names": ["pan", "gripper"], "gripper": "gripper", "cameras": cameras, "frames": n,
            "binary_gripper": {"open_deg": -120.0, "close_deg": 0.0},
            "grasp_miss_profile": {"held": -1.8, "empty": .1, "tolerance": .6},
            "camera_framing": {"top": {"save_size": [32, 24], "crop": [0, 0, 32, 24]}},
            "instruction": "Put the cube in the box.", "result": result, "hand_marks": [1.5], "fake": True}
    blob = cv2.imencode(".jpg", np.full((24, 32, 3), 80, np.uint8))[1].ravel()
    vals = {"action": cmd} if demo else {"cmd": cmd, "t": t, "gp": gp}
    if not (demo or legacy):
        vals.update(takeover=takeover, leader=leader)
    np.savez(path, qpos=q, meta=json.dumps(meta), top_jpeg=np.tile(blob, n), top_offsets=np.arange(n + 1) * len(blob), **vals)
    return meta


def test_exclusion_skips_health_and_updates_replay(tmp_path, capsys):
    p = tmp_path / "rollout_000.npz"
    fixture_trial(p)
    store = Store(tmp_path)
    assert store.index()[0]["miss_count"] == 1
    before = p.read_bytes()
    with pytest.raises(ValueError, match="requires a reason"):
        store.annotate(p.stem, {"result": "success", "exclude": True})
    store.annotate(p.stem, {"result": "success", "exclude": True, "exclude_reason": "GPU contention"})
    assert p.read_bytes() == before
    assert store.index()[0]["exclude"]
    summary = json.loads(store.summary_payload(p.stem))
    assert summary["meta"]["exclude_reason"] == "GPU contention"
    assert [s["kind"] for s in summary["takeover_spans"]] == [2, 1]
    assert summary["misses"][0]["t"] == pytest.approx(.6)
    assert metadata(p)["result"] == "success"
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    main(["--config", str(config), "health", "--data", str(tmp_path)])
    assert "000 excluded: GPU contention" in capsys.readouterr().out


def test_export_rollouts_loads_and_merges_with_lerobot(tmp_path, monkeypatch, capsys):
    package = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from lerobot.datasets.dataset_tools import merge_datasets

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    data, demos = tmp_path / "trials", tmp_path / "demos"
    data.mkdir()
    demos.mkdir()
    p = data / "rollout_001.npz"
    meta = fixture_trial(p)
    write_annotation(p, "success", "reviewed")
    fixture_trial(data / "rollout_002.npz", legacy=True)
    p = data / "rollout_003.npz"
    fixture_trial(p)
    write_annotation(p, "failure", "", True, "GPU contention")
    fixture_trial(demos / "episode_000.npz", demo=True)
    out, demo = tmp_path / "out", tmp_path / "demo"
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    for directory, dest, option in [(data, out, "--rollouts"), (demos, demo, "--with-rollout-labels")]:
        main(["--config", str(config), "export", option, "--data", str(directory), "--out", str(dest)])
    assert "1 successes" in capsys.readouterr().out
    ds = package.LeRobotDataset("local/rollouts", root=out, video_backend="pyav")
    demonstrations = package.LeRobotDataset("local/demo", root=demo, video_backend="pyav")
    assert ds.num_episodes == 2 and len(ds) == 48
    assert ds.features == demonstrations.features
    assert json.loads((out / "meta" / "camera_framing.json").read_text()) == meta["camera_framing"]
    assert (out / "meta" / "camera_framing.json").read_bytes() == (demo / "meta" / "camera_framing.json").read_bytes()
    assert ds.meta.episodes["source"][:] == ["rollout", "rollout"]
    assert ds.meta.episodes["success"][:] == [True, False]
    assert tuple(ds[8]["observation.images.top"].shape) == (3, 24, 32)
    for i, takeover, grip, miss, mark in [(6, 2, 1, 1, 0), (8, 1, 1, 1, 0), (11, 1, 1, 1, 1),
                                          (13, 0, 1, 1, 1), (17, 0, 0, 0, 1), (32, 0, 1, 1, 0)]:
        row = ds[i]
        assert [int(row[k]) for k in ("takeover", "grasp_miss", "hand_mark")] == [takeover, miss, mark]
        assert row["action"].tolist() == [(i % 24) * .5, grip]
    merged = merge_datasets([ds, demonstrations], "local/merged", tmp_path / "merged")
    assert merged.num_episodes == 3 and len(merged) == 72
    assert merged.meta.episodes["source"][:] == ["rollout", "rollout", "demonstration"]
