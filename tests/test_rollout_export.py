import json

import cv2
import numpy as np
import pytest

from macbot.cli import main
from macbot.config import load
from macbot.data.export import segment_plan
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


def fixture_trial(path, *, demo=False, legacy=False, result="failure", takeover=None):
    n = 24
    t = np.arange(n) / 10
    arm = np.arange(n) * 0.5
    cmd = np.column_stack([arm, np.r_[np.full(6, -120), np.zeros(10), np.full(8, -120)]])
    q = cmd.copy()
    q[6:16, 1] = 0.1
    leader = np.column_stack([arm + 30, np.r_[np.full(8, -120), np.full(3, -30), np.full(13, -65)]])
    if takeover is None:
        takeover = np.repeat([0, 2, 1, 0], [6, 2, 5, 11]).astype(np.uint8)
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


def test_segment_plan_uses_trial_time_and_keeps_control_boundaries():
    takeover = np.repeat([0, 2, 1, 2, 1, 0, 1, 0, 2, 0], [20, 2, 10, 1, 3, 12, 10, 4, 2, 10])
    t = np.arange(len(takeover)) / 10
    t[3] -= .01  # just outside the first window despite being 17 frames before the request
    runs, pre, short = segment_plan(t, takeover, 10, 1.0, 1.7)
    assert short == 2
    assert runs == [
        (slice(0, 20), "rollout_policy", 0),
        (slice(22, 32), "rollout_human", 1),
        (slice(36, 48), "rollout_policy", 3),
        (slice(48, 58), "rollout_human", 4),
        (slice(64, 74), "rollout_policy", 6),
    ]
    # Image-sampled requests at frames 20, 48 and 62: alignment, direct handover and a cancelled takeover.
    assert np.flatnonzero(pre).tolist() == [*range(4, 20), *range(36, 48), *range(58, 62)]
    # Leader dropout and re-entry do not count as a fresh takeover.
    assert not segment_plan(t, takeover, 10, 1.0, 0)[1].any()
    runs, pre, _ = segment_plan(t, takeover, 10, 0, .1)
    assert (slice(33, 36), "rollout_human", 2) in runs
    assert np.flatnonzero(pre).tolist() == [19, 47, 61]


@pytest.mark.parametrize(("options", "error"), [
    (["--segments"], "--segments requires --rollouts"),
    (["--rollouts", "--min-segment-s", "0"], "require --segments"),
    (["--rollouts", "--segments", "--min-segment-s", "-1"], "--min-segment-s must be finite and nonnegative"),
    (["--rollouts", "--segments", "--pre-takeover-s", "nan"], "--pre-takeover-s must be finite and nonnegative"),
])
def test_segment_options_are_validated(tmp_path, options, error):
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    with pytest.raises(SystemExit, match=error):
        main(["--config", str(config), "export", *options])


def test_all_short_segments_leave_existing_output_intact(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    fixture_trial(tmp_path / "rollout_000.npz", takeover=np.repeat([0, 2, 1, 0], [6, 2, 7, 9]))
    out = tmp_path / "existing"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(SystemExit, match="nothing to export"):
        main(["--config", str(config), "export", "--rollouts", "--segments",
              "--data", str(tmp_path), "--out", str(out), "--overwrite"])
    assert marker.read_text() == "keep"
    assert "2 label 2 frames dropped, 3 short segments dropped" in capsys.readouterr().out


def test_export_segments_load_and_pad_action_chunks_with_lerobot(tmp_path, monkeypatch, capsys):
    package = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    p = tmp_path / "rollout_000.npz"
    fixture_trial(p)
    write_annotation(p, "success", "reviewed")
    fixture_trial(tmp_path / "rollout_001.npz", legacy=True)
    fixture_trial(tmp_path / "rollout_002.npz", takeover=np.repeat([0, 2], [1, 23]))
    excluded = tmp_path / "rollout_003.npz"
    fixture_trial(excluded)
    write_annotation(excluded, "failure", "", True, "excluded from training")
    config, out = tmp_path / "config.yaml", tmp_path / "segments"
    config.write_text("{}")
    main(["--config", str(config), "export", "--rollouts", "--segments", "--min-segment-s", ".5",
          "--pre-takeover-s", ".3", "--data", str(tmp_path), "--out", str(out)])
    log = capsys.readouterr().out
    assert ("3 trials, 3 rollout_policy segments, 1 rollout_human segments, 25 label 2 frames dropped, "
            "1 short segments dropped, 0.50 human seconds, 3 pre-takeover frames") in log
    assert "skipped rollout_003.npz: excluded" in log
    ds = package.LeRobotDataset("local/segments", root=out, video_backend="pyav",
                                delta_timestamps={"action": (np.arange(50) / 10).tolist()})
    assert ds.num_episodes == 4 and len(ds) == 46
    assert ds.features["pre_takeover"]["dtype"] == "uint8"
    episodes = ds.meta.episodes
    assert episodes["trial_name"][:] == ["rollout_000.npz"] * 3 + ["rollout_001.npz"]
    assert episodes["original_start_time"][:] == pytest.approx([0, .8, 1.3, 0])
    assert episodes["segment_index"][:] == [0, 1, 2, 0]
    assert episodes["source"][:] == ["rollout_policy", "rollout_human", "rollout_policy", "rollout_policy"]
    assert episodes["success"][:] == [True, True, True, False]
    offset = 0
    for ep, (start, end, takeover) in enumerate([(0, 6, 0), (8, 13, 1), (13, 24, 0), (0, 24, 0)]):
        assert episodes[ep]["dataset_from_index"] == offset
        for i, original in enumerate(range(start, end)):
            row = ds[offset + i]
            assert int(row["episode_index"]) == ep
            assert int(row["frame_index"]) == i
            assert float(row["timestamp"]) == pytest.approx(i / 10)
            assert int(row["takeover"]) == takeover
            assert int(row["pre_takeover"]) == (ep == 0 and 3 <= original <= 5)
            assert int(row["grasp_miss"]) == (6 <= original < 16)
            assert int(row["hand_mark"]) == (10 <= original <= 20)
            assert row["action"][:, 0].tolist() == [min(original + j, end - 1) * .5 for j in range(50)]
            assert row["action_is_pad"].tolist() == [original + j >= end for j in range(50)]
        offset += end - start
        assert episodes[ep]["dataset_to_index"] == offset
