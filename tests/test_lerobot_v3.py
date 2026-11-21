import io
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from macbot.cli import main
from macbot.data.lerobot_v3 import Dataset
from macbot.hub import HubRepo
from macbot.viz.datasets import DatasetStore
from macbot.viz.replay import make_handler

CAM = "observation.images.wrist-cam"
DATA = "data/chunk-000/file-000.parquet"
EPISODES = "meta/episodes/chunk-000/file-000.parquet"
VIDEO = f"videos/{CAM}/chunk-000/file-000.mp4"


@pytest.fixture
def dataset(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    for path in (DATA, EPISODES, VIDEO):
        (tmp_path / path).parent.mkdir(parents=True)
    info = {"codebase_version": "v3.0", "fps": 10, "total_episodes": 2,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {CAM: {"dtype": "video", "shape": [32, 48, 3], "names": ["height", "width", "channels"]},
                         "observation.state": {"names": ["shoulder.pos", "gripper.pos"]},
                         "action": {"names": ["gripper.pos", "shoulder.pos"]}}}
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(pa.table({"episode_index": [2, 7], "length": [3, 3], "tasks": [["one"], ["two"]],
                             "dataset_from_index": [0, 3], "dataset_to_index": [3, 6], "data/chunk_index": [0, 0],
                             "data/file_index": [0, 0], f"videos/{CAM}/chunk_index": [0, 0],
                             f"videos/{CAM}/file_index": [0, 0], f"videos/{CAM}/from_timestamp": [0., .3],
                             f"videos/{CAM}/to_timestamp": [.3, .6]}), tmp_path / EPISODES)
    pq.write_table(pa.table({"index": [0, 1, 2, 3, 5], "episode_index": [2, 2, 2, 7, 7],
                             "timestamp": [0., .1, .2, 0., .2], "frame_index": [0, 1, 2, 0, 2],
                             "observation.state": [[0., 1.], [1., 2.], [2., 3.], [3., 4.], [5., 6.]],
                             "action": [[5., 4.]] * 5}), tmp_path / DATA)
    return tmp_path


def request(handler_class, method, path):
    handler = handler_class.__new__(handler_class)
    handler.path, handler.request_version, handler.headers = path, "HTTP/1.0", {}
    handler.connection = SimpleNamespace(settimeout=lambda _: None)
    wire = io.BytesIO()
    handler.wfile = wire
    handler.log_request = lambda *_: None
    getattr(handler, f"do_{method}")()
    head, data = wire.getvalue().split(b"\r\n\r\n", 1)
    return int(head.split()[1]), data


def test_local_dataset_health_summary_and_read_only(dataset, capsys):
    files = {p.relative_to(dataset): p.read_bytes() for p in dataset.rglob("*") if p.is_file()}
    main(["health", "--data", str(dataset), "--split", "7"])
    out = capsys.readouterr().out
    assert "7       2     0.30s                1          1   200.0ms" in out
    assert "from 7: 1 episodes, 1 missing indices, 1 late gaps" in out

    handler = make_handler(DatasetStore(Dataset(dataset)), "<demo>")
    code, page = request(handler, "GET", "/")
    assert code == 200 and b"&lt;demo&gt;" in page and b'"dataset" === "dataset"' in page
    code, body = request(handler, "GET", "/api/rollouts")
    assert code == 200 and [r["name"] for r in json.loads(body)["items"]] == ["episode_002", "episode_007"]
    code, body = request(handler, "GET", "/api/rollout/episode_007")
    assert code == 200
    summary = json.loads(body)
    assert summary["q"] == [[3., 4.], [5., 6.]] and summary["cmd"] == [[4., 5.]] * 2
    assert summary["timing"]["missing"] == 1 and summary["t"] == [0., .2]
    assert summary["camera_labels"] == {"cam0": CAM}
    assert request(handler, "POST", "/api/rollout/episode_007/annotation")[0] == 405
    assert {p.relative_to(dataset): p.read_bytes() for p in dataset.rglob("*") if p.is_file()} == files


def test_hub_metadata_and_shared_video_offset(dataset, monkeypatch):
    video = cv2.VideoWriter(str(dataset / VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), 10, (48, 32))
    try:
        if not video.isOpened():
            pytest.skip("OpenCV build has no MP4 encoder")
        for channel in [2, 2, 2, 1, 1, 0]:
            image = np.zeros((32, 48, 3), np.uint8)
            image[..., channel] = 255
            video.write(image)
    finally:
        video.release()
    calls = []
    files = frozenset(p.relative_to(dataset).as_posix() for p in dataset.rglob("*") if p.is_file())

    def mock_open(ref, kind):
        assert ref == ("owner/recordings", "release") and kind == "dataset"
        return HubRepo(*ref, kind, "deadbeef", files)

    def mock_download(self, path):
        calls.append((self.sha, path))
        return dataset / path

    monkeypatch.setattr(HubRepo, "open", mock_open)
    monkeypatch.setattr(HubRepo, "download", mock_download)
    source = Dataset("owner/recordings@release")
    assert source.label == "owner/recordings@deadbeef"
    assert calls == [("deadbeef", "meta/info.json"), ("deadbeef", EPISODES)]
    store = DatasetStore(source)
    handler = make_handler(store, source.label)
    try:
        for ep, frame, channel in [(7, 0, 1), (7, 1, 0), (2, 0, 2)]:
            code, jpg = request(handler, "GET", f"/api/rollout/episode_{ep:03d}/cam0/{frame}")
            assert code == 200
            image = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            assert image.mean(axis=(0, 1)).argmax() == channel
        assert set(calls) == {("deadbeef", f) for f in files}
    finally:
        store.close()


def test_unsupported_dataset_version_is_actionable(tmp_path: Path):
    info = tmp_path / "meta/info.json"
    info.parent.mkdir()
    info.write_text('{"codebase_version":"v2.1"}')
    with pytest.raises(SystemExit, match=r"v2\.1.*Supported: LeRobot v3\.0"):
        main(["health", "--dataset", str(tmp_path)])
