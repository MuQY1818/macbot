import hashlib
import io
import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import ZipFile

import cv2
import numpy as np

from macbot import web
from macbot.cli import main
from macbot.config import load
from macbot.policy.runner import State, make_handler
from macbot.viz.replay import make_handler as replay_handler
from macbot.viz.rollouts import Store, jpeg_frame


def trial(path, saved="2024-01-04 08:31:22", *, compressed=False, pad=0):
    data = []
    for color in (20, 80):
        ok, jpeg = cv2.imencode(".jpg", np.full((20, 32, 3), color, np.uint8))
        assert ok
        data.append(jpeg)
    q = np.arange(4, dtype=np.float32)[:, None]
    jpeg = np.concatenate([data[0], np.zeros(pad, np.uint8), data[1]])
    offsets = [0, len(data[0]), len(data[0]) + pad, len(jpeg), len(jpeg)] if pad else [0, len(data[0]), len(jpeg)]
    meta = {"fps": 30, "frames": 4, "result": "failure", "saved_at": saved, "cameras": ["top", "wrist"],
            "joint_names": ["joint_1"]}
    save = np.savez_compressed if compressed else np.savez
    save(path, meta=json.dumps(meta), t=[0., 0.03, 0.07, 0.15], qpos=q, cmd=q + 1,
         chunk_t=[0., 0.05], infer_ms=[4.2, 7.1],
         top_jpeg=jpeg, top_offsets=offsets, wrist_jpeg=jpeg, wrist_offsets=offsets)
    return [f.tobytes() for f in data]


@contextmanager
def serve(cfg, section, handler):
    with web.bind_console(cfg, section, handler, 0) as server:
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01))
        thread.start()
        try:
            yield server.url
        finally:
            server.shutdown()
            thread.join(timeout=2)


def request(url, path, payload=None):
    req = Request(url + path, data=None if payload is None else json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
    try:
        res = urlopen(req, timeout=3)
    except HTTPError as error:
        res = error
    with res:
        raw = res.read()
        return res.status, json.loads(raw) if res.headers.get_content_type() == "application/json" else raw


def test_stored_jpeg_reads_only_selected_range(tmp_path):
    path = tmp_path / "rollout_000.npz"
    first, last = trial(path, pad=8 * 1024 * 1024)

    class CountReads(io.BytesIO):
        count = 0

        def read(self, *args):
            result = super().read(*args)
            self.count += len(result)
            return result

    with CountReads(path.read_bytes()) as fp, ZipFile(fp) as z:
        assert jpeg_frame(fp, z, "top", 0) == first
        assert jpeg_frame(fp, z, "top", 2) == last
        assert fp.count < 32 * 1024


def test_review_is_shared_by_run_replay_and_health(tmp_path, capsys):
    path = tmp_path / "rollout_1001.npz"
    frames = trial(path, compressed=True)
    trial(tmp_path / "rollout_999.npz", saved="2024-01-05 08:31:22")
    trial(tmp_path / "rollout_000.tmp.npz")
    before = hashlib.sha256(path.read_bytes()).digest()
    cfg = load()
    st = State(cfg, tmp_path, SimpleNamespace(fake=True, speed=1.0, policy="fake"))
    with serve(cfg, "run", make_handler(st)) as url:
        status, data = request(url, "/api/rollouts")
        assert status == 200
        assert [r["name"] for r in data["items"]] == ["rollout_999", path.stem]
        assert data["items"][1]["saved_at"] == "2024-01-04 08:31:22"
        route = f"/api/rollout/{path.stem}"
        assert request(url, route + "/top/1") == (200, frames[1])
        assert request(url, route + "/top/2")[0] == 404
        assert request(url, route + "/annotation", {"result": "success", "note": "line\n"})[0] == 400
        review = {"result": "success", "note": "第三次夹取成功"}
        assert request(url, route + "/annotation", review) == (200, review)
        assert request(url, "/api/rollouts")[1]["items"][1]["result"] == "success"
    assert hashlib.sha256(path.read_bytes()).digest() == before
    with serve(cfg, "replay", replay_handler(Store(tmp_path), "fake")) as url:
        status, data = request(url, route)
        assert status == 200
        assert {k: data["meta"][k] for k in review} == review
        assert data["chunk_ms"] == [4.2, 7.1]
        assert data["sizes"]["top"] == [32, 20]
    main(["health", "--data", str(tmp_path)])
    assert "1001" in (report := capsys.readouterr().out)
    assert "success  第三次夹取成功" in report


def test_run_gates_all_replay_io_and_defers_motion_without_waiting(tmp_path, monkeypatch):
    path = tmp_path / "rollout_000.npz"
    trial(path)
    cfg = load()
    st = State(cfg, tmp_path, SimpleNamespace(fake=True, speed=1.0, policy="fake"))
    entered, release = threading.Event(), threading.Event()
    original = Store.frame

    def slow_frame(*args):
        entered.set()
        assert release.wait(timeout=3)
        return original(*args)

    monkeypatch.setattr(Store, "frame", slow_frame)
    with serve(cfg, "run", make_handler(st)) as url:
        route = "/api/rollout/rollout_000"
        response = []
        thread = threading.Thread(target=lambda: response.append(request(url, route + "/top/0")))
        thread.start()
        try:
            assert entered.wait(timeout=3)
            assert request(url, "/api/cmd", {"action": "start"})[1]["replay_ready"] is False
            assert st.take_command() is None
            assert not st.lock.locked()
            assert request(url, route + "/top/0") == (409, {"error": "pause_first"})
        finally:
            release.set()
            thread.join(timeout=3)
        assert response[0][0] == 200
        assert st.take_command() == "start"
        assert request(url, route)[0] == 409  # transition before the loop sets mode
        for mode in ("running", "homing"):
            st.mode = mode
            assert st.take_command() is None
            assert request(url, "/api/rollouts")[0] == 409
            assert request(url, route)[0] == 409
            assert request(url, route + "/top/0")[0] == 409
            assert request(url, route + "/annotation", {"result": "success"})[0] == 409
        st.mode = "paused"
        assert request(url, route)[0] == 200
        assert not path.with_suffix(".review.json").exists()
