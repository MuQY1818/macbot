import json
import threading
import time
from types import SimpleNamespace

import numpy as np

from macbot.data.chunks import align, switch_shifts, tracking_lag
from macbot.policy import executors
from macbot.policy.executors import AsyncChunk, SyncChunk
from macbot.policy.runner import stack_chunks


def ramp(n, start=0.0, step=1.0, dims=6):
    """A chunk moving every joint forward at a constant speed (no repeated actions)."""
    x = start + step * np.arange(n, dtype=float)
    return np.stack([x * (j + 1) for j in range(dims)], axis=1)


def test_align_finds_a_constructed_regression():
    cur = ramp(50)
    p = 20
    assert align(cur[p:], cur, p) == (0, 0.0)                    # new chunk continues the command
    s, r = align(cur[p - 3:p - 3 + 40], cur, p)                  # new chunk starts 3 steps behind
    assert s == 3 and r == 0.0
    s, _ = align(cur[p + 2:], cur, p)                            # ...or 2 steps ahead
    assert s == -2


def test_align_tie_rules_and_window():
    cur = np.zeros((30, 6))                                      # a static chunk: every shift fits equally
    assert align(np.zeros((30, 6)), cur, 10)[0] == 0
    cur = ramp(50)
    p = 20
    new = cur[p - 4:p - 4 + 40].copy()
    new[:6] = cur[p:p + 6]                                       # guided prefix sits on the current chunk
    assert align(new, cur, p, start=0)[0] in (0, 4)
    assert align(new, cur, p, start=6) == (4, 0.0)               # after the guided prefix the lag shows


def test_align_respects_chunk_end_and_nan_padding():
    cur = np.full((50, 6), np.nan)
    cur[:30] = ramp(30)
    assert align(ramp(30)[25:], cur, 25)[0] == 0
    assert align(ramp(10), np.full((10, 6), np.nan), 0) is None


def test_tracking_lag():
    t = np.arange(300) / 30
    cmd = np.stack([np.sin(t) * 40] * 6, axis=1)
    q = np.vstack([cmd[:1].repeat(3, 0), cmd[:-3]])
    assert tracking_lag(t, cmd, q, 30) == 3


class Pol:
    kind, H = "act", 20

    def __init__(self, delay=0.0):
        self.calls, self.delay = 0, delay

    def __call__(self, imgs, q, **_):
        time.sleep(self.delay)
        self.calls += 1
        return ramp(self.H, start=self.calls, dims=7)


def _observe():
    return {}, [1.0] * 7


def test_sync_chunk_records_switches():
    log = []
    ex = SyncChunk(Pol(), _observe, rate=30, log=lambda t, ms, **k: log.append(k), k=5)
    ex.target(time.time())
    ex.target(ex.t0 + 6 / 30)
    first, second = log
    assert first["chunk"].shape == (20, 7) and np.isnan(first["prev"]).all() and np.isnan(first["p"])
    assert np.isfinite(second["prev"]).all() and second["p"] >= 5 and second["entry"] == 0.0
    assert second["q_obs"].tolist() == [1.0] * 7


def test_async_chunk_records_switches():
    log, stop = [], threading.Event()
    ex = AsyncChunk(Pol(delay=0.01), _observe, 30, lambda t, ms, **k: log.append(k), stop=stop.is_set)
    ex.start()
    try:
        ex.reset(True)
        deadline = time.time() + 2
        while len(log) < 3 and time.time() < deadline:
            time.sleep(0.01)
    finally:
        stop.set()
        ex.join(1)
    assert len(log) >= 3
    rec = log[-1]
    assert rec["chunk"].shape == (20, 7) and np.isfinite(rec["prev"]).all()
    assert rec["p"] >= 0 and rec["entry"] >= 0


def test_stack_chunks_pads_ragged_rows():
    out = stack_chunks([np.ones((3, 7)), np.ones((5, 7))])
    assert out.shape == (2, 5, 7) and out.dtype == np.float32 and np.isnan(out[0, 3:]).all()
    assert stack_chunks([]).shape == (0,)


def test_switch_shifts_on_a_saved_rollout(tmp_path):
    cur = ramp(50, dims=7)
    news = [cur[17:57 - 10], cur[20:]]                           # 3 steps behind, then no regression
    chunk = stack_chunks(news)
    prev = stack_chunks([cur, cur])
    p = np.array([20.0, 20.0], np.float32)
    path = tmp_path / "rollout_000.npz"
    np.savez(path, ex_chunk=chunk, ex_prev=prev, ex_p=p, ex_delay=np.array([0.0, 0.0]),
             cmd=cur.astype(np.float32), meta=json.dumps({}))
    with np.load(path) as d:
        s, r = switch_shifts(d)
    assert s.tolist() == [3, 0] and np.allclose(r, 0, atol=1e-5)
    np.savez(path, cmd=cur.astype(np.float32), meta=json.dumps({}))
    with np.load(path) as d:
        assert len(switch_shifts(d)[0]) == 0                    # older rollouts without chunks


def test_async_query_every_spaces_queries(monkeypatch):
    now = 100.0
    requested, finished = [], []

    def clock():
        return now

    def advance(seconds):
        nonlocal now
        now += seconds

    def observe():
        requested.append(now)
        return _observe()

    def policy(imgs, q):
        advance(0.15 if len(requested) == 2 else 0.01)
        return ramp(20, dims=7)

    monkeypatch.setattr(executors, "time", SimpleNamespace(time=clock, sleep=lambda s: advance(s + 0.002)))
    policy.kind = "act"
    ex = AsyncChunk(policy, observe, 30, lambda *_args, **_kwargs: finished.append(now),
                    every=6, stop=lambda: len(finished) == 4)
    assert "query every 6 steps" in ex.describe()
    ex.reset(True)
    ex.run()
    gaps = np.diff(requested)
    assert (gaps >= 0.2 - 1e-9).all() and (gaps <= 0.202 + 1e-9).all()
    assert finished[2] - finished[1] < 0.18  # Faster inference can finish sooner without an early query.


def test_mlx_rtc_exponential_schedule():
    import pytest
    pytest.importorskip("mlx.core")
    from macbot.policy.mlx_pi05 import PI05MLX
    weights = {}
    for schedule in ("linear", "exp"):
        me = SimpleNamespace(rtc_beta=10.0, rtc_schedule=schedule)
        _, w = PI05MLX.rtc_target(me, np.zeros((40, 7)), 3, 20, (1, 50, 32))
        weights[schedule] = np.array(w)[0, :, 0]
    lin, exp = weights["linear"], weights["exp"]
    assert (lin[:3] == 1).all() and (exp[:3] == 1).all() and (lin[20:] == 0).all() and (exp[20:] == 0).all()
    ramp = lin[3:20]
    assert np.allclose(exp[3:20], ramp * np.expm1(ramp) / (np.e - 1), atol=1e-6)
    assert (exp[3:20] < lin[3:20]).all()
