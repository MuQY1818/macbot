"""How action chunks become a stream of joint targets.

Every executor exposes the same small interface, so new execution schemes plug in without touching the
control loop:

    ex.start()                    start background threads (if any)
    ex.reset(active)              drop all chunks; active=True starts inferring from the current observation
    ex.target(now) -> (a, id)     joint target for wall time ``now``, or (None, id) if nothing yet
    ex.peek(now) -> a | None      the planned target at a later time, without side effects (no inference)
    ex.err                        set to a message if inference failed (the loop pauses)

Executors receive ``observe() -> (images, q)`` and ``log(t_chunk, infer_ms, **extra)``; time runs at
``rate`` policy steps per second (fps x speed). Within a chunk, targets are linearly interpolated.

* ``SyncChunk``: infer, execute the first K steps, infer again (the robot waits during inference).
* ``Step``: give the policy a fresh observation on each policy tick and hold its selected action.
* ``Ensemble``: temporal ensembling as in ACT. A thread infers continuously; each target is the
  exp(m * age)-weighted mean of all chunks that cover the current time.
* ``AsyncChunk``: a thread infers while the current chunk keeps executing. The new chunk is aligned in
  time with the old one (step 0 of the new chunk = the step the old chunk was at when inference
  started). With ``guide=True`` (pi0.5 only) the unexecuted rest of the old chunk guides the new one
  (Real-Time Chunking, Black et al. 2025): ``delay`` = expected inference time in steps.
  ``every`` > 0 queries every ``every`` policy steps instead of as soon as the previous inference ends.
  Override ``accept()`` to decide whether and where to switch to a new chunk.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np


def switch_record(new, prev, p, entry, q) -> dict:
    """Chunk switch data for offline analysis of progress regression (saved as ``ex_*`` arrays).

    ``chunk``/``prev``: the new and the current chunk in executed joint space (``prev`` is NaN when there is none),
    ``p``: fractional index of ``prev`` that the new chunk's index 0 replaces, ``entry``: index of the new chunk at
    which it starts to be executed, ``q_obs``: the measured state the policy saw."""
    new = np.asarray(new, np.float32)
    old = np.full_like(new, np.nan)
    if prev is not None:
        prev = np.asarray(prev, np.float32)[:len(new)]
        old[:len(prev), :prev.shape[1]] = prev[:, :new.shape[1]]
    return {"chunk": new, "prev": old, "p": float("nan") if prev is None else float(p), "entry": float(entry),
            "q_obs": np.asarray(q, np.float32)}


def _interp(ch: np.ndarray, f: float) -> np.ndarray:
    if len(ch) == 1:
        return ch[0]
    f = min(max(f, 0.0), len(ch) - 1.0)
    i = min(int(f), len(ch) - 2)
    w = f - i
    return ch[i] * (1 - w) + ch[i + 1] * w


class Executor:
    name = "base"

    def __init__(self, policy, observe, rate: float, log):
        self.pol, self.observe, self.rate, self.log = policy, observe, rate, log
        self.err = None

    def start(self):
        pass

    def reset(self, active: bool):
        raise NotImplementedError

    def target(self, now: float):
        raise NotImplementedError

    def peek(self, now: float):
        """Planned target at wall time ``now`` (usually in the future) from the chunks already inferred."""
        return None

    def describe(self) -> str:
        return self.name


class SyncChunk(Executor):
    name = "chunk"

    def __init__(self, policy, observe, rate, log, k: int):
        super().__init__(policy, observe, rate, log)
        self.k = k
        self.chunk, self.t0, self.id = None, 0.0, -1

    def reset(self, active: bool):
        self.chunk = None

    def target(self, now):
        f = (now - self.t0) * self.rate if self.chunk is not None else None
        if self.chunk is None or f >= self.k:
            prev, p = self.chunk, f
            imgs, q = self.observe()
            t = time.perf_counter()
            try:
                self.chunk = self.pol(imgs, q)
            except Exception as e:
                self.err, self.chunk = repr(e), None
                return None, self.id
            ms = (time.perf_counter() - t) * 1e3
            self.t0, self.id, f = time.time(), self.id + 1, 0.0
            self.log(self.t0, ms, **switch_record(self.chunk, prev, p, 0.0, q))
        return _interp(self.chunk, f), self.id

    def peek(self, now):
        return None if self.chunk is None else _interp(self.chunk, (now - self.t0) * self.rate)

    def describe(self):
        return f"chunk (K={self.k}, synchronous)"


class Step(Executor):
    name = "step"

    def __init__(self, policy, observe, rate, log):
        super().__init__(policy, observe, rate, log)
        self.active, self.action, self.next_tick, self.id = False, None, 0.0, -1

    def reset(self, active: bool):
        self.pol.reset()
        self.active, self.action, self.next_tick = active, None, 0.0

    def target(self, now):
        if not self.active or now < self.next_tick:
            return self.action, self.id
        start = time.perf_counter()
        try:
            imgs, q = self.observe()
            self.action = self.pol(imgs, q)[0]
        except Exception as exc:
            self.err, self.action = repr(exc), None
            return None, self.id
        # Keep the phase of the policy clock, skipping missed ticks after a slow inference.
        period = 1.0 / self.rate
        deadline = self.next_tick or now
        self.next_tick = deadline + (math.floor((now - deadline) / period) + 1) * period
        self.id += 1
        self.log(time.time(), (time.perf_counter() - start) * 1e3)
        return self.action, self.id

    def peek(self, now):
        return self.action                                    # one action at a time: nothing planned ahead

    def describe(self):
        return "step (LeRobot select_action)"


class Ensemble(Executor, threading.Thread):
    name = "ensemble"
    start = threading.Thread.start      # Executor.start would shadow it in the MRO

    def __init__(self, policy, observe, rate, log, m: float = 0.01, max_hz: float = 30.0, stop=lambda: False):
        Executor.__init__(self, policy, observe, rate, log)
        threading.Thread.__init__(self, daemon=True)
        self.m, self.min_dt, self.stop = m, 1.0 / max_hz, stop
        self.lk = threading.Lock()
        self.buf, self.active, self.gen, self.n = [], False, 0, 0

    def reset(self, active: bool):
        with self.lk:
            self.buf, self.active = [], active
            self.gen += 1

    def target(self, now):
        with self.lk:
            live = [((now - t) * self.rate, ch) for t, ch in self.buf]
            self.buf = [b for b, (f, ch) in zip(self.buf, live, strict=False) if f < len(ch) - 1]
            n = self.n
        return self._blend(live), n - 1

    def peek(self, now):
        with self.lk:
            live = [((now - t) * self.rate, ch) for t, ch in self.buf]
        return self._blend(live)

    def _blend(self, live):
        live = [(f, ch) for f, ch in live if 0 <= f < len(ch) - 1]
        if not live:
            return None
        fs = np.array([f for f, _ in live])
        w = np.exp(self.m * (fs - fs.max()))                 # older chunks weigh more, as in ACT
        return sum(wi * _interp(ch, f) for wi, (f, ch) in zip(w, live, strict=False)) / w.sum()

    def run(self):
        while not self.stop():
            with self.lk:
                act, gen = self.active, self.gen
            if not act:
                time.sleep(0.005)
                continue
            t0 = time.perf_counter()
            try:
                imgs, q = self.observe()
                t_obs = time.time()
                chunk = self.pol(imgs, q)
            except Exception as e:
                with self.lk:
                    if gen == self.gen and self.active:
                        self.err = repr(e)
                        self.buf, self.active = [], False
                continue
            ms = (time.perf_counter() - t0) * 1e3
            with self.lk:
                if gen != self.gen or not self.active:
                    continue
                self.buf.append((t_obs, chunk))
                self.n += 1
            self.log(t_obs, ms)
            time.sleep(max(0.0, self.min_dt - (time.perf_counter() - t0)))

    def describe(self):
        return f"temporal ensemble (m={self.m})"


class AsyncChunk(Executor, threading.Thread):
    name = "async"
    start = threading.Thread.start

    def __init__(self, policy, observe, rate, log, guide: bool = False, horizon: int = 20, every: int = 0,
                 stop=lambda: False):
        Executor.__init__(self, policy, observe, rate, log)
        threading.Thread.__init__(self, daemon=True)
        self.guide, self.horizon, self.every, self.stop = guide, horizon, every, stop
        self.lk = threading.Lock()
        self.cur, self.active, self.gen, self.n = None, False, 0, 0
        self.lat = []                                         # recent inference times (s)
        self.last_req = None                                  # wall time of the previous query

    def reset(self, active: bool):
        with self.lk:
            self.cur, self.active = None, active
            self.gen += 1
            self.last_req = None

    def target(self, now):
        with self.lk:
            cur = self.cur
        if cur is None:
            return None, self.n - 1
        return _interp(cur["act"], (now - cur["T0"]) * self.rate), cur["id"]

    def peek(self, now):
        with self.lk:
            cur = self.cur
        return None if cur is None else _interp(cur["act"], (now - cur["T0"]) * self.rate)

    def accept(self, new_act, new_raw, cur, t_req, t_done, T_aligned):
        """Return the wall time of step 0 of the new chunk, or None to keep executing ``cur``.
        Default: always switch, time-aligned with the old chunk (first chunk starts now)."""
        return t_done if cur is None else T_aligned

    def run(self):
        while not self.stop():
            with self.lk:
                act, gen, cur = self.active, self.gen, self.cur
            if not act:
                time.sleep(0.005)
                continue
            if self.every and cur is not None and self.last_req is not None:
                wait = self.last_req + self.every / self.rate - time.time()
                if wait > 0:                                  # query every `every` steps, not continuously
                    time.sleep(min(wait, 0.005))
                    continue
            t_req = time.time()
            self.last_req = t_req
            leftover, delay, T_aligned = None, 0, t_req
            if cur is not None:
                i0 = int((t_req - cur["T0"]) * self.rate)
                est = max(self.lat[-5:]) * 1.1 if self.lat else 1.0
                delay = int(np.ceil(est * self.rate)) + 1
                left = cur["raw"][i0:] if cur["raw"] is not None else []
                if len(left) > delay:
                    leftover, T_aligned = (left if self.guide else None), cur["T0"] + i0 / self.rate
                else:                                         # old chunk almost done: align to the observation
                    delay = 0
            try:
                imgs, q = self.observe()
                if self.pol.kind == "pi05" or (self.guide and getattr(self.pol, "supports_rtc", False)):
                    a, raw = self.pol(imgs, q, leftover=leftover, delay=delay, horizon=self.horizon, return_raw=True)
                else:
                    a, raw = self.pol(imgs, q), None
            except Exception as e:
                with self.lk:
                    if gen == self.gen and self.active:
                        self.err = repr(e)
                        self.cur, self.active = None, False
                continue
            t_done = time.time()
            self.lat = (self.lat + [t_done - t_req])[-20:]
            T0 = self.accept(a, raw, cur, t_req, t_done, T_aligned)
            if T0 is None:
                self.log(None, (t_done - t_req) * 1e3, switched=0)
                continue
            with self.lk:
                if gen != self.gen or not self.active:
                    continue
                jump = 0.0                                    # arm target jump at the switch (deg)
                if self.cur is not None:
                    jump = float(np.abs(_interp(a, (t_done - T0) * self.rate)[:6]
                                        - _interp(self.cur["act"], (t_done - self.cur["T0"]) * self.rate)[:6]).max())
                prev = self.cur
                self.cur = {"act": a, "raw": raw, "T0": T0, "id": self.n}
                self.n += 1
            rec = switch_record(a, None if prev is None else prev["act"],
                                0.0 if prev is None else (T0 - prev["T0"]) * self.rate, (t_done - T0) * self.rate, q)
            self.log(T0, (t_done - t_req) * 1e3, switched=1, delay=delay, jump=round(jump, 3),
                     overrun=round((t_done - T0) * self.rate - delay, 2), **rec)

    def describe(self):
        name = "real-time chunking (guided)" if self.guide else "asynchronous chunks"
        return name + (f", query every {self.every} steps" if self.every else "")
