"""Shared HTTP routes and bounded, on-demand access to rollout JPEGs."""
from __future__ import annotations

import ctypes
import json
import os
import re
import socket
import struct
import sys
import threading
from collections import OrderedDict
from concurrent.futures import BrokenExecutor
from contextlib import contextmanager, nullcontext
from datetime import datetime
from functools import lru_cache
from zipfile import ZIP_STORED, ZipFile

import cv2
import numpy as np

from ..config import load
from ..data.grasp import from_archive, settings
from ..data.rollouts import READ_ERRORS, ROLLOUT, array_header, metadata, read_array, rollout_files, write_annotation

MAX_FRAME = 4 * 1024 * 1024
MAX_CACHED_SUMMARY = 1024 * 1024
SUMMARY_CACHE_SIZE = 4
# A ZIP local header can have different extra fields from its central directory entry (e.g. ZIP64).
LOCAL_HEADER = struct.Struct("<4s22xHH")
if sys.platform == "darwin":
    _native = ctypes.CDLL(None)
    _get_qos = _native.qos_class_self
    _set_qos = _native.pthread_set_qos_class_self_np
else:
    _get_qos = _set_qos = None
QOS_BACKGROUND = 0x09
QOS_DEFAULT = 0x15


def version_of(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def stored_start(fp, info, start):
    fp.seek(info.header_offset)
    sig, filename_len, extra_len = LOCAL_HEADER.unpack(fp.read(LOCAL_HEADER.size))
    if sig != b"PK\x03\x04":
        raise ValueError("Invalid ZIP header")
    return info.header_offset + LOCAL_HEADER.size + filename_len + extra_len + start


@contextmanager
def byte_array(fp, archive, key):
    """Yield array shape, dtype, and a range reader. Stored members support direct seeks."""
    info = archive.getinfo(f"{key}.npy")
    with archive.open(info) as member:
        shape, _, dtype = array_header(member, info.file_size)
        start = member.tell()
        if len(shape) != 1:
            raise ValueError("Invalid rollout camera array")

        if info.compress_type == ZIP_STORED:
            base, stream = stored_start(fp, info, start), fp
        else:
            # Older externally compressed trials still work. ZipExtFile seeks with bounded
            # buffers, though it may have to decompress the prefix of the member.
            base, stream = start, member

        def read(offset, length):
            if not 0 <= offset < offset + length <= info.file_size - start:
                raise ValueError("Invalid rollout frame range")
            stream.seek(base + offset)
            data = stream.read(length)
            if len(data) != length:
                raise ValueError("Truncated rollout frame")
            return data

        yield shape[0], dtype, read


def jpeg_frame(fp, archive, camera, frame):
    with byte_array(fp, archive, f"{camera}_offsets") as (n, dtype, read):
        if dtype.kind not in "iu" or not 0 <= frame < n - 1:
            raise KeyError("Unknown frame")
        first, last = map(int, np.frombuffer(read(frame * dtype.itemsize, 2 * dtype.itemsize), dtype=dtype))
    if not 0 < last - first <= MAX_FRAME:
        raise ValueError("Invalid rollout frame size")
    with byte_array(fp, archive, f"{camera}_jpeg") as (_, dtype, read):
        if dtype != np.uint8:
            raise ValueError("Invalid rollout JPEG type")
        return read(first, last - first)


@lru_cache(maxsize=16)
def stored_camera(path, version, camera):
    """Cache member locations and dtypes, never JPEG data or open file handles."""
    with path.open("rb") as fp:
        if version_of(os.fstat(fp.fileno())) != version:
            return None
        with ZipFile(fp) as archive:
            parts = []
            for key in ("offsets", "jpeg"):
                info = archive.getinfo(f"{camera}_{key}.npy")
                if info.compress_type != ZIP_STORED:
                    return None
                with archive.open(info) as member:
                    shape, _, dtype = array_header(member, info.file_size)
                    if len(shape) != 1:
                        raise ValueError("Invalid rollout camera array")
                    parts.append((stored_start(fp, info, member.tell()), shape[0], dtype))
            return tuple(parts)


def read_stored_frame(fp, camera, frame):
    (offset_base, n, dtype), (jpeg_base, jpeg_size, jpeg_dtype) = camera
    if dtype.kind not in "iu" or not 0 <= frame < n - 1:
        raise KeyError("Unknown frame")
    fp.seek(offset_base + frame * dtype.itemsize)
    first, last = map(int, np.frombuffer(fp.read(2 * dtype.itemsize), dtype=dtype))
    if jpeg_dtype != np.uint8 or not (0 <= first < last <= jpeg_size and last - first <= MAX_FRAME):
        raise ValueError("Invalid rollout JPEG range")
    fp.seek(jpeg_base + first)
    data = fp.read(last - first)
    if len(data) != last - first:
        raise ValueError("Truncated rollout frame")
    return data


def encode_summary(folder, name, grasp_cfg):
    return json.dumps(Store(folder, grasp_cfg=grasp_cfg).summary(name), ensure_ascii=False, allow_nan=False).encode()


def review_version(path):
    try:
        return version_of(path.with_suffix(".review.json").stat())
    except FileNotFoundError:
        return None


def takeover_spans(z, t, fps):
    if "takeover.npy" not in z.namelist():
        return []
    state = read_array(z, "takeover")
    if state.shape != t.shape:
        raise ValueError("Invalid takeover array")
    gaps = np.diff(t) > 2.5 / fps
    edges = np.r_[0, np.flatnonzero((state[1:] != state[:-1]) | gaps) + 1, len(t)]
    return [{"kind": int(state[first]), "t": float(t[first]),
             "until": float(t[end] if end < len(t) and not gaps[end - 1] else t[end - 1] + 1 / fps)}
            for first, end in zip(edges, edges[1:], strict=False) if end > first and state[first] in (1, 2)]


def read_index_item(path, grasp_cfg):
    with ZipFile(path) as archive:
        meta = metadata(path, archive)
        report = from_archive(archive, meta, grasp_cfg)
    saved = meta.get("saved_at") or datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ", timespec="seconds")
    return {"name": path.stem, "result": meta.get("result"), "s": round(meta["frames"] / meta["fps"], 1),
            "saved_at": saved, "mode": meta.get("mode", ""), "exclude": bool(meta.get("exclude")),
            "miss_count": report["miss_count"], "unknown_closes": report["unknown_closes"]}


def index_item(path, grasp_cfg):
    try:
        return read_index_item(path, grasp_cfg)
    except (*READ_ERRORS, ZeroDivisionError):
        return None


class Store:
    def __init__(self, folder, *, trace_worker=None, grasp_cfg=None):
        self.folder = folder
        self.trace_worker = trace_worker
        self.grasp_cfg = grasp_cfg if grasp_cfg is not None else settings(load())
        self.index_item = lru_cache(maxsize=256)(self._index_item)
        self.slots = threading.BoundedSemaphore(4)
        self.summary_lock = threading.Lock()
        self.summaries = OrderedDict()

    def path(self, name):
        if not ROLLOUT.fullmatch(name):
            raise FileNotFoundError(name)
        path = self.folder / f"{name}.npz"
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(name)
        return path

    def _index_item(self, path, version, review):
        if self.trace_worker is not None:
            return self.trace_worker.submit(index_item, path, self.grasp_cfg).result()
        return index_item(path, self.grasp_cfg)

    def index(self):
        items = []
        for path in rollout_files(self.folder):
            try:
                if item := self.index_item(path, version_of(path.stat()), review_version(path)):
                    items.append(item)
            except FileNotFoundError:
                continue
        return sorted(items, key=lambda r: (r["saved_at"], int(r["name"].split("_")[-1])), reverse=True)

    def summary(self, name):
        path = self.path(name)
        with path.open("rb") as fp, ZipFile(fp) as z:
            meta = metadata(path, z)
            keys = set(z.namelist())
            cams = meta.get("cameras") or [k[:-9] for k in z.namelist() if k.endswith("_jpeg.npy")]
            sizes = {}
            framing = meta.get("camera_framing") or {}
            for c in cams:
                sizes[c] = framing.get(c, {}).get("save_size")
                if not sizes[c]:
                    image = cv2.imdecode(np.frombuffer(jpeg_frame(fp, z, c, 0), np.uint8), cv2.IMREAD_COLOR)
                    if image is not None:
                        sizes[c] = [image.shape[1], image.shape[0]]
            traces = {key: np.round(read_array(z, member).astype(float), precision).tolist()
                      for key, member, precision in (("t", "t", 3), ("q", "qpos", 2), ("cmd", "cmd", 2))}
            ct = read_array(z, "chunk_t") if "chunk_t.npy" in keys else np.zeros(0)
            ims = read_array(z, "infer_ms") if "infer_ms.npy" in keys else np.zeros(0)
            spans = takeover_spans(z, np.asarray(traces["t"]), meta["fps"])
            report = from_archive(z, meta, self.grasp_cfg)
        return {"meta": meta, "cams": cams, "sizes": sizes, **traces, "chunk_t": np.round(ct, 3).tolist(),
                "takeover_spans": spans, **report,
                "chunk_ms": np.round(ims.astype(float), 1).tolist(),
                "infer_ms": round(float(np.median(ims)), 1) if len(ims) else None, "chunks": len(ct)}

    def frame(self, name, camera, frame):
        path = self.path(name)
        with path.open("rb") as fp:
            if layout := stored_camera(path, version_of(os.fstat(fp.fileno())), camera):
                return read_stored_frame(fp, layout, frame)
            with ZipFile(fp) as z:
                return jpeg_frame(fp, z, camera, frame)

    def summary_payload(self, name):
        path = self.path(name)
        key = (name, version_of(path.stat()), review_version(path))
        with self.summary_lock:
            if cached := self.summaries.get(key):
                self.summaries.move_to_end(key)
                return cached
            # The run console encodes cold traces in a separate process; large JSON responses
            # would otherwise hold the GIL shared by the paused control loop and HTTP threads.
            if self.trace_worker is not None:
                payload = self.trace_worker.submit(encode_summary, self.folder, name, self.grasp_cfg).result()
            else:
                payload = encode_summary(self.folder, name, self.grasp_cfg)
            if len(payload) <= MAX_CACHED_SUMMARY:
                self.summaries[key] = payload
                if len(self.summaries) > SUMMARY_CACHE_SIZE:
                    self.summaries.popitem(last=False)
            return payload

    def annotate(self, name, body):
        if not isinstance(body, dict):
            raise ValueError("Expected an annotation object")
        return write_annotation(self.path(name), body.get("result"), body.get("note", ""),
                                body.get("exclude"), body.get("exclude_reason", ""))


def serve_rollouts(handler, store, path, *, method="GET", lease=nullcontext):
    """Return whether the request belongs to replay. The lease never covers socket writes."""
    kind = "episode" if getattr(store, "readonly", False) else "rollout"
    trial = re.fullmatch(rf"/api/rollout/({kind}_\d+)", path) if method == "GET" else None
    frame = re.fullmatch(rf"/api/rollout/({kind}_\d+)/(\w{{1,80}})/(\d{{1,10}})", path) if method == "GET" else None
    edit = re.fullmatch(r"/api/rollout/(rollout_\d+)/annotation", path) if method == "POST" else None
    index = method == "GET" and path == "/api/rollouts"
    if not (index or trial or frame or edit):
        return False
    # Reuse replay GET connections so scrubbing does not start a thread per camera frame.
    # The response has a Content-Length; other console endpoints keep their existing protocol.
    version = handler.protocol_version
    if (method == "GET" and handler.request_version == "HTTP/1.1"
            and handler.headers.get("Connection", "").lower() != "close"
            and handler.headers.get("Content-Length") in (None, "0") and "Transfer-Encoding" not in handler.headers):
        handler.protocol_version = "HTTP/1.1"
        handler.close_connection = False
        handler.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    handler.connection.settimeout(2)
    acquired = store.slots.acquire(blocking=False)
    prior_qos = None
    if acquired and _get_qos is not None and lease is not nullcontext:
        # Only the run console competes with the hold loop. Restore the thread's
        # priority before it can handle another URL on the same connection.
        previous = _get_qos()
        if _set_qos(QOS_BACKGROUND, 0) == 0:
            prior_qos = previous or QOS_DEFAULT
    try:
        if not acquired:
            handler.json({"error": "replay_busy"}, 429)
            return True
        status, ctype = 200, "application/json"
        try:
            # Read small request bodies before taking the lease, so slow clients cannot defer Start.
            if edit:
                length = int(handler.headers.get("Content-Length", 0))
                if handler.headers.get_content_type() != "application/json" or not 0 < length <= 4096:
                    raise ValueError("Invalid annotation body")
                payload = handler.body_json()
            with lease() as allowed:
                if allowed is False:
                    status, response = 409, {"error": "pause_first"}
                elif index:
                    response = {"folder": str(store.folder), "items": store.index()}
                elif trial:
                    response = store.summary_payload(trial[1])
                elif frame:
                    response = store.frame(frame[1], frame[2], int(frame[3]))
                    ctype = "image/jpeg"
                else:
                    response = store.annotate(edit[1], payload)
                if ctype == "application/json" and not isinstance(response, bytes):
                    response = json.dumps(response, ensure_ascii=False, allow_nan=False).encode()
        except (FileNotFoundError, KeyError):
            status, response = 404, b'{"error":"replay_missing"}'
        except (*READ_ERRORS, cv2.error, struct.error):
            status, response = (400 if edit else 422), b'{"error":"review_failed"}' if edit else b'{"error":"replay_failed"}'
        except BrokenExecutor:
            status, response = 503, b'{"error":"replay_failed"}'
        handler.send(status, response, ctype)
    except (BrokenPipeError, ConnectionResetError, TimeoutError):
        handler.close_connection = True
    finally:
        if acquired:
            store.slots.release()
        if prior_qos is not None:
            _set_qos(prior_qos, 0)
        handler.protocol_version = version
    return True
