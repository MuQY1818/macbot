"""Shared HTTP routes and bounded, on-demand access to rollout JPEGs."""
from __future__ import annotations

import json
import os
import re
import struct
import threading
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from datetime import datetime
from functools import lru_cache
from zipfile import ZIP_STORED, ZipFile

import cv2
import numpy as np

from ..data.rollouts import READ_ERRORS, ROLLOUT, array_header, metadata, read_array, rollout_files, write_annotation

MAX_FRAME = 4 * 1024 * 1024
MAX_CACHED_SUMMARY = 1024 * 1024
SUMMARY_CACHE_SIZE = 4
# A ZIP local header can have different extra fields from its central directory entry (e.g. ZIP64).
LOCAL_HEADER = struct.Struct("<4s22xHH")


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


def encode_summary(folder, name):
    return json.dumps(Store(folder).summary(name), ensure_ascii=False, allow_nan=False).encode()


class Store:
    def __init__(self, folder, *, trace_worker=None):
        self.folder = folder
        self.trace_worker = trace_worker
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

    def index(self):
        items = []
        for path in rollout_files(self.folder):
            try:
                meta = metadata(path)
                saved = meta.get("saved_at") or datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ", timespec="seconds")
                items.append({"name": path.stem, "result": meta.get("result"), "s": round(meta["frames"] / meta["fps"], 1),
                              "saved_at": saved, "mode": meta.get("mode", "")})
            except (*READ_ERRORS, ZeroDivisionError):
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
        return {"meta": meta, "cams": cams, "sizes": sizes, **traces, "chunk_t": np.round(ct, 3).tolist(),
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
        try:
            review = version_of(path.with_suffix(".review.json").stat())
        except FileNotFoundError:
            review = None
        key = (name, version_of(path.stat()), review)
        with self.summary_lock:
            if cached := self.summaries.get(key):
                self.summaries.move_to_end(key)
                return cached
            # The run console encodes cold traces in a separate process; large JSON responses
            # would otherwise hold the GIL shared by the paused control loop and HTTP threads.
            if self.trace_worker is not None:
                payload = self.trace_worker.submit(encode_summary, self.folder, name).result()
            else:
                payload = encode_summary(self.folder, name)
            if len(payload) <= MAX_CACHED_SUMMARY:
                self.summaries[key] = payload
                if len(self.summaries) > SUMMARY_CACHE_SIZE:
                    self.summaries.popitem(last=False)
            return payload

    def annotate(self, name, body):
        if not isinstance(body, dict):
            raise ValueError("Expected an annotation object")
        return write_annotation(self.path(name), body.get("result"), body.get("note", ""))


def serve_rollouts(handler, store, path, *, method="GET", lease=nullcontext):
    """Return whether the request belongs to replay. The lease never covers socket writes."""
    trial = re.fullmatch(r"/api/rollout/(rollout_\d+)", path) if method == "GET" else None
    frame = re.fullmatch(r"/api/rollout/(rollout_\d+)/(\w{1,80})/(\d{1,10})", path) if method == "GET" else None
    edit = re.fullmatch(r"/api/rollout/(rollout_\d+)/annotation", path) if method == "POST" else None
    index = method == "GET" and path == "/api/rollouts"
    if not (index or trial or frame or edit):
        return False
    if not store.slots.acquire(blocking=False):
        handler.json({"error": "replay_busy"}, 429)
        return True
    try:
        status, ctype = 200, "application/json"
        try:
            # Read small request bodies before taking the lease, so slow clients cannot defer Start.
            if edit:
                length = int(handler.headers.get("Content-Length", 0))
                if handler.headers.get_content_type() != "application/json" or not 0 < length <= 4096:
                    raise ValueError("Invalid annotation body")
                handler.connection.settimeout(2)
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
        handler.connection.settimeout(2)
        try:
            handler.send(status, response, ctype)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
    finally:
        store.slots.release()
    return True
