"""Read trial metadata and review annotations without loading recorded frames."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile, ZipFile

import numpy as np

from ..system import give_back

ROLLOUT = re.compile(r"rollout_\d+")
RESULTS = ("success", "failure")
NOTE_LIMIT = 400
READ_ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError, EOFError, BadZipFile, NotImplementedError)
NPY_HEADERS = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}


def rollout_files(folder):
    return (p for p in folder.glob("rollout_*.npz") if ROLLOUT.fullmatch(p.stem) and not p.is_symlink())


def array_header(member, size):
    shape, fortran, dtype = NPY_HEADERS[np.lib.format.read_magic(member)](member)
    if dtype.hasobject or member.tell() + math.prod(shape) * dtype.itemsize != size:
        raise ValueError("Invalid rollout array")
    return shape, fortran, dtype


def read_array(archive, key, limit=8 * 1024 * 1024):
    info = archive.getinfo(f"{key}.npy")
    if info.file_size > limit:
        raise ValueError(f"Rollout member {key} is too large")
    with archive.open(info) as f:
        shape, fortran, dtype = array_header(f, info.file_size)
        return np.frombuffer(f.read(), dtype=dtype).reshape(shape, order="F" if fortran else "C")


def single_line(note):
    if not isinstance(note, str) or len(note) > NOTE_LIMIT or note.splitlines() not in ([], [note]):
        raise ValueError("Expected a short, single-line note")
    return note.strip()


def annotation(result, note, exclude=None, exclude_reason=""):
    if result not in RESULTS:
        raise ValueError("Expected success or failure and a single-line note")
    review = {"result": result, "note": single_line(note)}
    if exclude is not None:
        reason = single_line(exclude_reason)
        if type(exclude) is not bool or (exclude and not reason):
            raise ValueError("Excluding a trial requires a reason")
        review.update(exclude=exclude, exclude_reason=reason if exclude else "")
    return review


def metadata(path: Path, archive: ZipFile | None = None):
    if archive is None:
        with ZipFile(path) as z:
            return metadata(path, z)
    meta = json.loads(str(read_array(archive, "meta", 128 * 1024)))
    if not isinstance(meta, dict):
        raise ValueError("Invalid rollout metadata")
    try:
        with path.with_suffix(".review.json").open(encoding="utf-8") as f:
            review = json.loads(f.read(4096))
        meta.update(annotation(review["result"], review.get("note", ""), review.get("exclude"), review.get("exclude_reason", "")))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass  # An absent or malformed sidecar leaves the original result intact.
    return meta


def write_annotation(path: Path, result, note, exclude=None, exclude_reason=""):
    if exclude is None:  # Older clients can still edit results without clearing an exclusion.
        current = metadata(path)
        exclude, exclude_reason = current.get("exclude"), current.get("exclude_reason", "")
    review = annotation(result, note, exclude, exclude_reason)
    tmp = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            tmp = Path(f.name)
            json.dump(review, f, ensure_ascii=False)
            f.write("\n")
        give_back(tmp)
        tmp.replace(path.with_suffix(".review.json"))
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return review
