"""Read standard LeRobot v3.0 Parquet datasets without importing a robot or policy."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import PurePosixPath
from string import Formatter

import numpy as np

from ..hub import HubRepo, hub_reference
from ..system import expand

INFO = "meta/info.json"
EPISODES = "meta/episodes/*/*.parquet"
STATE = "observation.state"
ACTION = "action"
SUPPORTED = "LeRobot v3.0 (Parquet, with optional MP4 or embedded images)"


def recognize(value):
    """Keep the existing NPZ commands and only autodetect an explicit dataset directory."""
    return value is not None and (expand(value) / INFO).is_file()


def _path(template, **fields):
    if not isinstance(template, str):
        raise ValueError("Missing LeRobot shard path template.")
    for _, name, fmt, conversion in Formatter().parse(template):
        if name is not None and (name not in fields or fmt not in ("", "03d", "04d", "05d", "06d") or conversion):
            raise ValueError(f"Unsupported LeRobot shard path template: {template!r}.")
    filename = PurePosixPath(template.format(**fields))
    if filename.is_absolute() or ".." in filename.parts or not filename.parts or "\\" in str(filename):
        raise ValueError("LeRobot shard paths must stay inside the dataset.")
    return str(filename)


def _pq():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("Reading a LeRobot dataset requires PyArrow (included with the LeRobot dataset extra).") from exc
    return pq


def _names(spec, width):
    names = spec.get("names")
    if isinstance(names, list) and len(names) == width and all(isinstance(n, str) for n in names):
        return names
    return [f"{i + 1}" for i in range(width)]


class Dataset:
    def __init__(self, value):
        root = expand(value)
        reference = hub_reference(value)
        self.repo = HubRepo.open(reference, "dataset") if not root.exists() and reference else None
        self.root = root
        self.label = self.repo.pinned if self.repo else root.name
        self.info = json.loads(self.file(INFO).read_text())
        version = self.info.get("codebase_version")
        storage = self.info.get("storage_format")
        if version != "v3.0" or storage not in (None, "parquet"):
            raise ValueError(f"Unsupported LeRobot dataset version or storage: {version or 'unknown'}, "
                             f"{storage or 'default'}. Supported: {SUPPORTED}.")
        self.fps = float(self.info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Invalid LeRobot dataset frame rate.")
        self.features = self.info["features"]
        self.cameras = [key for key, spec in self.features.items() if spec.get("dtype") in ("image", "video")]
        self.episodes = {}
        files = (self.file(f) for f in sorted(self.repo.files) if PurePosixPath(f).match(EPISODES)) if self.repo else (
            sorted(root.glob(EPISODES))
        )
        for filename in files:
            pq = _pq()
            with pq.ParquetFile(filename) as parquet:
                fields = [name for name in parquet.schema_arrow.names if not name.startswith("stats/")]
                for row in parquet.read(columns=fields).to_pylist():
                    idx = int(row["episode_index"])
                    if idx < 0 or idx in self.episodes:
                        raise ValueError(f"Invalid or duplicate episode index: {idx}.")
                    self.episodes[idx] = row
        if not self.episodes and self.info.get("total_episodes"):
            raise ValueError("LeRobot v3.0 episode metadata is missing (meta/episodes/).")
        self.read_episode = lru_cache(maxsize=2)(self._read_episode)

    def file(self, name):
        if self.repo:
            if name not in self.repo.files:
                raise FileNotFoundError(f"Hub dataset {self.repo.pinned}: missing {name}.")
            return self.repo.download(name)
        path = self.root / name
        if not path.is_file():
            raise FileNotFoundError(f"LeRobot dataset: missing {path}. Supported: {SUPPORTED}.")
        return path

    def _data_path(self, row):
        return self.file(_path(self.info["data_path"], chunk_index=row["data/chunk_index"],
                               file_index=row["data/file_index"]))

    @staticmethod
    def _filters(idx, row):
        return [("episode_index", "=", idx), ("index", ">=", row["dataset_from_index"]),
                ("index", "<", row["dataset_to_index"])]

    def _read_episode(self, idx):
        row = self.episodes[idx]
        columns = ["index", "timestamp", "frame_index", *(key for key in (STATE, ACTION) if key in self.features)]
        table = _pq().read_table(self._data_path(row), columns=columns, filters=self._filters(idx, row)).sort_by("index")
        return {key: table[key].to_pylist() for key in columns}

    def timing(self, idx):
        row = self.episodes[idx]
        episode = self.read_episode(idx)
        ts = np.asarray(episode["timestamp"], dtype=float)
        if not np.isfinite(ts).all() or (np.diff(ts) < 0).any():
            raise ValueError(f"Episode {idx} has invalid or unsorted timestamps.")
        gaps = np.diff(ts)
        frame_indices = {i for i in episode["frame_index"] if i >= 0}
        expected = max(row["length"], max(frame_indices, default=-1) + 1)
        return {"frames": len(ts), "seconds": (float(ts[-1] - ts[0]) + 1 / self.fps) if len(ts) else 0.,
                "missing": expected - len(frame_indices), "late": int((gaps > 1.5 / self.fps).sum()),
                "max_dt_ms": round(float(gaps.max(initial=0) * 1000), 2)}

    def traces(self, idx):
        episode = self.read_episode(idx)
        count = len(episode["timestamp"])
        arrays = {}
        names = {}
        for key in (STATE, ACTION):
            if key not in episode or not count:
                continue
            values = np.asarray(episode[key], dtype=float).reshape(count, -1)
            arrays[key] = np.where(np.isfinite(values), values, None).tolist()
            names[key] = _names(self.features[key], values.shape[1])
        order = list(dict.fromkeys(n for col in names.values() for n in col))

        def align(key):
            cols = {n: i for i, n in enumerate(names.get(key, []))}
            return [[row[cols[n]] if n in cols else None for n in order]
                    for row in arrays.get(key, [[] for _ in range(count)])]

        return order, align(STATE), align(ACTION)

    def video(self, idx, key, frame):
        row = self.episodes[idx]
        prefix = f"videos/{key}"
        path = self.file(_path(self.info["video_path"], video_key=key,
                               chunk_index=row[f"{prefix}/chunk_index"], file_index=row[f"{prefix}/file_index"]))
        when = row[f"{prefix}/from_timestamp"] + self.read_episode(idx)["timestamp"][frame]
        if not 0 <= when < row[f"{prefix}/to_timestamp"]:
            raise ValueError(f"Frame timestamp is outside the episode video segment for {key}.")
        return path, when

    def image(self, idx, key, frame):
        row = self.episodes[idx]
        global_index = self.read_episode(idx)["index"][frame]
        table = _pq().read_table(self._data_path(row), columns=[key], filters=[("index", "=", global_index),
                                                                             ("episode_index", "=", idx)])
        value = table[key][0].as_py()
        if not isinstance(value, dict) or not value.get("bytes"):
            raise ValueError(f"Missing embedded image bytes for {key}.")
        return value["bytes"]


def health(dataset, split=None):
    rows = [(idx, dataset.timing(idx)) for idx in sorted(dataset.episodes)]
    print(f"LeRobot v3.0: {dataset.label} ({dataset.fps:g} fps)")
    print("episode  frames  duration  missing indices  late gaps  max gap")
    for idx, row in rows:
        print(f"{idx:7d} {row['frames']:7d} {row['seconds']:8.2f}s {row['missing']:16d} "
              f"{row['late']:10d} {row['max_dt_ms']:7.1f}ms")
    print("Frame gaps come from recorded indices and timestamps; camera drops before recording are not observable.")
    if split is not None:
        before = [row for i, row in rows if i < split]
        after = [row for i, row in rows if i >= split]
        if not before or not after:
            raise ValueError(f"--split {split} leaves one side empty")
        for title, group in (("before", before), ("from", after)):
            print(f"{title} {split}: {len(group)} episodes, {sum(r['missing'] for r in group)} missing indices, "
                  f"{sum(r['late'] for r in group)} late gaps")
