"""Extract matched current and four-frame-old observations from LeRobot v3 files."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def decode(path, times, fps):
    import av

    wanted = {round(t * fps): t for t in times}
    samples = {}
    with av.open(str(path)) as container:
        container.streams.video[0].thread_type = "AUTO"
        for frame in container.decode(video=0):
            tick = round(frame.time * fps)
            if tick in wanted:
                if tick in samples or abs(frame.time - wanted[tick]) > 0.002:
                    raise ValueError(f"Ambiguous video timestamp at {tick / fps:.6f} in {path}")
                samples[tick] = frame.to_ndarray(format="rgb24")
                if samples.keys() == wanted.keys():
                    break
    if missing := wanted.keys() - samples.keys():
        raise ValueError(f"Missing {len(missing)} requested frames in {path}")
    return np.stack([samples[round(t * fps)] for t in times])


def extract(data, episodes, tasks, videos, fps, out):
    import pyarrow.parquet as pq

    columns = ["episode_index", "frame_index", "task_index", "timestamp", "observation.state"]
    rows = pq.read_table(data, columns=columns).to_pylist()
    headers = pq.read_table(episodes).to_pylist()
    headers.sort(key=lambda x: x["episode_index"])
    task_names = {r["task_index"]: r["__index_level_0__"] for r in pq.read_table(tasks).to_pylist()}
    pairs = []
    for slot in np.linspace(0, len(headers) - 1, 5, dtype=int):
        h = headers[slot]
        start, stop = h["dataset_from_index"], h["dataset_to_index"]
        if stop - start < 9:
            raise ValueError("Expected a full data shard and at least nine frames per episode")
        for i in np.linspace(start + 4, stop - 1, 5, dtype=int):
            current, prior = rows[i], rows[i - 4]
            if any(row["episode_index"] != h["episode_index"] for row in (current, prior)):
                raise ValueError("Episode offsets do not match the data shard")
            if current["frame_index"] - prior["frame_index"] != 4:
                raise ValueError("Nonconsecutive frame indices")
            pairs.append((h, current, prior))

    arrays = {}
    for index, prefix in ((1, ""), (2, "prior_")):
        sample = [pair[index] for pair in pairs]
        arrays[prefix + "q"] = np.array([r["observation.state"] for r in sample], dtype=np.float32)
        arrays[prefix + "task"] = np.array([task_names[r["task_index"]] for r in sample])
        for column in ("episode_index", "frame_index", "timestamp"):
            arrays[prefix + column] = np.array([r[column] for r in sample])
    for name, path in videos.items():
        key = f"videos/observation.images.{name}"
        for h, *_ in pairs:
            if h[f"{key}/chunk_index"] or h[f"{key}/file_index"]:
                raise ValueError("This extractor expects one video shard per camera")
        times = [pair[0][f"{key}/from_timestamp"] + pair[i]["timestamp"] for i in (1, 2) for pair in pairs]
        current, prior = np.split(decode(path, times, fps), 2)
        arrays[f"rgb_{name}"], arrays[f"prior_rgb_{name}"] = current, prior
    np.savez_compressed(out, **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "episodes", "tasks", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--video", action="append", required=True, metavar="CAMERA=FILE")
    parser.add_argument("--fps", type=float, required=True)
    args = parser.parse_args()
    videos = dict(v.split("=", 1) for v in args.video)
    extract(args.data, args.episodes, args.tasks, videos, args.fps, args.out)


if __name__ == "__main__":
    main()
