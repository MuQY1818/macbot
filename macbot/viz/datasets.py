"""Read-only LeRobot dataset adapter for the standalone replay console."""
from __future__ import annotations

import json
import threading
from collections import OrderedDict

import cv2
import numpy as np

from ..data import lerobot_v3


class DatasetStore:
    readonly = True

    def __init__(self, dataset):
        self.dataset = dataset
        self.folder = dataset.label
        self.slots = threading.BoundedSemaphore(4)
        self.video_lock = threading.Lock()
        self.captures = OrderedDict()

    def index(self):
        return [{"name": f"episode_{idx:03d}", "s": round(row["length"] / self.dataset.fps, 2)}
                for idx, row in sorted(self.dataset.episodes.items())]

    def summary_payload(self, name):
        dataset = self.dataset
        idx = int(name.split("_")[1])
        row = dataset.episodes[idx]
        episode = dataset.read_episode(idx)
        t = episode["timestamp"]
        joints, q, cmd = dataset.traces(idx)
        cameras = {f"cam{i}": c for i, c in enumerate(dataset.cameras)}
        sizes = {}
        for key, camera in cameras.items():
            spec = dataset.features[camera]
            shape, names = spec.get("shape", []), spec.get("names") or []
            if "width" in names and "height" in names and len(shape) == len(names):
                sizes[key] = [shape[names.index("width")], shape[names.index("height")]]
        summary = {
            "meta": {"joint_names": joints, "instruction": " · ".join(row.get("tasks") or []), "fps": dataset.fps},
            "t": [stamp - t[0] for stamp in t], "q": q, "cmd": cmd,
            "cams": list(cameras), "camera_labels": cameras, "sizes": sizes,
            "chunk_t": [], "chunk_ms": [], "timing": dataset.timing(idx),
        }
        return json.dumps(summary, ensure_ascii=False, allow_nan=False).encode()

    def frame(self, name, camera, frame):
        dataset = self.dataset
        idx = int(name.split("_")[1])
        if not camera.startswith("cam") or not camera[3:].isdigit():
            raise KeyError(camera)
        try:
            key = dataset.cameras[int(camera[3:])]
            if not 0 <= frame < len(dataset.read_episode(idx)["timestamp"]):
                raise IndexError
        except IndexError as exc:
            raise KeyError("Unknown camera or frame") from exc
        if dataset.features[key]["dtype"] == "image":
            blob = dataset.image(idx, key, frame)
            image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = self._video_frame(*dataset.video(idx, key, frame))
        if image is None:
            raise ValueError(f"Could not decode {key} in episode {idx}.")
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError(f"Could not display {key} in episode {idx}.")
        return encoded.tobytes()

    def _video_frame(self, path, when):
        # LeRobot writes constant-rate video; the offset is relative to the shared video file.
        with self.video_lock:
            if path not in self.captures:
                cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    cap.release()
                    raise ValueError(f"Could not open video: {path.name}.")
                self.captures[path] = cap
                if len(self.captures) > 4:
                    self.captures.popitem(last=False)[1].release()
            cap = self.captures[path]
            self.captures.move_to_end(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or self.dataset.fps
            target, current = round(when * fps), round(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if target < current or target > current + 5:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, target):
                    raise ValueError(f"Could not seek video: {path.name}.")
            else:
                for _ in range(target - current):
                    if not cap.grab():
                        return None
            ok, image = cap.read()
            return image if ok else None

    def close(self):
        with self.video_lock:
            for cap in self.captures.values():
                cap.release()
            self.captures.clear()


def open_dataset(value):
    try:
        return lerobot_v3.Dataset(value)
    except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
