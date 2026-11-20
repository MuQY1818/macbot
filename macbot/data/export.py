"""Recorded episodes (``episode_XXX.npz``) -> LeRobot dataset (v3 format, camera streams as video).

Feature names match LeRobot's ``rebot_b601_follower``, so a trained policy plugs straight into
``macbot run``:

    observation.state            (7,)  measured follower joints, degrees
    action                       (7,)  command sent to the follower (gripper: leader angle, or 0/1)
    observation.images.<camera>        one entry per configured camera

``--gripper binary`` turns the gripper action into 0 (open) / 1 (closed) with hysteresis on the leader
reading, and writes ``gripper.json`` next to the dataset. ``macbot run`` finds that file and maps the
policy output back to the two angles. Each frame's task string is the episode's instruction.
Episodes that failed the quality check are skipped; ``--skip-warnings`` also skips those with warnings.
"""
from __future__ import annotations

import json
import shutil
import sys

import cv2
import numpy as np

from ..config import JOINTS, Config
from ..i18n import text
from ..system import expand


def binarize(g, close_above: float, open_below: float) -> np.ndarray:
    """Leader gripper angle -> 0/1 (1 = closed) with hysteresis, so values near a threshold do not flicker."""
    out, s = np.zeros(len(g), np.float32), 0.0
    for i, v in enumerate(g):
        if s == 0.0 and v > close_above:
            s = 1.0
        elif s == 1.0 and v < open_below:
            s = 0.0
        out[i] = s
    return out


def frames(d, cam):
    off, blob = d[f"{cam}_offsets"], d[f"{cam}_jpeg"]
    for i in range(len(off) - 1):
        yield cv2.cvtColor(cv2.imdecode(blob[off[i]:off[i + 1]], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _msg(items) -> str:
    return "; ".join(text(m, "en") if isinstance(m, dict) else str(m) for m in items)


def run(cfg: Config, args):
    binary = args.gripper == "binary"
    name = cfg.task + ("_bin" if binary else "")
    data = expand(args.data) if args.data else cfg.raw_dir()
    out = expand(args.out) if args.out else cfg.dataset_dir(name)
    repo_id = args.repo_id or f"local/{name}"
    files = sorted(data.glob("episode_*.npz"))
    if not files:
        sys.exit(f"no episode_*.npz in {data}")
    if out.exists():
        if not args.overwrite:
            sys.exit(f"{out} exists; add --overwrite to rebuild it")
        shutil.rmtree(out)

    keep, skipped = [], []
    for p in files:
        with np.load(p) as d:
            meta = json.loads(str(d["meta"]))
        c = meta.get("check") or {"ok": True, "errors": [], "warnings": []}
        if not c["ok"] or (args.skip_warnings and c["warnings"]):
            skipped.append((p.name, _msg(c.get("errors", []) + c.get("warnings", []))))
        else:
            keep.append((p, meta))
    if not keep:
        sys.exit("nothing to export")

    m0 = keep[0][1]
    fps, cams = int(m0["fps"]), list(m0["cameras"])
    for p, m in keep[1:]:
        if int(m["fps"]) != fps or any(m["cameras"].get(c, {}).get("size") != m0["cameras"][c]["size"] for c in cams):
            sys.exit(f"{p.name}: fps or image size differs from the first episode; they cannot share a dataset")
    with np.load(keep[0][0]) as d:
        shape = {c: next(frames(d, c)).shape for c in cams}

    names = [f"{n}.pos" for n in JOINTS]
    kind = "image" if args.images else "video"
    features = {"observation.state": {"dtype": "float32", "shape": (7,), "names": names},
                "action": {"dtype": "float32", "shape": (7,), "names": names},
                **{f"observation.images.{c}": {"dtype": kind, "shape": shape[c], "names": ["height", "width", "channels"]}
                   for c in cams}}
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=out,
                               robot_type="rebot_b601_follower", use_videos=not args.images)
    total = 0
    for p, m in keep:
        with np.load(p) as d:
            q, a = d["qpos"].astype(np.float32), d["action"].astype(np.float32)
            if binary:
                a[:, 6] = binarize(a[:, 6], args.close_above, args.open_below)
            streams = [frames(d, c) for c in cams]
            for i, imgs in enumerate(zip(*streams)):
                ds.add_frame({"observation.state": q[i], "action": a[i], "task": m["instruction"],
                              **{f"observation.images.{c}": im for c, im in zip(cams, imgs)}})
        ds.save_episode()
        total += len(q)
        print(f"  {p.name}: {len(q)} frames  \"{m['instruction']}\"", flush=True)
    ds.finalize()
    if binary:
        (out / "gripper.json").write_text(json.dumps({
            "mode": "binary", "close_value": 1.0, "open_value": 0.0, "open_deg": args.open_deg, "close_deg": args.close_deg,
            "label_from": "leader gripper with hysteresis", "close_above": args.close_above, "open_below": args.open_below}, indent=1))
        print(f"binary gripper: closed above {args.close_above} deg, open below {args.open_below} deg; "
              f"deploy open {args.open_deg} deg, closed {args.close_deg} deg")
    print(f"exported {len(keep)} episodes, {total} frames ({total / fps / 60:.1f} min) to {out} (repo_id {repo_id}, {fps} fps)")
    for n, why in skipped:
        print(f"  skipped {n}: {why}")
