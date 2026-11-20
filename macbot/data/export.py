"""Recorded episodes (``episode_XXX.npz``) -> LeRobot dataset (v3 format, camera streams as video).

Feature names follow the LeRobot robot class (``<joint>.pos``), so a trained policy plugs straight
into ``macbot run`` and into LeRobot's own tools:

    observation.state            (n,)  measured follower joints (n = the robot's joints)
    action                       (n,)  command sent to the follower (gripper: leader value, or 0/1)
    observation.images.<camera>        one entry per configured camera

``--gripper binary`` turns the gripper action into 0 (open) / 1 (closed) with hysteresis on the leader
reading, and writes ``gripper.json`` next to the dataset. ``macbot run`` finds that file and maps the
policy output back to the two angles. Each frame's task string is the episode's instruction.
Episodes that failed the quality check are skipped; ``--skip-warnings`` also skips those with warnings.
"""
from __future__ import annotations

import json
import math
import shutil
import sys

import cv2
import numpy as np

from ..cameras.framing import FRAMING_FILE, dataset_framing, write_framing
from ..config import Config
from ..i18n import text
from ..system import expand
from .grasp import detect, gripper_index, schmitt, settings
from .rollouts import RESULTS, metadata, rollout_files

ROLLOUT_FEATURES = {
    key: {"dtype": "uint8", "shape": (1,), "names": None}
    for key in ("takeover", "grasp_miss", "hand_mark")
}
HAND_MARK_RADIUS_S = 0.5


def binarize(g, close_above: float, open_below: float) -> np.ndarray:
    """Leader gripper angle -> 0/1 (1 = closed) with hysteresis, so values near a threshold do not flicker."""
    return schmitt(g, close_above, open_below).astype(np.float32)


def robot_type(meta: dict) -> str:
    """LeRobot robot type recorded in the dataset: the follower type for generic LeRobot arms."""
    r = meta.get("robot") or {}
    if r.get("robot_type"):
        return r["robot_type"]
    if r.get("kind") == "lerobot":
        return r["follower"]["type"]
    return "rebot_b601_follower"


def frames(d, cam):
    off, blob = d[f"{cam}_offsets"], d[f"{cam}_jpeg"]
    for i in range(len(off) - 1):
        yield cv2.cvtColor(cv2.imdecode(blob[off[i]:off[i + 1]], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _msg(items) -> str:
    return "; ".join(text(m, "en") if isinstance(m, dict) else str(m) for m in items)


def rollout_labels(d, meta, q, a, gi, args, grasp_cfg):
    t = d["t"]
    takeover = d["takeover"] if "takeover" in d.files else np.zeros(len(q), np.uint8)
    if (t.shape != (len(q),) or takeover.shape != t.shape or not np.isin(takeover, [0, 1, 2]).all()
            or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)):
        raise ValueError("invalid frame timestamps or takeover labels")
    miss = np.zeros(len(q), np.uint8)
    report = detect(meta, t, a, q, grasp_cfg)
    for span in report["misses"]:
        miss[span["first"]:span["end"]] = 1
    hand = np.zeros(len(q), np.uint8)
    for mark in meta.get("hand_marks", []):
        start, end = t.searchsorted(mark - HAND_MARK_RADIUS_S), t.searchsorted(mark + HAND_MARK_RADIUS_S, side="right")
        hand[start:end] = 1
    if gi is not None:
        gp = d["gp"] if "gp" in d.files else np.full(len(q), np.nan)
        leader = d["leader"][:, gi] if "leader" in d.files else np.full(len(q), np.nan)
        human = takeover == 1
        if gp.shape != t.shape or leader.shape != t.shape or not np.isfinite(leader[human]).all():
            raise ValueError("human-controlled frames need a valid leader gripper reading")
        # Old recordings did not identify binary policies. A probability-like range is
        # evidence for legacy gp; raw angle policies use command hysteresis instead.
        finite = gp[(takeover == 0) & np.isfinite(gp)]
        binary = meta.get("binary_gripper", len(finite) > 0 and finite.min() >= -0.1 and finite.max() <= 1.1)
        policy = (takeover == 0) & np.isfinite(gp) if binary else np.zeros(len(q), bool)
        command = a[:, gi].copy()
        new_config = meta.get("binary_gripper")
        closed = False
        for i in range(len(q)):
            if policy[i]:
                closed = gp[i] >= 0.5
            elif not human[i] and isinstance(new_config, dict):
                closed = abs(command[i] - new_config["close_deg"]) < abs(command[i] - new_config["open_deg"])
            else:
                angle = leader[i] if human[i] else command[i]
                closed = angle >= args.open_below if closed else angle > args.close_above
            a[i, gi] = closed
    return np.column_stack((takeover, miss, hand)).astype(np.uint8), report


def label_episodes(out, results, source):
    """LeRobot 0.7's public writer has no custom episode metadata argument."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    for path in (out / "meta" / "episodes").glob("chunk-*/file-*.parquet"):
        table = pq.read_table(path)
        review = [results[i] for i in table["episode_index"].to_pylist()]
        table = table.append_column("success", pa.array([r[0] for r in review], type=pa.bool_()))
        table = table.append_column("source", pa.array([source] * len(review), type=pa.string()))
        table = table.append_column("grasp_miss_count", pa.array([r[1] for r in review], type=pa.int32()))
        table = table.append_column("grasp_unknown_closes", pa.array([r[2] for r in review], type=pa.int32()))
        tmp = path.with_suffix(".tmp")
        pq.write_table(table, tmp)
        tmp.replace(path)


def frame_layout(d, cams):
    return {c: next(frames(d, c)).shape for c in cams}


def run(cfg: Config, args):
    rollouts = getattr(args, "rollouts", False)
    labeled = rollouts or getattr(args, "with_rollout_labels", False)
    binary = labeled or args.gripper == "binary"
    ex = cfg.get("export", {})
    for k in ("close_above", "open_below", "open_deg", "close_deg"):
        if getattr(args, k) is None:
            setattr(args, k, float(ex[f"gripper_{k}"]))
    if binary and (not all(math.isfinite(getattr(args, k)) for k in ("close_above", "open_below", "open_deg", "close_deg"))
                   or args.open_below >= args.close_above):
        sys.exit("binary gripper needs finite angles and --open-below less than --close-above")
    name = cfg.task + ("_rollouts" if rollouts else "_bin_labeled" if labeled else "_bin" if binary else "")
    data = expand(args.data) if args.data else cfg.rollout_dir() if rollouts else cfg.raw_dir()
    out = expand(args.out) if args.out else cfg.dataset_dir(name)
    repo_id = args.repo_id or f"local/{name}"
    pattern = "rollout_*.npz" if rollouts else "episode_*.npz"
    files = sorted(rollout_files(data) if rollouts else data.glob(pattern))
    if not files:
        sys.exit(f"no {pattern} in {data}")
    if out.resolve() == data.resolve() or out.resolve() in data.resolve().parents:
        sys.exit("output must not contain the input folder")
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} exists; add --overwrite to rebuild it")

    keep, skipped = [], []
    for p in files:
        if rollouts:
            meta = metadata(p)
            if meta.get("exclude"):
                skipped.append((p.name, f"excluded: {meta['exclude_reason']}"))
                continue
            if meta.get("result") not in RESULTS:
                sys.exit(f"{p.name}: missing trial result; review it before export")
        else:
            with np.load(p) as d:
                meta = json.loads(str(d["meta"]))
            c = meta.get("check") or {"ok": True, "errors": [], "warnings": []}
            if not c["ok"] or (args.skip_warnings and c["warnings"]):
                skipped.append((p.name, _msg(c.get("errors", []) + c.get("warnings", []))))
                continue
        keep.append((p, meta))
    if not keep:
        for n, why in skipped:
            print(f"  skipped {n}: {why}")
        sys.exit("nothing to export")

    m0 = keep[0][1]
    fps, cams = int(m0["fps"]), list(m0["cameras"])
    for p, m in keep[1:]:
        if int(m["fps"]) != fps or set(m["cameras"]) != set(cams) or (not rollouts and any(
                m["cameras"][c]["size"] != m0["cameras"][c]["size"] for c in cams)):
            sys.exit(f"{p.name}: fps, cameras or image size differs from the first episode")
        if (robot_type(m) != robot_type(m0) or bool(m.get("fake")) != bool(m0.get("fake"))
                or (m.get("robot") or {}).get("semantics") != (m0.get("robot") or {}).get("semantics")):
            sys.exit(f"{p.name}: robot or observation semantics differ from the first episode")
    with np.load(keep[0][0]) as d:
        shape = frame_layout(d, cams)

    joints = m0.get("joint_names") or []
    if any((m.get("joint_names") or []) != joints for _, m in keep[1:]):
        sys.exit("episodes were recorded with different joint lists; they cannot share a dataset")
    recorded_framing = dataset_framing(keep)
    gi = gripper_index(m0)
    if rollouts and any(gripper_index(m) != gi for _, m in keep):
        sys.exit("rollouts were recorded with different gripper joints")
    if binary and gi is None and not rollouts:
        sys.exit("--gripper binary needs a joint named gripper")
    rtype = robot_type(m0)
    names = [f"{n}.pos" for n in joints]
    dim = len(names)
    kind = "image" if args.images else "video"
    features = {"observation.state": {"dtype": "float32", "shape": (dim,), "names": names},
                "action": {"dtype": "float32", "shape": (dim,), "names": names},
                **{f"observation.images.{c}": {"dtype": kind, "shape": shape[c], "names": ["height", "width", "channels"]}
                   for c in cams}}
    if labeled:
        features.update(ROLLOUT_FEATURES)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=out,
                               robot_type=rtype, use_videos=not args.images)
    total, takeover_seconds, results = 0, 0.0, []
    grasp_cfg = settings(cfg) if rollouts else None
    for p, m in keep:
        with np.load(p) as d:
            q, a = d["qpos"].astype(np.float32), d["cmd" if rollouts else "action"].astype(np.float32)
            if not len(q) or q.shape != a.shape or q.shape != (len(q), dim):
                sys.exit(f"{p.name}: expected matching state and action arrays with {dim} joints")
            labels = [()] * len(q)
            if rollouts:
                try:
                    labels, report = rollout_labels(d, m, q, a, gi, args, grasp_cfg)
                except (ValueError, IndexError, KeyError) as exc:
                    sys.exit(f"{p.name}: {exc}")  # includes the source trial for malformed labels
                results.append((m["result"] == "success", report["miss_count"], report["unknown_closes"]))
                takeover_seconds += m.get("takeover_seconds", float(np.count_nonzero(labels[:, 0] == 1)) / fps)
            elif binary:
                a[:, gi] = binarize(a[:, gi], args.close_above, args.open_below)
                if labeled:
                    labels = np.zeros((len(q), len(ROLLOUT_FEATURES)), np.uint8)
                    results.append((True, None, None))
            streams = [frames(d, c) for c in cams]
            for qi, ai, tags, *imgs in zip(q, a, labels, *streams, strict=True):
                if any(im.shape != shape[c] for c, im in zip(cams, imgs, strict=True)):
                    sys.exit(f"{p.name}: decoded camera size differs from the first episode")
                ds.add_frame({"observation.state": qi, "action": ai, "task": m["instruction"],
                              **{f"observation.images.{c}": im for c, im in zip(cams, imgs, strict=True)},
                              **{key: np.array([value], dtype=np.uint8) for key, value in zip(ROLLOUT_FEATURES, tags, strict=False)}})
        ds.save_episode(parallel_encoding=not labeled)
        total += len(q)
        print(f"  {p.name}: {len(q)} frames  \"{m['instruction']}\"", flush=True)
    ds.finalize()
    if labeled:
        label_episodes(out, results, "rollout" if rollouts else "demonstration")
    write_framing(out / "meta" / FRAMING_FILE, recorded_framing)
    if binary and gi is not None:
        (out / "gripper.json").write_text(json.dumps({
            "mode": "binary", "close_value": 1.0, "open_value": 0.0, "open_deg": args.open_deg, "close_deg": args.close_deg,
            "label_from": "policy output or leader gripper with hysteresis" if rollouts else "leader gripper with hysteresis",
            "close_above": args.close_above, "open_below": args.open_below}, indent=1))
        print(f"binary gripper: closed above {args.close_above} deg, open below {args.open_below} deg; "
              f"deploy open {args.open_deg} deg, closed {args.close_deg} deg")
    print(f"exported {len(keep)} episodes, {total} frames ({total / fps / 60:.1f} min) to {out} (repo_id {repo_id}, {fps} fps)")
    if rollouts:
        print(f"  {sum(r[0] for r in results)} successes, {takeover_seconds:.2f} takeover seconds, "
              f"{sum(r[1] or 0 for r in results)} grasp misses; "
              f"{sum(r[1] is None for r in results)} trials without calibration, "
              f"{sum(r[2] or 0 for r in results)} inconclusive closes")
    for n, why in skipped:
        print(f"  skipped {n}: {why}")
