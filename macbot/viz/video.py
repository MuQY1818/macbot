"""Rollout or episode -> one shareable MP4 with all cameras side by side at equal height.

Frames are decoded from the stored JPEGs and encoded once with H.264 (``ffmpeg`` on PATH, for example
``brew install ffmpeg``). Without ffmpeg, OpenCV writes an MPEG-4 file instead.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import cv2
import numpy as np

from ..system import expand, give_back

GAP, BG = 8, (20, 18, 18)


def _frames(d, cam):
    off, buf = d[f"{cam}_offsets"], d[f"{cam}_jpeg"]
    for i in range(len(off) - 1):
        yield cv2.imdecode(buf[off[i]:off[i + 1]], cv2.IMREAD_COLOR)


def _fit(im, h):
    w = round(im.shape[1] * h / im.shape[0])
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_LANCZOS4 if h > im.shape[0] else cv2.INTER_AREA)


def _tag(im, text, corner):
    font, s, th = cv2.FONT_HERSHEY_SIMPLEX, im.shape[0] / 900, max(1, round(im.shape[0] / 500))
    (tw, tht), _ = cv2.getTextSize(text, font, s, th)
    pad = round(10 * s + 4)
    x = pad if corner == "left" else im.shape[1] - tw - 3 * pad
    y = im.shape[0] - tht - 3 * pad
    ov = im.copy()
    cv2.rectangle(ov, (x, y), (x + tw + 2 * pad, y + tht + 2 * pad), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, im, 0.45, 0, im)
    cv2.putText(im, text, (x + pad, y + tht + pad), font, s, (255, 255, 255), th, cv2.LINE_AA)


def run(args):
    src = expand(args.rollout)
    d = np.load(src)
    meta = json.loads(str(d["meta"]))
    cams = meta.get("cameras") if isinstance(meta.get("cameras"), list) else list(meta.get("cameras") or [])
    cams = [c for c in cams if f"{c}_jpeg" in d.files] or [k[:-5] for k in d.files if k.endswith("_jpeg")]
    fps = float(meta.get("fps", 30))
    h = args.height - args.height % 2
    out = expand(args.out) if args.out else src.with_suffix(".mp4")
    streams = [_frames(d, c) for c in cams]
    first = [_fit(next(_frames(d, c)), h) for c in cams]
    W = sum(f.shape[1] for f in first) + GAP * (len(cams) - 1)
    W += W % 2
    ff = shutil.which("ffmpeg")
    if ff:
        enc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{h}",
                                "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "21",
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
        write = lambda fr: enc.stdin.write(fr.tobytes())  # noqa: E731
    else:
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, h))
        write = vw.write
    n = 0
    for imgs in zip(*streams, strict=False):
        canvas = np.full((h, W, 3), BG, np.uint8)
        x = 0
        for c, im in zip(cams, imgs, strict=False):
            im = _fit(im, h)
            _tag(im, c, "left")
            canvas[:, x:x + im.shape[1]] = im
            x += im.shape[1] + GAP
        _tag(canvas, f"{n / fps:5.1f} s", "right")
        write(np.ascontiguousarray(canvas))
        n += 1
    if ff:
        enc.stdin.close()
        enc.wait()
    else:
        vw.release()
    give_back(out)
    print(f"wrote {out} ({n} frames, {n / fps:.1f} s, {W}x{h})")
