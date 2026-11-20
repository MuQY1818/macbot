"""Measure the run control loop and save screenshot data using only fake devices and a NumPy policy."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from macbot import robot  # noqa: E402
from macbot.cli import main as cli  # noqa: E402
from macbot.policy import runner  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="new folder for the fake rollout and timing summary")
    parser.add_argument("--seconds", type=float, default=12, help="duration of each steady mode")
    args = parser.parse_args()
    if args.seconds < 4:
        parser.error("--seconds must be at least 4")
    out = args.out.resolve()
    out.mkdir()
    cfg_path, policy = out / "config.yaml", out / "policy" / "numpy"
    policy.mkdir(parents=True)
    cfg_path.write_text(yaml.safe_dump({"data_root": str(out), "task": {"name": "takeover_check"}}))
    (policy / "config.json").write_text('{"type": "act"}')
    center = [16., -38., -49., 12., -10., 12., -110.]
    periods, transitions = {"running": [], "following": []}, {}
    previous = None
    open_robot = robot.open_robot

    class Policy:
        H, kind, shapes = 30, "act", None

        def __call__(self, images, q):
            return np.tile(center, (self.H, 1))

    class TimingState(runner.State):
        def take_command(self):
            nonlocal previous
            now = time.perf_counter()
            if previous and previous[0] == self.mode and self.mode in periods:
                if now - transitions[self.mode] >= .5:
                    periods[self.mode].append(now - previous[1])
            previous = self.mode, now
            start = transitions.setdefault(self.mode, now)
            if self.mode == "paused" and self.rec is None:
                self.request("quit" if "following" in transitions else "start")
            elif self.mode == "running" and now - start >= args.seconds:
                self.request("takeover")
            elif self.mode == "following":
                if not self.rec["hand_marks"]:
                    self.request("mark_hand")
                if now - start >= args.seconds:
                    self.request("save_fail")
            elif self.mode == "aligning" and now - start > 3:
                raise RuntimeError("fake leader did not pass alignment")
            return super().take_command()

    def timing_robot(config, *, fake, leader=False, **kwargs):
        assert fake and leader
        arm = open_robot(config, fake=True, leader=True, **kwargs)
        wave = arm.read_leader

        def leader_pose():
            if "aligning" not in transitions or time.perf_counter() - transitions["aligning"] < 1:
                return wave()  # start unaligned, using the same fake leader as collect
            if "following" not in transitions:
                return center
            elapsed = time.perf_counter() - transitions["following"]
            return [v + 8 * np.sin(elapsed + i) for i, v in enumerate(center)]

        arm.read_leader = leader_pose
        return arm

    with (
        patch.object(runner, "State", TimingState),
        patch.object(runner, "load_policy", return_value=Policy()),
        patch.object(robot, "open_robot", side_effect=timing_robot),
    ):
        cli(["--config", str(cfg_path), "run", "--fake", "--takeover", "--no-browser", "--speed", "1",
             "--device", "cpu", "--mode", "async", "--policy", str(policy)])

    dt = 1 / 60
    values = {mode: {"cycles": len(p), "p50_ms": float(np.median(p) * 1e3), "p99_ms": float(np.percentile(p, 99) * 1e3),
                     "late_cycles": int(np.count_nonzero(np.asarray(p) > 1.5 * dt))}
              for mode, p in periods.items()}
    text = json.dumps(values, indent=2) + "\n"
    (out / "timing.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
