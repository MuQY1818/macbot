"""Binary gripper: the policy outputs a close probability, the robot gets an open or close angle.

Datasets exported with ``macbot export --gripper binary`` carry a ``gripper.json`` with the two angles.
Hysteresis: close as soon as p > 0.6. Once closed, open only after p < 0.4 for ``debounce`` seconds, so a
short spurious "open" does not drop the object. Keep ``debounce`` shorter than the shortest open phase in
your demonstrations (for example, a quick release of the last object).
"""
from __future__ import annotations

import json
from pathlib import Path


class BinaryGripper:
    def __init__(self, cfg: dict, debounce: float):
        self.open_deg, self.close_deg = float(cfg["open_deg"]), float(cfg["close_deg"])
        self.debounce = debounce
        self.closed = False
        self._open_since = None

    def __call__(self, p: float, t: float) -> float:
        if not self.closed:
            self._open_since = None
            self.closed = p > 0.6
        elif p < 0.4:
            if self._open_since is None:
                self._open_since = t
            if t - self._open_since >= self.debounce:
                self._open_since, self.closed = None, False
        else:
            self._open_since = None
        return self.close_deg if self.closed else self.open_deg


def find_gripper_config(policy: Path, arg: str) -> dict | None:
    """``auto``: look for gripper.json in the policy folder and up to three parents. ``raw``: none."""
    if arg == "raw":
        return None
    cands = [Path(arg).expanduser()] if arg != "auto" else [d / "gripper.json" for d in [policy, *list(policy.parents)[:3]]]
    for c in cands:
        if c.is_file():
            cfg = json.loads(c.read_text())
            print(f"binary gripper ({c}): open {cfg['open_deg']} deg, closed {cfg['close_deg']} deg", flush=True)
            return cfg
    if arg != "auto":
        raise FileNotFoundError(arg)
    return None
