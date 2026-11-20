"""Robots. ``open_robot`` returns a :class:`~macbot.robot.base.Robot` for the ``robot`` config section.

    type: rebot_b601     reBot B601-RS + Arm 102 with feedforward and two-stage zeroing (reference setup)
    type: lerobot        any follower / leader LeRobot provides (SO-100/101, Koch, OMX, OpenArm, YAM, ...)
    type: sim            a local LeRobot environment with absolute joint position control

``fake=True`` returns a simulated arm with the same joints, so every console runs without hardware.
"""
from __future__ import annotations

from .base import Robot


def open_robot(rcfg: dict, fake: bool = False, leader: bool = False) -> Robot:
    typ = rcfg.get("type", "rebot_b601")
    if fake:
        from .fake import fake_for
        return fake_for(rcfg)
    if typ == "rebot_b601":
        from ..system import preload_pcan
        preload_pcan()
        from .rebot import RebotArm
        return RebotArm(rcfg, leader)
    if typ == "lerobot":
        from .lerobot_generic import LeRobotArm
        return LeRobotArm(rcfg, leader)
    if typ == "sim":
        from .sim import SimArm
        return SimArm(rcfg)
    raise SystemExit(f"unknown robot.type {typ!r}: use rebot_b601, lerobot, or sim")


__all__ = ["Robot", "open_robot"]
