"""Static gravity torque model of the reBot B601-RS, read from the URDF.

``gravity_torque_public(q_deg)`` returns the torque (N*m) each of the 6 arm joints must produce
to hold the arm still, in LeRobot public joint coordinates. URDF angles relate to public
angles by ``q_urdf = SIGN * q_pub`` with the same zero. SIGN was fitted on static frames of
teleoperation logs: model/measured ratio 0.98 (shoulder) and 0.90 (elbow).

The URDF is not redistributed here. Get ``ReBot_Arm_RS.urdf`` from Seeed's reBot description
package and set ``robot.urdf`` in your config.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
G = np.array([0.0, 0.0, -9.81])
SIGN = np.array([1.0, -1.0, -1.0, -1.0, -1.0, 1.0])


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_rot(axis, th):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _f(s):
    return np.array([float(v) for v in s.split()])


class GravityModel:
    def __init__(self, urdf, sign=None):
        root = ET.parse(urdf).getroot()
        self.links = {}
        for l in root.findall("link"):
            i = l.find("inertial")
            if i is not None:
                self.links[l.get("name")] = (float(i.find("mass").get("value")), _f(i.find("origin").get("xyz")))
        self.joints = []   # kinematic chain from base_link (gripper fingers treated as fixed)
        for j in root.findall("joint"):
            o = j.find("origin")
            a = j.find("axis")
            self.joints.append(dict(
                name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                child=j.find("child").get("link"), xyz=_f(o.get("xyz")), R=_rpy(*_f(o.get("rpy"))),
                axis=_f(a.get("xyz")) if a is not None else np.zeros(3)))
        self.sign = np.array(SIGN if sign is None else sign, dtype=float)

    def _fk(self, q_urdf):
        """(R, p) of every link, and (world axis, world origin) of every arm joint."""
        frames = {"base_link": (np.eye(3), np.zeros(3))}
        jinfo = {}
        qmap = dict(zip(ARM, q_urdf))
        pending = list(self.joints)
        while pending:
            for j in list(pending):
                if j["parent"] not in frames:
                    continue
                Rp, pp = frames[j["parent"]]
                Rj = Rp @ j["R"]
                pj = pp + Rp @ j["xyz"]
                if j["type"] == "revolute":
                    th = qmap.get(j["name"], 0.0)
                    jinfo[j["name"]] = (Rj @ (j["axis"] / np.linalg.norm(j["axis"])), pj)
                    Rj = Rj @ _axis_rot(j["axis"], th)
                frames[j["child"]] = (Rj, pj)
                pending.remove(j)
        return frames, jinfo

    def _descendants(self, joint_name):
        children = {}
        for j in self.joints:
            children.setdefault(j["parent"], []).append(j["child"])
        start = next(j["child"] for j in self.joints if j["name"] == joint_name)
        out, stack = [], [start]
        while stack:
            k = stack.pop()
            out.append(k)
            stack += children.get(k, [])
        return out

    def torque_urdf(self, q_urdf):
        """Torque each motor must produce to hold the arm against gravity (URDF frame)."""
        frames, jinfo = self._fk(q_urdf)
        tau = np.zeros(len(ARM))
        for i, jn in enumerate(ARM):
            axis, p = jinfo[jn]
            t = 0.0
            for ln in self._descendants(jn):
                if ln not in self.links:
                    continue
                m, com = self.links[ln]
                R, pl = frames[ln]
                c = pl + R @ com
                t += axis @ np.cross(c - p, m * G)
            tau[i] = -t   # motor torque = minus gravity torque
        return tau

    def gravity_torque_public(self, q_pub_deg):
        q = np.radians(np.asarray(q_pub_deg[:6], dtype=float)) * self.sign
        return self.torque_urdf(q) * self.sign
