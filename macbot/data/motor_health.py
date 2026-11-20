"""NPZ fields shared by trial and episode writers."""
from __future__ import annotations

import time
from dataclasses import asdict, fields

import numpy as np

from ..robot.telemetry import MotorSample
from ..system import ensure_dir


def diagnostic_path(out, prefix):
    return ensure_dir(out / "watchdog") / f"{prefix}_{time.time_ns()}.npz"


def record_motor_meta(rec, meta):
    meta["watchdog_events"] = rec.get("watchdog_events", [])
    if "motor_at_trip" in rec:
        meta["motor_at_trip"] = [asdict(m) for m in rec["motor_at_trip"]]


def motor_arrays(rows, num_joints):
    out = {}
    for field in fields(MotorSample):
        name = field.name
        integer = name in ("status", "fault", "warning")
        missing = -1 if integer else np.nan
        out[f"motor_{name}"] = np.array(
            [[missing if (v := getattr(m, name)) is None else v for m in row] for row in rows],
            dtype=np.int64 if integer else np.float32,
        ).reshape(len(rows), num_joints)
    return out


def print_motor_health(d, meta):
    names = meta.get("joint_names", [])
    peak = {}
    for key in ("motor_temp_c", "motor_rotor_c"):
        if key in d.files:
            for name, col in zip(names, d[key].T, strict=True):
                valid = col[np.isfinite(col)]
                if valid.size:
                    peak[name] = max(peak.get(name, -np.inf), float(valid.max()))
    for name, motor in zip(names, meta.get("motor_at_trip", []), strict=False):
        temps = [motor[k] for k in ("temp_c", "rotor_c") if motor[k] is not None]
        if temps:
            peak[name] = max(peak.get(name, -np.inf), *temps)
    temps = ", ".join(f"{n} {peak[n]:.1f} °C" for n in names if n in peak)
    print(f"    motor maxima: {temps or 'unavailable'}")
    events = meta.get("watchdog_events")
    status = ("disabled" if meta.get("safety", {}).get("enabled") is False else
              "not recorded" if events is None else "no events" if not events else "")
    print(f"    watchdog: {status}".rstrip())
    for event in events or ():
        where = f" {event['joint']}" if "joint" in event else ""
        details = []
        for key in ("error_deg", "temp_c", "status", "fault", "warning"):
            value = event.get(key)
            if value is not None:
                details.append(f"{key}=0x{value:x}" if isinstance(value, int) else f"{key}={value:.1f}")
        print(f"      {event['time_s']:.3f} s {event['kind']}{where} {' '.join(details)}".rstrip())
