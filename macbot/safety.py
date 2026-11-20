"""Motor watchdog shared by the run and collect control loops."""
from __future__ import annotations

import math

from .i18n import tr


class MotorWatch:
    def __init__(self, config):
        self.enabled = config["enabled"]
        if not isinstance(self.enabled, bool):
            raise SystemExit("safety.enabled must be true or false")
        for key in ("track_deg", "track_s", "hold_warn_s", "temp_warn_c", "temp_alarm_c", "temp_resume_c"):
            value = float(config[key])
            if not math.isfinite(value) or value <= 0:
                raise SystemExit(f"safety.{key} must be positive and finite")
            setattr(self, key, value)
        if not self.temp_resume_c < self.temp_warn_c < self.temp_alarm_c:
            raise SystemExit("safety temperatures must be ordered: resume < warning < alarm")
        self.names, self.arm = (), ()
        self.samples = ()
        self.above = {}
        self.hot = {}
        self.trip = None
        self.pose = None
        self.hold_since = self.hold_target = self.last = None
        self.rest = False

    def set_joints(self, names, gripper):
        self.names = tuple(names)
        self.arm = tuple(i for i, n in enumerate(names) if n != gripper)

    def reset_motion(self):
        self.above.clear()
        self.hold_since = self.hold_target = self.last = None
        self.rest = False

    def update(self, tick, q, cmd, samples, *, holding=False, home=None):
        """Check the last applied command against the new feedback before sending any new command."""
        self.samples = samples
        if not self.enabled:
            return []
        if self.last is not None and tick - self.last > 0.1:
            self.reset_motion()
        self.last = tick
        events, trips = [], []
        for name, motor in zip(self.names, samples, strict=True):
            temp = motor.hottest
            if temp is not None:
                if temp > self.temp_warn_c:
                    if name not in self.hot:
                        events.append(dict(kind="temperature_warning", joint=name, temp_c=temp))
                    self.hot[name] = temp
                elif name in self.hot:
                    if temp < self.temp_resume_c:
                        del self.hot[name]
                    else:
                        self.hot[name] = temp
            if self.trip is None:
                if motor.status or motor.fault:
                    trips.append(dict(kind="fault", joint=name, status=motor.status, fault=motor.fault,
                                      warning=motor.warning))
                if temp is not None and temp > self.temp_alarm_c:
                    trips.append(dict(kind="temperature_alarm", joint=name, temp_c=temp))
        if self.trip is not None:
            return events
        for i in self.arm:
            err = abs(cmd[i] - q[i])
            if err > self.track_deg:
                since = self.above.setdefault(i, tick)
                if tick - since > self.track_s:
                    trips.append(dict(kind="tracking", joint=self.names[i], error_deg=err, sustained_s=tick - since))
            else:
                self.above.pop(i, None)
        if trips:
            self.trip, self.pose = trips[0], tuple(q)
            self.rest = False
            return events + trips
        # A fixed command away from home is a proxy for a loaded hold, not a load measurement.
        holding = holding and (home is None or any(abs(cmd[i] - home[i]) > 2 for i in self.arm))
        if not holding:
            self.hold_since = self.hold_target = None
            self.rest = False
        elif self.hold_since is None or any(abs(cmd[i] - self.hold_target[i]) > 1 for i in self.arm):
            self.hold_since, self.hold_target = tick, tuple(cmd)
            self.rest = False
        elif not self.rest and tick - self.hold_since > self.hold_warn_s:
            self.rest = True
            events.append(dict(kind="static_hold", sustained_s=tick - self.hold_since))
        return events

    def blocked(self):
        if self.trip:
            return trip_message(self.trip)
        if self.hot:
            n = max(self.hot, key=self.hot.get)
            return tr(f"{n}: motor hot. Start is locked until it cools below {self.temp_resume_c:g} °C.",
                      f"{n}：电机温度偏高。低于 {self.temp_resume_c:g} °C 后才能开始。")
        return None

    def status(self):
        temps = [m.hottest for m in self.samples]
        levels = ["alarm" if m.status or m.fault or (t is not None and t > self.temp_alarm_c)
                  else "warning" if m.warning or (t is not None and t > self.temp_warn_c)
                  else "" for m, t in zip(self.samples, temps, strict=True)]
        return {"temp_c": temps, "levels": levels, "blocked": self.blocked(), "tripped": self.trip is not None,
                "rest": tr("Move to the start pose to rest the motors.", "请回到起始姿态，让电机休息。") if self.rest else None}


def trip_message(event):
    name = event["joint"]
    if event["kind"] == "fault":
        detail = f"status 0x{event['status'] or 0:02x}, fault 0x{event['fault'] or 0:08x}"
        en, zh = f"motor fault ({detail})", f"电机故障（{detail}）"
    elif event["kind"] == "temperature_alarm":
        detail = f"{event['temp_c']:.1f} °C"
        en, zh = f"motor temperature {detail}", f"电机温度 {detail}"
    else:
        detail = f"{event['error_deg']:.1f}°"
        en, zh = f"sustained tracking error {detail}", f"持续跟踪误差 {detail}"
    return tr(f"Motor watchdog stopped motion: {name}, {en}. Holding the measured pose. Inspect before restarting.",
              f"电机监控已停止运动：{name}，{zh}。已切换为保持实测姿态。请检查后重启。")


def timed_events(events, now, t0):
    return [dict(e, time_s=max(0.0, now - t0), unix_s=now) for e in events]
