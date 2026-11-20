"""Motor feedback in robot joint order, independent of policy observations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MotorSample:
    temp_c: float | None = None
    rotor_c: float | None = None
    torque_nm: float | None = None
    status: int | None = None
    fault: int | None = None
    warning: int | None = None

    @property
    def hottest(self):
        if self.rotor_c is None:
            return self.temp_c
        return self.rotor_c if self.temp_c is None else max(self.temp_c, self.rotor_c)


UNKNOWN = MotorSample()


def synthetic(joints):
    return tuple(MotorSample(30.0 + i, 32.0 + i, 0.1 * i, 0, 0, 0) for i in range(len(joints)))
