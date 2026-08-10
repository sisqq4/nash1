"""Low-fidelity atmosphere and missile aerodynamic drag."""
from __future__ import annotations

from dataclasses import dataclass
import bisect
import math


@dataclass(frozen=True, slots=True)
class DragResult:
    acceleration_mps2: float
    mach: float
    cd0: float
    cd: float


_MACH_POINTS = (0.0, 0.8, 1.0, 1.2, 2.0, 3.0, 4.0, 6.0, 8.0)
_CD0_POINTS = (0.18, 0.18, 0.35, 0.42, 0.30, 0.25, 0.22, 0.20, 0.19)


def atmosphere(altitude_m: float) -> tuple[float, float]:
    """Return ISA density and speed of sound for the lower atmosphere."""
    h = min(20_000.0, max(0.0, altitude_m))
    if h <= 11_000.0:
        temperature = 288.15 - 0.0065 * h
        pressure = 101_325.0 * (temperature / 288.15) ** 5.2558797
    else:
        temperature = 216.65
        pressure = 22_632.06 * math.exp(-9.80665 * (h - 11_000.0) / (287.05287 * temperature))
    density = pressure / (287.05287 * temperature)
    sound_speed = math.sqrt(1.4 * 287.05287 * temperature)
    return density, sound_speed


def zero_lift_drag_coefficient(mach: float) -> float:
    if mach <= _MACH_POINTS[0]:
        return _CD0_POINTS[0]
    if mach >= _MACH_POINTS[-1]:
        return _CD0_POINTS[-1]
    index = bisect.bisect_right(_MACH_POINTS, mach) - 1
    ratio = (mach - _MACH_POINTS[index]) / (_MACH_POINTS[index + 1] - _MACH_POINTS[index])
    return _CD0_POINTS[index] + ratio * (_CD0_POINTS[index + 1] - _CD0_POINTS[index])


def missile_drag(
    speed_mps: float,
    altitude_m: float,
    mass_kg: float,
    reference_area_m2: float,
    induced_drag_factor: float,
    lateral_load_g: float,
    fixed_drag_coefficient: float | None = None,
) -> DragResult:
    density, sound_speed = atmosphere(altitude_m)
    mach = speed_mps / sound_speed
    cd0 = fixed_drag_coefficient if fixed_drag_coefficient is not None else zero_lift_drag_coefficient(mach)
    dynamic_pressure = 0.5 * density * speed_mps * speed_mps
    lift_coefficient = mass_kg * 9.80665 * abs(lateral_load_g) / max(dynamic_pressure * reference_area_m2, 1.0e-9)
    cd = cd0 + induced_drag_factor * lift_coefficient * lift_coefficient
    return DragResult(dynamic_pressure * cd * reference_area_m2 / mass_kg, mach, cd0, cd)
