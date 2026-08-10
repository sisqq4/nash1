"""Continuous closest approach over one physics frame."""
from __future__ import annotations

from dataclasses import dataclass

from air_combat_rl.core.coordinates import VecXZY


@dataclass(frozen=True, slots=True)
class ClosestApproach:
    distance_m: float
    fraction: float


def segment_closest_approach(
    missile_start: VecXZY,
    missile_end: VecXZY,
    target_start: VecXZY,
    target_end: VecXZY,
) -> ClosestApproach:
    """Return the minimum relative distance between simultaneous linear paths."""
    r0 = _subtract(missile_start, target_start)
    r1 = _subtract(missile_end, target_end)
    dr = _subtract(r1, r0)
    denominator = _dot(dr, dr)
    fraction = 0.0 if denominator <= 1.0e-18 else min(1.0, max(0.0, -_dot(r0, dr) / denominator))
    closest = VecXZY(r0.x + fraction * dr.x, r0.z + fraction * dr.z, r0.y + fraction * dr.y)
    return ClosestApproach(closest.norm(), fraction)


def _subtract(a: VecXZY, b: VecXZY) -> VecXZY:
    return VecXZY(a.x - b.x, a.z - b.z, a.y - b.y)


def _dot(a: VecXZY, b: VecXZY) -> float:
    return a.x * b.x + a.z * b.z + a.y * b.y
