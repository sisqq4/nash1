"""Coordinate primitives for the project-wide XZY convention.

The serialized order is [x, z, y], where y is altitude/up. Code should use
named fields instead of positional indexing for semantic access.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class VecXZY:
    """Three-dimensional vector serialized as [x, z, y]."""

    x: float
    z: float
    y: float

    def as_xzy(self) -> tuple[float, float, float]:
        return (self.x, self.z, self.y)

    @classmethod
    def from_xzy(cls, values: tuple[float, float, float] | list[float]) -> "VecXZY":
        if len(values) != 3:
            raise ValueError("VecXZY requires exactly three values in [x,z,y] order")
        return cls(float(values[0]), float(values[1]), float(values[2]))

    @property
    def altitude(self) -> float:
        return self.y

    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.z * self.z + self.y * self.y)


def normalize_angle_rad(angle: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
