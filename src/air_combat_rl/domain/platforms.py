from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class PlatformSpec:
    name: str
    min_speed: float
    max_speed: float
    min_altitude: float
    max_altitude: float
