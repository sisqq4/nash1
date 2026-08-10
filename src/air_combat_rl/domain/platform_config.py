"""Validated platform parameters loaded by application runtimes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class PlatformConfig:
    name: str
    min_speed: float
    max_speed: float
    min_altitude: float
    max_altitude: float
    max_g: float
    low_speed: float
    high_speed: float
    low_altitude: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PlatformConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("platform config must be a YAML mapping")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown platform fields: {sorted(unknown)}")
        result = cls(**data)
        if not 0 < result.min_speed < result.low_speed < result.high_speed < result.max_speed:
            raise ValueError("platform speeds must satisfy min < low < high < max")
        if not 0 <= result.min_altitude < result.low_altitude < result.max_altitude:
            raise ValueError("platform altitudes must satisfy min <= low < max")
        if result.max_g <= 0:
            raise ValueError("platform max_g must be positive")
        return result
