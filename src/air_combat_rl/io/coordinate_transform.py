"""Local XZY simulation coordinates to geodetic coordinates."""
from __future__ import annotations

from dataclasses import dataclass
import math


EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True, slots=True)
class GeodeticOrigin:
    """Required origin for the local north/east/up simulation frame."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (
            self.latitude_deg, self.longitude_deg, self.altitude_m
        )):
            raise ValueError("origin latitude, longitude, and altitude must be finite")
        if not (-90.0 <= self.latitude_deg <= 90.0):
            raise ValueError("origin latitude must be in [-90, 90] degrees")
        if not (-180.0 <= self.longitude_deg <= 180.0):
            raise ValueError("origin longitude must be in [-180, 180] degrees")
        if abs(math.cos(math.radians(self.latitude_deg))) < 1e-12:
            raise ValueError("local longitude conversion is undefined at the poles")


def xzy_to_geodetic(
    x_m: float, z_m: float, y_m: float, origin: GeodeticOrigin
) -> tuple[float, float, float]:
    """Convert north/east/up XZY coordinates using a small-area approximation."""
    x_m, z_m, y_m = float(x_m), float(z_m), float(y_m)
    if not all(math.isfinite(value) for value in (x_m, z_m, y_m)):
        raise ValueError("XZY position values must be finite")
    latitude = origin.latitude_deg + math.degrees(x_m / EARTH_RADIUS_M)
    longitude = origin.longitude_deg + math.degrees(
        z_m / (EARTH_RADIUS_M * math.cos(math.radians(origin.latitude_deg)))
    )
    return latitude, longitude, origin.altitude_m + y_m


def heading_deg(psi_rad: float) -> float:
    """Map north-zero, clockwise-positive simulation heading to [0, 360)."""
    psi_rad = float(psi_rad)
    if not math.isfinite(psi_rad):
        raise ValueError("psi must be finite")
    return math.degrees(psi_rad) % 360.0


def pitch_deg(gamma_rad: float) -> float:
    """Map positive-climb flight-path angle to Tacview pitch degrees."""
    gamma_rad = float(gamma_rad)
    if not math.isfinite(gamma_rad):
        raise ValueError("gamma must be finite")
    return math.degrees(gamma_rad)
