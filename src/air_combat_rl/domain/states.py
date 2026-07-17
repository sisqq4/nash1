"""Domain state contracts independent of RL algorithms and rewards."""
from __future__ import annotations

from dataclasses import dataclass
from air_combat_rl.core.coordinates import VecXZY


@dataclass(frozen=True, slots=True)
class FlightPathAngles:
    gamma: float
    psi: float


@dataclass(frozen=True, slots=True)
class KinematicState:
    position: VecXZY
    speed: float
    angles: FlightPathAngles


@dataclass(frozen=True, slots=True)
class AircraftState:
    kinematics: KinematicState
    alive: bool
    platform: str


@dataclass(frozen=True, slots=True)
class MissileState:
    kinematics: KinematicState
    alive: bool
    locked: bool
    powered: bool
    age_s: float
