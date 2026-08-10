"""Domain state contracts independent of RL algorithms and rewards."""
from __future__ import annotations

from dataclasses import dataclass
from air_combat_rl.core.coordinates import VecXZY


@dataclass(frozen=True, slots=True)
class FlightPathAngles:
    gamma: float
    psi: float


@dataclass(frozen=True, slots=True)
class FlightState:
    """3-DoF flight state in SI units and [x,z,y,V,gamma,psi] order."""

    position: VecXZY
    speed: float
    angles: FlightPathAngles

    def as_xzy_v_gamma_psi(self) -> tuple[float, float, float, float, float, float]:
        return (*self.position.as_xzy(), self.speed, self.angles.gamma, self.angles.psi)

    @classmethod
    def from_xzy_v_gamma_psi(cls, values: tuple[float, ...] | list[float]) -> "FlightState":
        if len(values) != 6:
            raise ValueError("FlightState requires [x,z,y,V,gamma,psi]")
        return cls(
            position=VecXZY.from_xzy(values[:3]),
            speed=float(values[3]),
            angles=FlightPathAngles(gamma=float(values[4]), psi=float(values[5])),
        )


KinematicState = FlightState


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
    launch_time_s: float = 0.0
