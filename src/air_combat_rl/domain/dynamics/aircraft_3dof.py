"""3-DoF point-mass aircraft dynamics in the project XZY convention."""
from __future__ import annotations

import math
from air_combat_rl.core.math3d import euler_step, flight_velocity, safe_cos_gamma, safe_speed
from air_combat_rl.core.units import STANDARD_GRAVITY
from air_combat_rl.domain.commands import ManeuverCommand
from air_combat_rl.domain.states import FlightPathAngles, KinematicState


def aircraft_derivative(state: KinematicState, command: ManeuverCommand) -> KinematicState:
    """Return d[x,z,y,V,gamma,psi]/dt for the aircraft 3-DoF model."""
    velocity = flight_velocity(state)
    speed = safe_speed(state.speed)
    gamma = state.angles.gamma
    cos_gamma = safe_cos_gamma(gamma)
    return KinematicState(
        position=velocity,
        speed=STANDARD_GRAVITY * (command.nx - math.sin(gamma)),
        angles=FlightPathAngles(
            gamma=STANDARD_GRAVITY / speed * (command.nf * math.cos(command.gamma_s) - math.cos(gamma)),
            psi=STANDARD_GRAVITY * command.nf * math.sin(command.gamma_s) / (speed * cos_gamma),
        ),
    )


def integrate_aircraft(state: KinematicState, command: ManeuverCommand, dt: float) -> KinematicState:
    return euler_step(state, lambda s: aircraft_derivative(s, command), dt)
