"""3-DoF point-mass missile dynamics and proportional-navigation interface."""
from __future__ import annotations

import math
from dataclasses import dataclass
from air_combat_rl.core.coordinates import VecXZY
from air_combat_rl.core.math3d import euler_step, flight_velocity, global_to_local, safe_cos_gamma, safe_speed
from air_combat_rl.core.units import STANDARD_GRAVITY
from air_combat_rl.domain.states import FlightPathAngles, KinematicState


@dataclass(frozen=True, slots=True)
class MissileCommand:
    """Missile overload command in g units: tangential, normal, and lateral."""

    nx: float
    nn: float
    ns: float


@dataclass(frozen=True, slots=True)
class ProportionalNavigationCommand:
    navigation_constant: float = 3.0
    closing_speed: float = 0.0
    los_rate_gamma: float = 0.0
    los_rate_psi: float = 0.0
    tangential_g: float = 0.0

    def to_missile_command(self, missile: KinematicState) -> MissileCommand:
        speed = safe_speed(missile.speed)
        normal = self.navigation_constant * self.closing_speed * self.los_rate_gamma / (STANDARD_GRAVITY * speed)
        lateral = self.navigation_constant * self.closing_speed * self.los_rate_psi / (STANDARD_GRAVITY * speed)
        return MissileCommand(nx=self.tangential_g, nn=normal, ns=lateral)


def proportional_navigation_command(missile: KinematicState, target: KinematicState, navigation_constant: float = 3.0, tangential_g: float = 0.0) -> MissileCommand:
    rel = VecXZY(
        target.position.x - missile.position.x,
        target.position.z - missile.position.z,
        target.position.y - missile.position.y,
    )
    rel_local = global_to_local(rel, missile.angles)
    distance2 = max(rel.norm() ** 2, 1.0e-12)
    missile_vel = flight_velocity(missile)
    target_vel = flight_velocity(target)
    rel_vel = VecXZY(target_vel.x - missile_vel.x, target_vel.z - missile_vel.z, target_vel.y - missile_vel.y)
    closing = -((rel.x * rel_vel.x + rel.z * rel_vel.z + rel.y * rel_vel.y) / max(rel.norm(), 1.0e-6))
    los_rate_gamma = (rel_local.x * rel_vel.y - rel_local.y * rel_vel.x) / distance2
    los_rate_psi = (rel_local.x * rel_vel.z - rel_local.z * rel_vel.x) / distance2
    return ProportionalNavigationCommand(navigation_constant, closing, los_rate_gamma, los_rate_psi, tangential_g).to_missile_command(missile)


def missile_derivative(state: KinematicState, command: MissileCommand) -> KinematicState:
    velocity = flight_velocity(state)
    speed = safe_speed(state.speed)
    gamma = state.angles.gamma
    return KinematicState(
        position=velocity,
        speed=STANDARD_GRAVITY * (command.nx - math.sin(gamma)),
        angles=FlightPathAngles(
            gamma=STANDARD_GRAVITY / speed * (command.nn - math.cos(gamma)),
            psi=STANDARD_GRAVITY * command.ns / (speed * safe_cos_gamma(gamma)),
        ),
    )


def integrate_missile(state: KinematicState, command: MissileCommand, dt: float) -> KinematicState:
    return euler_step(state, lambda s: missile_derivative(s, command), dt)
