"""Minimal 3-DoF point-mass aircraft propagation with no RL dependencies."""
from __future__ import annotations

import math
from air_combat_rl.core.coordinates import VecXZY, normalize_angle_rad
from air_combat_rl.domain.commands import ManeuverCommand
from air_combat_rl.domain.states import FlightPathAngles, KinematicState

G0 = 9.80665


def integrate_aircraft(state: KinematicState, command: ManeuverCommand, dt: float) -> KinematicState:
    speed = max(1e-6, state.speed + command.nx * G0 * dt)
    gamma = normalize_angle_rad(state.angles.gamma + command.nz * G0 / speed * dt)
    psi = normalize_angle_rad(state.angles.psi + command.ny * G0 / max(speed * math.cos(gamma), 1e-6) * dt)
    dx = speed * math.cos(gamma) * math.cos(psi) * dt
    dz = speed * math.cos(gamma) * math.sin(psi) * dt
    dy = speed * math.sin(gamma) * dt
    return KinematicState(
        position=VecXZY(state.position.x + dx, state.position.z + dz, state.position.y + dy),
        speed=speed,
        angles=FlightPathAngles(gamma=gamma, psi=psi),
    )
