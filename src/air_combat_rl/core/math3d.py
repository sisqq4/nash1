"""Math helpers for 3-DoF XZY flight dynamics."""
from __future__ import annotations

import math
from collections.abc import Callable
from air_combat_rl.core.coordinates import VecXZY, normalize_angle_rad
from air_combat_rl.core.units import EPS_COS_GAMMA, EPS_SPEED
from air_combat_rl.domain.states import FlightPathAngles, FlightState


def safe_speed(speed: float) -> float:
    return max(abs(speed), EPS_SPEED)


def safe_cos_gamma(gamma: float) -> float:
    c = math.cos(gamma)
    if abs(c) >= EPS_COS_GAMMA:
        return c
    return math.copysign(EPS_COS_GAMMA, c if c != 0.0 else 1.0)


def flight_velocity(state: FlightState) -> VecXZY:
    v = state.speed
    gamma = state.angles.gamma
    psi = state.angles.psi
    return VecXZY(v * math.cos(gamma) * math.cos(psi), v * math.cos(gamma) * math.sin(psi), v * math.sin(gamma))


def global_to_local(vector: VecXZY, angles: FlightPathAngles) -> VecXZY:
    gamma = angles.gamma
    psi = angles.psi
    forward = VecXZY(math.cos(gamma) * math.cos(psi), math.cos(gamma) * math.sin(psi), math.sin(gamma))
    right = VecXZY(-math.sin(psi), math.cos(psi), 0.0)
    up = VecXZY(-math.sin(gamma) * math.cos(psi), -math.sin(gamma) * math.sin(psi), math.cos(gamma))
    return VecXZY(_dot(vector, forward), _dot(vector, right), _dot(vector, up))


def local_to_global(vector: VecXZY, angles: FlightPathAngles) -> VecXZY:
    gamma = angles.gamma
    psi = angles.psi
    forward = VecXZY(math.cos(gamma) * math.cos(psi), math.cos(gamma) * math.sin(psi), math.sin(gamma))
    right = VecXZY(-math.sin(psi), math.cos(psi), 0.0)
    up = VecXZY(-math.sin(gamma) * math.cos(psi), -math.sin(gamma) * math.sin(psi), math.cos(gamma))
    return VecXZY(
        vector.x * forward.x + vector.z * right.x + vector.y * up.x,
        vector.x * forward.z + vector.z * right.z + vector.y * up.z,
        vector.x * forward.y + vector.z * right.y + vector.y * up.y,
    )


def euler_step(state: FlightState, derivative: Callable[[FlightState], FlightState], dt: float) -> FlightState:
    d = derivative(state)
    return FlightState(
        VecXZY(state.position.x + d.position.x * dt, state.position.z + d.position.z * dt, state.position.y + d.position.y * dt),
        max(EPS_SPEED, state.speed + d.speed * dt),
        FlightPathAngles(normalize_angle_rad(state.angles.gamma + d.angles.gamma * dt), normalize_angle_rad(state.angles.psi + d.angles.psi * dt)),
    )


def _dot(a: VecXZY, b: VecXZY) -> float:
    return a.x * b.x + a.z * b.z + a.y * b.y
