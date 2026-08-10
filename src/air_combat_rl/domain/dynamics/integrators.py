"""Numerical integration entry points for 3-DoF flight states."""
from __future__ import annotations

from collections.abc import Callable
from src.air_combat_rl.core.math3d import euler_step
from src.air_combat_rl.domain.states import FlightState


def integrate_euler(state: FlightState, derivative: Callable[[FlightState], FlightState], dt: float) -> FlightState:
    return euler_step(state, derivative, dt)
