
"""Aircraft and radar-missile classes for the 3D escape game.

These classes wrap the existing kinematic update functions so that
the environment can treat the blue aircraft and red missiles as
objects, while keeping all physical parameters and behaviour
identical to the earlier implementation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .missile_dynamics import update_blue_state, update_missiles_pn


class Aircraft:
    """Blue aircraft model (evasive target)."""

    def __init__(self, dt: float, accel_mag: float, v_max: float) -> None:
        self.dt = float(dt)
        self.accel_mag = float(accel_mag)
        self.v_max = float(v_max)

    def step(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        action: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update aircraft state according to the chosen action."""
        pos, vel = update_blue_state(
            pos,
            vel,
            int(action),
            dt=self.dt,
            accel_mag=self.accel_mag,
            v_max=self.v_max,
        )
        return pos, vel


class Missiles:
    """Radar missile model (pursuit attacker).

    This class updates a *batch* of missiles in one call. Per-missile
    position, velocity and navigation gains are still stored and
    managed by the environment for compatibility with the original
    vectorised implementation.
    """

    def __init__(self, dt: float, speed: float) -> None:
        self.dt = float(dt)
        self.speed = float(speed)

    def step(
        self,
        missile_pos: np.ndarray,
        missile_vel: np.ndarray,
        blue_pos: np.ndarray,
        blue_vel: np.ndarray,
        nav_gains: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return update_missiles_pn(
            missile_pos,
            missile_vel,
            blue_pos,
            blue_vel,
            self.speed,
            self.dt,
            nav_gains,
        )
