
"""Simple point-mass kinematics for the blue aircraft and red missiles.

This module contains small helper functions that the environment uses to
propagate the states forward in time. The kinematics are intentionally simple:
- the blue aircraft is modeled as a point mass with bounded speed and
  discrete acceleration actions;
- each missile steers directly towards the current blue position at a fixed
  speed (pure pursuit).
"""

from typing import Tuple
import numpy as np


def update_blue_state(
    pos: np.ndarray,
    vel: np.ndarray,
    action: int,
    dt: float,
    accel_mag: float,
    v_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Update the blue aircraft state.

    Args:
        pos: (3,) current position
        vel: (3,) current velocity
        action: discrete action id in {0..6}
        dt: time step
        accel_mag: magnitude of acceleration for non-zero actions
        v_max: maximum speed

    Returns:
        new_pos, new_vel
    """
    pos = pos.astype(float)
    vel = vel.astype(float)

    # Map discrete action to acceleration vector.
    # 0: no acceleration, 1..6: +/- x,y,z directions.
    a = np.zeros(3, dtype=float)
    if action == 1:
        a[0] = accel_mag
    elif action == 2:
        a[0] = -accel_mag
    elif action == 3:
        a[1] = accel_mag
    elif action == 4:
        a[1] = -accel_mag
    elif action == 5:
        a[2] = accel_mag
    elif action == 6:
        a[2] = -accel_mag
    # else action 0 -> zero acceleration

    # Integrate velocity and position (Euler).
    vel = vel + a * dt

    # Enforce speed limit.
    speed = np.linalg.norm(vel)
    if speed > v_max and speed > 1e-8:
        vel = vel / speed * v_max

    pos = pos + vel * dt
    return pos, vel


def update_missiles_towards_blue(
    missile_pos: np.ndarray,
    blue_pos: np.ndarray,
    missile_speed: float,
    dt: float,
) -> np.ndarray:
    """Move all missiles one step towards the blue aircraft.

    Args:
        missile_pos: (M, 3) missile positions
        blue_pos: (3,) blue aircraft position
        missile_speed: scalar missile speed
        dt: time step

    Returns:
        new_missile_pos: (M, 3)
    """
    missile_pos = missile_pos.astype(float)
    blue_pos = blue_pos.astype(float).reshape(1, 3)

    diff = blue_pos - missile_pos
    dist = np.linalg.norm(diff, axis=1, keepdims=True)  # (M,1)
    # Avoid division by zero
    dist = np.maximum(dist, 1e-6)

    direction = diff / dist
    step = direction * (missile_speed * dt)
    return missile_pos + step
