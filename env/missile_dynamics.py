
"""Point-mass kinematics for the blue aircraft and red missiles.

This module contains small helper functions that the environment uses to
propagate the states forward in time.

- The blue aircraft is modeled as a point mass with bounded speed and
  discrete acceleration actions;
- Each missile uses a *proportional-navigation-like* guidance law: its
  velocity direction is continuously steered towards the line-of-sight to
  the blue aircraft, while keeping speed constant.
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

    Coordinates are in kilometers, time in seconds, speed in km/s.

    Args:
        pos: (3,) current position [km]
        vel: (3,) current velocity [km/s]
        action: discrete action id in {0..6}
        dt: time step [s]
        accel_mag: magnitude of acceleration for non-zero actions [km/s^2]
        v_max: maximum speed [km/s]

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


def update_missiles_pn(
    missile_pos: np.ndarray,
    missile_vel: np.ndarray,
    blue_pos: np.ndarray,
    blue_vel: np.ndarray,
    missile_speed: float,
    dt: float,
    nav_gain,
) -> Tuple[np.ndarray, np.ndarray]:
    """Update missiles using a simple proportional-navigation-like guidance.

    This is not a full 3DoF PN implementation, but a discrete-time heading
    update that captures the main idea: the missile's velocity direction is
    rotated towards the instantaneous line-of-sight (LOS) to the target.

    Args:
        missile_pos: (M, 3) missile positions [km]
        missile_vel: (M, 3) missile velocities [km/s]
        blue_pos: (3,) blue aircraft position [km]
        blue_vel: (3,) blue aircraft velocity [km/s] (currently unused but kept for extensibility)
        missile_speed: scalar missile speed (kept constant) [km/s]
        dt: time step [s]
        nav_gain: either a scalar gain N, or an array of shape (M,) with
            individual gains for each missile.

    Returns:
        new_missile_pos: (M, 3)
        new_missile_vel: (M, 3)
    """
    M = missile_pos.shape[0]
    new_pos = missile_pos.astype(float).copy()
    new_vel = missile_vel.astype(float).copy()
    blue_pos = blue_pos.astype(float).reshape(1, 3)

    # Determine whether nav_gain is scalar or per-missile array.
    nav_array = np.asarray(nav_gain, dtype=float)
    if nav_array.shape == ():  # scalar
        nav_array = np.full(M, float(nav_array))
    else:
        assert nav_array.shape[0] == M, "nav_gain array must have shape (M,)"

    for i in range(M):
        p = new_pos[i]
        v = new_vel[i]
        N_gain = float(nav_array[i])

        # Ensure non-zero velocity; if zero, initialize towards the target.
        speed = np.linalg.norm(v)
        if speed < 1e-6:
            direction = blue_pos[0] - p
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                direction = np.array([1.0, 0.0, 0.0])
                norm = 1.0
            u = direction / norm
            v = u * missile_speed
            speed = missile_speed
        else:
            u = v / speed

        # Line-of-sight unit vector from missile to target.
        r = blue_pos[0] - p
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-6:
            # Practically at target: keep heading.
            los = u
        else:
            los = r / r_norm

        # Component of LOS orthogonal to current velocity direction.
        # This approximates the normal acceleration direction.
        los_perp = los - np.dot(los, u) * u
        perp_norm = np.linalg.norm(los_perp)
        if perp_norm > 1e-6 and N_gain != 0.0:
            los_perp /= perp_norm
            # Heading update (discrete-time analogue of lateral acceleration),
            # scaled by navigation gain.
            u_new = u + N_gain * los_perp * dt
            u_new_norm = np.linalg.norm(u_new)
            if u_new_norm > 1e-6:
                u = u_new / u_new_norm

        # Keep constant speed.
        v = u * missile_speed

        # Integrate position.
        p = p + v * dt

        new_pos[i] = p
        new_vel[i] = v

    return new_pos, new_vel
