
"""Basic kinematics for blue aircraft and red missiles."""

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
    """Update blue aircraft state with a simple 3D acceleration model.

    Actions:
        0: keep current acceleration (no change)
        1: +x
        2: -x
        3: +y
        4: -y
        5: +z
        6: -z
    """
    pos = pos.astype(float)
    vel = vel.astype(float)

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

    vel = vel + a * dt

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
    """Proportional-navigation-like update for a batch of missiles.

    Args:
        missile_pos: (M, 3)
        missile_vel: (M, 3)
        blue_pos: (3,)
        blue_vel: (3,)
        missile_speed: scalar speed (kept constant) [km/s]
        dt: [s]
        nav_gain: scalar or shape (M,) navigation gain(s)
    """
    M = missile_pos.shape[0]
    new_pos = missile_pos.astype(float).copy()
    new_vel = missile_vel.astype(float).copy()
    blue_pos = blue_pos.astype(float).reshape(1, 3)

    nav_array = np.asarray(nav_gain, dtype=float)
    if nav_array.shape == ():
        nav_array = np.full(M, float(nav_array))
    else:
        assert nav_array.shape[0] == M, "nav_gain array must have shape (M,)"

    for i in range(M):
        p = new_pos[i]
        v = new_vel[i]
        N_gain = float(nav_array[i])

        speed = np.linalg.norm(v)
        if speed < 1e-6:
            # If speed is zero, keep it zero (e.g. not yet launched)
            new_pos[i] = p
            new_vel[i] = v
            continue

        u = v / speed

        r = blue_pos[0] - p
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-6:
            los = u
        else:
            los = r / r_norm

        # LOS component perpendicular to velocity
        los_perp = los - np.dot(los, u) * u
        perp_norm = np.linalg.norm(los_perp)

        if perp_norm > 1e-6 and N_gain != 0.0:
            los_perp /= perp_norm
            u_new = u + N_gain * los_perp * dt
            u_norm = np.linalg.norm(u_new)
            if u_norm > 1e-6:
                u = u_new / u_norm

        v = u * missile_speed
        p = p + v * dt

        new_pos[i] = p
        new_vel[i] = v

    return new_pos, new_vel
