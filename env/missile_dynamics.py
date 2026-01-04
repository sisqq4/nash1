
"""Basic kinematics for blue aircraft and red missiles."""

from typing import Tuple
import numpy as np


def update_blue_state(
    pos: np.ndarray,
    vel: np.ndarray,
    action: np.ndarray,
    dt: float,
    accel_mag: float,
    v_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Update blue aircraft state with a simple 3D acceleration model.

    Actions:
        action: [nx, ny, roll, pitch] from the blue action library.
        - nx: tangential load factor (forward acceleration)
        - ny: normal load factor (1.0 keeps level flight in the original model)
        - roll: bank angle (rad) to rotate the normal load in the right/up plane
        - pitch: placeholder (unused in this simplified model)
    """
    pos = pos.astype(float)
    vel = vel.astype(float)

    action = np.asarray(action, dtype=float).reshape(-1)
    if action.shape[0] != 4:
        raise ValueError("blue action must be a 4D vector [nx, ny, roll, pitch].")

    nx, ny, roll, _ = action
    ny_eff = ny - 1.0

    speed = np.linalg.norm(vel)
    if speed < 1e-6:
        forward = np.array([1.0, 0.0, 0.0])
    else:
        forward = vel / speed

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)

    if right_norm < 1e-6:
        right = np.array([0.0, 1.0, 0.0])
        right_norm = 1.0

    right = right / right_norm
    up = np.cross(right, forward)
    up_norm = np.linalg.norm(up)
    if up_norm > 1e-6:
        up = up / up_norm

    normal_dir = np.cos(roll) * up + np.sin(roll) * right
    a = accel_mag * (nx * forward + ny_eff * normal_dir)

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
    missile_speed: float | np.ndarray,
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

    speed_array = np.asarray(missile_speed, dtype=float)
    if speed_array.shape == ():
        speed_array = np.full(M, float(speed_array))
    else:
        assert speed_array.shape[0] == M, "missile_speed array must have shape (M,)"

    for i in range(M):
        p = new_pos[i]
        v = new_vel[i]
        N_gain = float(nav_array[i])
        target_speed = float(speed_array[i])

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

        v = u * target_speed
        p = p + v * dt

        new_pos[i] = p
        new_vel[i] = v

    return new_pos, new_vel
