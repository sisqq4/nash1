import math

import pytest

from air_combat_rl.core.coordinates import VecXZY
from air_combat_rl.core.math3d import flight_velocity, global_to_local, local_to_global
from air_combat_rl.core.units import STANDARD_GRAVITY
from air_combat_rl.domain.commands import ManeuverCommand
from air_combat_rl.domain.dynamics.aircraft_3dof import integrate_aircraft
from air_combat_rl.domain.dynamics.missile_3dof import MissileCommand, integrate_missile
from air_combat_rl.domain.states import FlightPathAngles, FlightState


def state(speed=100.0, gamma=0.0, psi=0.0):
    return FlightState(VecXZY(0.0, 0.0, 1000.0), speed, FlightPathAngles(gamma, psi))


def assert_vec_close(vec, expected, tol=1e-9):
    assert vec.x == pytest.approx(expected.x, abs=tol)
    assert vec.z == pytest.approx(expected.z, abs=tol)
    assert vec.y == pytest.approx(expected.y, abs=tol)


def test_horizontal_constant_speed_straight_kinematics():
    vel = flight_velocity(state(speed=250.0))
    assert_vec_close(vel, VecXZY(250.0, 0.0, 0.0))


def test_positive_gamma_climbs_in_y():
    vel = flight_velocity(state(gamma=math.radians(30.0)))
    assert vel.y > 0.0


def test_negative_gamma_dives_in_y():
    vel = flight_velocity(state(gamma=math.radians(-30.0)))
    assert vel.y < 0.0


def test_positive_psi_moves_toward_positive_z():
    vel = flight_velocity(state(psi=math.radians(45.0)))
    assert vel.z > 0.0


def test_global_to_local_coordinate_transform():
    angles = FlightPathAngles(gamma=0.0, psi=math.pi / 2.0)
    local = global_to_local(VecXZY(0.0, 10.0, 5.0), angles)
    assert_vec_close(local, VecXZY(10.0, 0.0, 5.0))


def test_local_to_global_inverse_transform():
    angles = FlightPathAngles(gamma=math.radians(20.0), psi=math.radians(35.0))
    original = VecXZY(10.0, -3.0, 2.0)
    assert_vec_close(local_to_global(global_to_local(original, angles), angles), original, tol=1e-8)


def test_aircraft_single_step_integration_uses_3dof_equations():
    start = state(speed=100.0)
    result = integrate_aircraft(start, ManeuverCommand(nx=0.0, nf=1.0, gamma_s=0.0), 1.0)
    assert result.position.x == pytest.approx(100.0)
    assert result.speed == pytest.approx(100.0)
    assert result.angles.gamma == pytest.approx(0.0)
    assert result.angles.psi == pytest.approx(0.0)


def test_missile_single_step_integration_uses_3dof_equations():
    start = state(speed=200.0)
    result = integrate_missile(start, MissileCommand(nx=0.0, nn=1.0, ns=0.0), 0.5)
    assert result.position.x == pytest.approx(100.0)
    assert result.speed == pytest.approx(200.0)
    assert result.angles.gamma == pytest.approx(0.0)
    assert result.angles.psi == pytest.approx(0.0)


def test_numerical_boundaries_do_not_crash_near_zero_speed_or_vertical():
    near_boundary = state(speed=0.0, gamma=math.pi / 2.0, psi=0.0)
    result = integrate_aircraft(near_boundary, ManeuverCommand(nx=0.0, nf=1.0, gamma_s=math.pi / 2.0), 0.01)
    assert math.isfinite(result.speed)
    assert math.isfinite(result.angles.psi)
    assert result.speed > 0.0


def test_si_units_are_meters_seconds_mps_radians():
    start = state(speed=10.0, gamma=math.asin(0.5), psi=0.0)
    result = integrate_missile(start, MissileCommand(nx=0.5, nn=math.cos(start.angles.gamma), ns=0.0), 2.0)
    assert result.position.x == pytest.approx(10.0 * math.cos(start.angles.gamma) * 2.0)
    assert result.position.y == pytest.approx(1000.0 + 10.0 * math.sin(start.angles.gamma) * 2.0)
    assert result.speed == pytest.approx(10.0)
    assert STANDARD_GRAVITY == pytest.approx(9.80665)
