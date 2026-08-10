import math

import pytest

from air_combat_rl.core.coordinates import VecXZY
from air_combat_rl.domain.collision.closest_approach import segment_closest_approach
from air_combat_rl.domain.commands import ManeuverCommand
from air_combat_rl.domain.propulsion.drag import atmosphere, missile_drag, zero_lift_drag_coefficient
from air_combat_rl.domain.states import FlightPathAngles, KinematicState
from air_combat_rl.simulation.scenarios.factory import ScenarioConfig, build_scenario
from air_combat_rl.simulation.world import MissileRuntimeState, WorldConfig


def test_documented_scenario_defaults_and_annular_sector_sampling():
    config = ScenarioConfig(seed=12)
    assert config.blue_speed_mps == (300.0, 400.0)
    assert config.blue_heading_deg == (-180.0, 180.0)
    assert config.missile_initial_mach == (0.6, 0.9)
    assert config.missile_spawn_distance_m == (140_000.0, 160_000.0)
    assert config.missile_spawn_bearing_deg == (-30.0, 30.0)
    world = build_scenario(config)
    missile = world.missiles[0]
    radius = math.hypot(missile.kinematics.position.x, missile.kinematics.position.z)
    bearing = math.degrees(math.atan2(missile.kinematics.position.z, missile.kinematics.position.x))
    assert 300.0 <= world.blue.kinematics.speed <= 400.0
    assert 140_000.0 <= radius <= 160_000.0
    assert -30.0 <= bearing <= 30.0
    assert 8_000.0 <= missile.kinematics.position.y <= 10_000.0
    assert 0.6 * 295.0 <= missile.kinematics.speed <= 0.9 * 295.0
    direct_heading = math.atan2(-missile.kinematics.position.z, -missile.kinematics.position.x)
    offset = (missile.kinematics.angles.psi - direct_heading + math.pi) % (2 * math.pi) - math.pi
    assert abs(math.degrees(offset)) <= 15.0


def test_documented_missile_defaults():
    config = WorldConfig()
    assert config.missile_dry_mass_kg == 120.0
    assert config.missile_propellant_mass_kg == 45.0
    assert config.missile_boost_time_s == 7.0
    assert config.missile_boost_target_mach == 6.0
    assert config.navigation_constant == 3.5
    assert config.max_guidance_time_s == 180.0
    assert config.seeker_acquisition_fov_deg == 35.0
    assert config.seeker_tracking_fov_deg == 60.0
    assert config.seeker_lock_hold_time_s == 0.75
    assert config.missile_max_load_g == 35.0
    assert config.passed_distance_growth_m == 600.0
    assert config.passed_receding_speed_mps == 40.0


def test_boost_reaches_target_speed_and_consumes_all_fuel():
    world = build_scenario(ScenarioConfig(physics_dt=0.01, policy_dt=0.1))
    for _ in range(70):
        world.step_policy_interval(ManeuverCommand(0.0, 1.0, 0.0))
    missile = world.missiles[0]
    runtime = world.missile_runtime[0]
    assert not missile.powered
    assert missile.kinematics.speed == pytest.approx(6.0 * 295.0, rel=1e-6)
    assert runtime.fuel_mass_kg == 0.0
    assert runtime.mass_kg == 120.0


def test_atmospheric_mach_drag_and_fixed_cd_override():
    sea_density, sea_sound = atmosphere(0.0)
    high_density, high_sound = atmosphere(10_000.0)
    assert high_density < sea_density
    assert high_sound < sea_sound
    assert zero_lift_drag_coefficient(1.2) > zero_lift_drag_coefficient(3.0)
    result = missile_drag(900.0, 10_000.0, 120.0, 0.028, 0.08, 10.0)
    fixed = missile_drag(900.0, 10_000.0, 120.0, 0.028, 0.08, 10.0, 0.1)
    assert result.mach > 1.0 and result.cd > result.cd0
    assert fixed.cd0 == 0.1


def test_continuous_closest_approach_detects_between_sample_crossing():
    result = segment_closest_approach(
        VecXZY(-10.0, 0.0, 10_000.0), VecXZY(10.0, 0.0, 10_000.0),
        VecXZY(0.0, 0.0, 10_000.0), VecXZY(0.0, 0.0, 10_000.0),
    )
    assert result.distance_m == pytest.approx(0.0)
    assert result.fraction == pytest.approx(0.5)


def test_seeker_transitions_from_hold_to_inertial_and_keeps_prediction():
    world = build_scenario(ScenarioConfig(seed=3))
    missile = KinematicState(VecXZY(1_000.0, 0.0, 10_000.0), 500.0, FlightPathAngles(0.0, 0.0))
    runtime = MissileRuntimeState(
        seeker_mode="locked", last_lock_time_s=0.0, estimated_target=world.blue.kinematics,
    )
    world.time_s = 0.5
    held_target, locked = world._seeker_target(missile, runtime)
    assert not locked and runtime.seeker_mode == "lock_hold" and held_target is not None
    world.time_s = 1.0
    inertial_target, locked = world._seeker_target(missile, runtime)
    assert not locked and runtime.seeker_mode == "inertial" and inertial_target is not None


def test_powered_missile_does_not_start_with_a_preexisting_seeker_lock():
    world = build_scenario(ScenarioConfig(seed=5))
    assert world.missile_runtime[0].seeker_mode == "boost"
    assert world.missile_runtime[0].last_lock_time_s is None


def test_empty_world_is_immediately_safe():
    world = build_scenario(ScenarioConfig(seed=7))
    world.missiles.clear()
    world.missile_runtime.clear()
    world.min_missile_distances.clear()
    world.closest_approach_passed.clear()
    assert world.all_live_threats_safely_passed()
