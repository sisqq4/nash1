import math
from pathlib import Path

import pytest
import yaml

from src.air_combat_rl.domain.commands import ManeuverCommand
from src.air_combat_rl.domain.dynamics.missile_3dof import ProportionalNavigationCommand
from src.air_combat_rl.core.units import STANDARD_GRAVITY
from src.air_combat_rl.domain.platform_config import PlatformConfig
from src.air_combat_rl.simulation.scenarios.factory import ScenarioConfig, build_scenario
from src.air_combat_rl.tasks.blue_escape.rewards.components import RewardConfig
from src.air_combat_rl.runtime import build_blue_escape_env
from src.air_combat_rl.tasks.blue_escape.continuous.wrapper import ProjectedContinuousActionWrapper


def test_platform_and_reward_yaml_parameters_are_loaded():
    platform = PlatformConfig.from_yaml("configs/platform/zdj.yaml")
    reward = RewardConfig.from_yaml("configs/reward/escape_1v1.yaml")
    assert platform.name == "zdj"
    assert platform.max_g == 9.0
    assert platform.max_speed == 600.0
    assert reward.terminal_ground == -20.0
    assert reward.safe_distance_m == 30_000.0


def test_runtime_loads_configs_outside_repository_working_directory(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    env, runtime = build_blue_escape_env(
        root / "configs/scenario/fixed_1v1.yaml",
        root / "configs/actions/blue_29.yaml",
        "zdj",
        0,
    )
    wrapped = ProjectedContinuousActionWrapper(env)
    assert runtime.platform_config.max_speed == 600.0
    assert env.reward_model.config.terminal_hit == -10.0
    assert wrapped.projector.config.max_speed_mps == 600.0


def test_multi_threat_runtime_selects_multi_reward(tmp_path):
    env, runtime = build_blue_escape_env(
        "configs/scenario/1v2.yaml", "configs/actions/blue_29.yaml", "zdj", 0
    )
    assert len(env.world.missiles) == 2
    assert runtime.reward_config_path.endswith("escape_1vn.yaml")


def test_invalid_reward_parameter_is_rejected(tmp_path):
    path = tmp_path / "reward.yaml"
    path.write_text(yaml.safe_dump({"parameters": {"short_range_m": 40_000}}))
    with pytest.raises(ValueError, match="short_range"):
        RewardConfig.from_yaml(path)


def test_red_missile_launches_upward_boosts_then_coasts_with_drag():
    world = build_scenario(
        ScenarioConfig(
            mode="powered_launch",
            missile_launch_climb_angle_deg=20.0,
            physics_dt=0.01,
            policy_dt=0.1,
        )
    )
    missile = world.missiles[0]
    initial_speed = missile.kinematics.speed
    assert missile.powered
    assert missile.kinematics.angles.gamma == pytest.approx(math.radians(20.0))

    for _ in range(71):
        world.step_policy_interval(ManeuverCommand(0.0, 1.0, 0.0))
    boosted_speed = world.missiles[0].kinematics.speed
    assert boosted_speed > initial_speed
    assert not world.missiles[0].powered

    for _ in range(10):
        world.step_policy_interval(ManeuverCommand(0.0, 1.0, 0.0))
    assert world.missiles[0].kinematics.speed < boosted_speed


def test_proportional_navigation_acceleration_has_correct_units():
    command = ProportionalNavigationCommand(
        navigation_constant=3.0,
        closing_speed=100.0,
        los_rate_gamma=0.1,
        los_rate_psi=-0.2,
    ).to_missile_command(None)
    assert command.nn == pytest.approx(3.0 * 100.0 * 0.1 / STANDARD_GRAVITY)
    assert command.ns == pytest.approx(3.0 * 100.0 * -0.2 / STANDARD_GRAVITY)


def test_regional_spawn_staggered_launch_and_detection_gated_blue_action(tmp_path):
    scenario = tmp_path / "regional.yaml"
    scenario.write_text(yaml.safe_dump({
        "mode": "1vN",
        "blue_altitude_m": [10000.0, 10000.0],
        "blue_heading_deg": 90.0,
        "missile_count": 3,
        "missile_spawn_distance_m": [20000.0, 22000.0],
        "missile_spawn_bearing_deg": [-20.0, 20.0],
        "missile_spawn_altitude_m": [9000.0, 11000.0],
        "missile_first_launch_time_s": 1.0,
        "missile_launch_interval_s": 2.0,
        "blue_detection_range_m": 1000.0,
        "physics_dt": 0.01,
        "policy_dt": 0.1,
    }), encoding="utf-8")
    env, _ = build_blue_escape_env(
        scenario, "configs/actions/blue_29.yaml", "zdj", 7
    )
    assert env.world.blue.kinematics.angles.psi == pytest.approx(math.pi / 2)
    assert [missile.launch_time_s for missile in env.world.missiles] == [1.0, 3.0, 5.0]
    for missile in env.world.missiles:
        position = missile.kinematics.position
        distance = math.hypot(position.x, position.z)
        bearing = math.degrees(math.atan2(position.z, position.x))
        assert 20000.0 <= distance <= 22000.0
        assert -20.0 <= bearing <= 20.0
        assert 9000.0 <= position.y <= 11000.0

    initial_missile_position = env.world.missiles[0].kinematics.position
    result = env.step(7)
    assert result.info["threat_detected"] is False
    assert result.info["requested_action_id"] == 7
    assert result.info["executed_action_id"] == 0
    assert env.world.missiles[0].kinematics.position == initial_missile_position


def test_blue_executes_requested_action_after_launched_threat_is_detected(tmp_path):
    scenario = tmp_path / "detected.yaml"
    scenario.write_text(yaml.safe_dump({
        "mode": "fixed_1v1",
        "missile_spawn_distance_m": [500.0, 500.0],
        "missile_spawn_bearing_deg": [0.0, 0.0],
        "blue_detection_range_m": 1000.0,
    }), encoding="utf-8")
    env, _ = build_blue_escape_env(
        scenario, "configs/actions/blue_29.yaml", "zdj", 0
    )
    result = env.step(7)
    assert result.info["threat_detected"] is True
    assert result.info["executed_action_id"] == 7
