import math
from pathlib import Path

import pytest
import yaml

from air_combat_rl.domain.commands import ManeuverCommand
from air_combat_rl.domain.dynamics.missile_3dof import ProportionalNavigationCommand
from air_combat_rl.core.units import STANDARD_GRAVITY
from air_combat_rl.domain.platform_config import PlatformConfig
from air_combat_rl.simulation.scenarios.factory import ScenarioConfig, build_scenario
from air_combat_rl.tasks.blue_escape.rewards.components import RewardConfig
from air_combat_rl.runtime import build_blue_escape_env
from air_combat_rl.tasks.blue_escape.continuous.wrapper import ProjectedContinuousActionWrapper


def test_platform_and_reward_yaml_parameters_are_loaded():
    platform = PlatformConfig.from_yaml("configs/platform/zdj.yaml")
    reward = RewardConfig.from_yaml("configs/reward/escape_1v1.yaml")
    assert platform.name == "zdj"
    assert platform.max_g == 9.0
    assert platform.max_speed == 420.0
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
    assert runtime.platform_config.max_speed == 420.0
    assert env.reward_model.config.terminal_hit == -10.0
    assert wrapped.projector.config.max_speed_mps == 420.0


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
