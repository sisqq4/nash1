"""Shared application runtime builders for scripts."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import random

import numpy as np

from air_combat_rl.simulation.scenarios.factory import ScenarioConfig, build_scenario
from air_combat_rl.tasks.blue_escape.action_catalog import ActionCatalog
from air_combat_rl.tasks.blue_escape.environment import BlueEscapeEnv


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    scenario_path: str
    actions_path: str
    platform: str
    seed: int
    scenario: ScenarioConfig
    max_policy_steps: int | None


def build_blue_escape_env(
    scenario_path: str | Path,
    actions_path: str | Path,
    platform: str,
    seed: int,
    max_policy_steps: int | None = None,
) -> tuple[BlueEscapeEnv, RuntimeConfig]:
    """Build a seeded BlueEscapeEnv from scenario and action config files."""
    random.seed(seed)
    np.random.seed(seed)
    if platform not in {"zdj", "yjj"}:
        raise ValueError(f"unsupported platform {platform!r}; expected 'zdj' or 'yjj'")
    scenario = ScenarioConfig.from_yaml(str(scenario_path))
    scenario = replace(scenario, seed=seed)
    world = build_scenario(scenario)
    world.blue = replace(world.blue, platform=platform)
    actions = ActionCatalog.from_yaml(str(actions_path))
    if not any(actions.action_mask(platform)):
        raise ValueError(f"platform {platform!r} has no legal actions")
    if world.clock.physics_dt != scenario.physics_dt or world.clock.policy_dt != scenario.policy_dt:
        raise ValueError("world clock does not match scenario physics_dt/policy_dt")
    env = BlueEscapeEnv(world, actions, platform, max_time_s=scenario.max_episode_time_s, max_policy_steps=max_policy_steps)
    return env, RuntimeConfig(str(scenario_path), str(actions_path), platform, seed, scenario, max_policy_steps)
