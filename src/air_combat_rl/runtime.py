"""Shared application runtime builders for scripts."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import random

import numpy as np

from air_combat_rl.simulation.scenarios.factory import ScenarioConfig, build_scenario
from air_combat_rl.domain.platform_config import PlatformConfig
from air_combat_rl.tasks.blue_escape.action_catalog import ActionCatalog
from air_combat_rl.tasks.blue_escape.environment import BlueEscapeEnv
from air_combat_rl.tasks.blue_escape.rewards.components import RewardConfig


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _bundled_config(relative_path: str) -> Path:
    """Resolve bundled configs independently of the process working directory."""
    return _REPOSITORY_ROOT / "configs" / relative_path


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    scenario_path: str
    actions_path: str
    platform: str
    seed: int
    scenario: ScenarioConfig
    max_policy_steps: int | None
    platform_config_path: str
    reward_config_path: str
    platform_config: PlatformConfig
    reward_config: RewardConfig


def build_blue_escape_env(
    scenario_path: str | Path,
    actions_path: str | Path,
    platform: str,
    seed: int,
    max_policy_steps: int | None = None,
    platform_config_path: str | Path | None = None,
    reward_config_path: str | Path | None = None,
) -> tuple[BlueEscapeEnv, RuntimeConfig]:
    """Build a seeded BlueEscapeEnv from scenario and action config files."""
    random.seed(seed)
    np.random.seed(seed)
    if platform not in {"zdj", "yjj"}:
        raise ValueError(f"unsupported platform {platform!r}; expected 'zdj' or 'yjj'")
    scenario = ScenarioConfig.from_yaml(str(scenario_path))
    scenario = replace(scenario, seed=seed)
    platform_path = Path(platform_config_path) if platform_config_path else _bundled_config(f"platform/{platform}.yaml")
    platform_config = PlatformConfig.from_yaml(platform_path)
    if platform_config.name != platform:
        raise ValueError("platform config name does not match --platform")
    def world_factory(world_seed: int):
        configured = replace(scenario, seed=int(world_seed))
        built = build_scenario(configured)
        built.blue = replace(built.blue, platform=platform)
        built.config = replace(
            built.config,
            blue_min_speed_mps=platform_config.min_speed,
            blue_max_speed_mps=platform_config.max_speed,
        )
        return built
    world = world_factory(seed)
    reward_path = Path(reward_config_path) if reward_config_path else _bundled_config(
        f"reward/escape_{'1vn' if len(world.missiles) > 1 else '1v1'}.yaml"
    )
    reward_config = RewardConfig.from_yaml(reward_path)
    actions = ActionCatalog.from_yaml(str(actions_path))
    if not any(actions.action_mask(platform)):
        raise ValueError(f"platform {platform!r} has no legal actions")
    if world.clock.physics_dt != scenario.physics_dt or world.clock.policy_dt != scenario.policy_dt:
        raise ValueError("world clock does not match scenario physics_dt/policy_dt")
    env = BlueEscapeEnv(world, actions, platform, max_time_s=scenario.max_episode_time_s, max_policy_steps=max_policy_steps, world_factory=world_factory, initial_seed=seed, reward_config=reward_config, platform_config=platform_config)
    return env, RuntimeConfig(str(scenario_path), str(actions_path), platform, seed, scenario, max_policy_steps, str(platform_path), str(reward_path), platform_config, reward_config)

from air_combat_rl.algorithms.ppo.actor_critic import PPOActorCritic
from air_combat_rl.algorithms.ppo.policy import ProjectedPPOPolicy
from air_combat_rl.algorithms.ppo.trainer import PPOProjectedTrainer, PPOTrainerConfig
from air_combat_rl.algorithms.ppo.discrete import PPODiscreteActorCritic, PPODiscreteTrainer, PPODiscreteTrainerConfig, DiscretePPOPolicy
from air_combat_rl.algorithms.rainbow.network import RainbowQNetwork
from air_combat_rl.algorithms.rainbow.policy import RainbowDQNPolicyAdapter
from air_combat_rl.algorithms.rainbow.trainer import RainbowDQNTrainer, RainbowTrainerConfig
from air_combat_rl.tasks.blue_escape.continuous.wrapper import ProjectedContinuousActionWrapper

@dataclass(frozen=True, slots=True)
class AlgorithmRuntime:
    name: str
    env: object
    policy: object
    trainer: object

def _cfg_obj(cls, data: dict):
    allowed = cls.__dataclass_fields__.keys()
    return cls(**{k: v for k, v in (data or {}).items() if k in allowed})

def build_algorithm_runtime(config: dict, env):
    name = config.get("algorithm", {}).get("name", config.get("name", "ppo_projected"))
    seed = config.get("seed")
    obs, _ = env.reset(seed)
    obs_dim = len(obs)
    if name == "rainbow_dqn":
        q = RainbowQNetwork(obs_dim, action_dim=env.action_space.n, seed=seed)
        trainer = RainbowDQNTrainer(env, q, config=_cfg_obj(RainbowTrainerConfig, config.get("rainbow", {})), seed=seed)
        return AlgorithmRuntime(name, env, RainbowDQNPolicyAdapter(env.actions, env.platform, q), trainer)
    if name == "ppo_projected":
        wrapped = ProjectedContinuousActionWrapper(env)
        ac = PPOActorCritic(obs_dim=obs_dim, action_dim=3, seed=seed)
        trainer = PPOProjectedTrainer(wrapped, ac, _cfg_obj(PPOTrainerConfig, config.get("ppo", {})))
        return AlgorithmRuntime(name, wrapped, ProjectedPPOPolicy(ac, env), trainer)
    if name == "ppo_discrete":
        ac = PPODiscreteActorCritic(obs_dim=obs_dim, action_dim=env.action_space.n, seed=seed)
        trainer = PPODiscreteTrainer(env, ac, _cfg_obj(PPODiscreteTrainerConfig, config.get("ppo", {})))
        return AlgorithmRuntime(name, env, DiscretePPOPolicy(ac, env), trainer)
    raise ValueError(f"unsupported algorithm.name: {name}")
