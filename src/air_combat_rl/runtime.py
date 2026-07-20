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
