"""Algorithm selection and trainer construction helpers."""
from __future__ import annotations
from dataclasses import dataclass
from air_combat_rl.algorithms.ppo.actor_critic import PPOActorCritic
from air_combat_rl.algorithms.ppo.policy import ProjectedPPOPolicy
from air_combat_rl.algorithms.ppo.trainer import PPOProjectedTrainer, PPOTrainerConfig
from air_combat_rl.algorithms.rainbow.network import RainbowQNetwork
from air_combat_rl.algorithms.rainbow.policy import RainbowDQNPolicyAdapter
from air_combat_rl.algorithms.rainbow.trainer import RainbowDQNTrainer, RainbowTrainerConfig
from air_combat_rl.tasks.blue_escape.continuous.wrapper import ProjectedContinuousActionWrapper
@dataclass(frozen=True, slots=True)
class AlgorithmRuntime:
    name: str; env: object; policy: object; trainer: object
def _cfg_obj(cls, data: dict):
    allowed = cls.__dataclass_fields__.keys(); return cls(**{k:v for k,v in data.items() if k in allowed})
def build_algorithm_runtime(config: dict, env):
    name=config.get("algorithm",{}).get("name", "rainbow_dqn"); seed=config.get("seed")
    obs,_=env.reset(seed); obs_dim=len(obs)
    if name=="rainbow_dqn":
        q=RainbowQNetwork(obs_dim, action_dim=env.action_space.n, seed=seed); trainer=RainbowDQNTrainer(env, q, config=_cfg_obj(RainbowTrainerConfig, config.get("rainbow",{})), seed=seed)
        return AlgorithmRuntime(name, env, RainbowDQNPolicyAdapter(env.actions, env.platform, q), trainer)
    if name=="ppo_projected":
        wrapped=ProjectedContinuousActionWrapper(env); ac=PPOActorCritic(obs_dim=obs_dim, action_dim=3, seed=seed); trainer=PPOProjectedTrainer(wrapped, ac, _cfg_obj(PPOTrainerConfig, config.get("ppo",{})))
        return AlgorithmRuntime(name, wrapped, ProjectedPPOPolicy(ac, env), trainer)
    raise ValueError(f"unsupported algorithm.name: {name}")
