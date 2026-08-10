from __future__ import annotations
from src.air_combat_rl.interfaces.policy import PolicyDecision
from src.air_combat_rl.algorithms.ppo.actor_critic import PPOActorCritic
from src.air_combat_rl.tasks.blue_escape.continuous.projector import ContinuousCommandProjector
from src.air_combat_rl.tasks.blue_escape.continuous.mapper import NearestManeuverMapper
class ProjectedPPOPolicy:
    def __init__(self, actor_critic: PPOActorCritic, env, projector=None, mapper=None):
        self.algorithm_name="ppo_projected"; self.actor_critic=actor_critic; self.env=env; self.projector=projector or ContinuousCommandProjector(); self.mapper=mapper or NearestManeuverMapper(env.actions)
    def act(self, observation, deterministic: bool=False):
        s=self.actor_critic.act(observation, deterministic); bounded=self.projector.scale_from_unit(s.squashed_action); state=self.env.world.blue.kinematics
        projected=self.projector.project(bounded,state,self.env.platform); mask=self.env.actions.action_mask(self.env.platform); valid=[i for i,m in enumerate(mask) if m]
        mapped=self.mapper.map(state,projected,self.env.platform,valid)
        return PolicyDecision(self.algorithm_name, s.raw_action, bounded, projected, mapped.action_id, mapped.command, mapped.distance, s.value, s.log_prob)
