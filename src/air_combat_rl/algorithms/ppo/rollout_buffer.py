from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass(slots=True)
class RolloutBuffer:
    observations: list[object]=field(default_factory=list); actions: list[object]=field(default_factory=list); rewards: list[float]=field(default_factory=list)
    values: list[float]=field(default_factory=list); log_probs: list[float]=field(default_factory=list); terminated: list[bool]=field(default_factory=list); truncated: list[bool]=field(default_factory=list); episode_starts: list[bool]=field(default_factory=list)
    projected_actions: list[object]=field(default_factory=list); executed_action_ids: list[int]=field(default_factory=list); projection_distances: list[float]=field(default_factory=list)
    advantages: np.ndarray | None=None; returns: np.ndarray | None=None
    def add(self, observation, action, reward, value, log_prob, terminated, truncated, episode_start, projected_action=None, executed_action_id=-1, projection_distance=0.0):
        self.observations.append(observation); self.actions.append(action); self.rewards.append(float(reward)); self.values.append(float(value)); self.log_probs.append(float(log_prob)); self.terminated.append(bool(terminated)); self.truncated.append(bool(truncated)); self.episode_starts.append(bool(episode_start)); self.projected_actions.append(projected_action); self.executed_action_ids.append(int(executed_action_id)); self.projection_distances.append(float(projection_distance))
    def compute_returns_and_advantages(self, last_value: float, gamma: float=0.99, gae_lambda: float=0.95):
        n=len(self.rewards); adv=np.zeros(n); last_gae=0.0
        for t in reversed(range(n)):
            nonterminal=0.0 if self.terminated[t] else 1.0; next_value=last_value if t==n-1 else self.values[t+1]
            delta=self.rewards[t]+gamma*next_value*nonterminal-self.values[t]; last_gae=delta+gamma*gae_lambda*nonterminal*last_gae; adv[t]=last_gae
        self.advantages=(adv-adv.mean())/(adv.std()+1e-8) if n else adv; self.returns=adv+np.asarray(self.values); return self.advantages,self.returns
    def clear(self):
        for name in ("observations","actions","rewards","values","log_probs","terminated","truncated","episode_starts","projected_actions","executed_action_ids","projection_distances"): getattr(self,name).clear()
        self.advantages=None; self.returns=None
