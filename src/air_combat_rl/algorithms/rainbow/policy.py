"""Compatibility adapter for existing Rainbow DQN discrete policies/checkpoints."""
from __future__ import annotations
import numpy as np
from src.air_combat_rl.interfaces.policy import PolicyDecision
from src.air_combat_rl.algorithms.rainbow.checkpoint import RainbowCheckpointError, load_rainbow_checkpoint
class RainbowDQNPolicyAdapter:
    """Wrap a discrete Q-network/legacy policy without changing Rainbow action semantics."""
    algorithm_name="rainbow_dqn"
    def __init__(self, actions, platform: str, q_network=None): self.actions=actions; self.platform=platform; self.q_network=q_network
    def act(self, observation, deterministic: bool=False):
        mask=np.asarray(self.actions.action_mask(self.platform), dtype=bool)
        if self.q_network is None: q=np.zeros(len(mask))
        elif hasattr(self.q_network,"predict_q"): q=np.asarray(self.q_network.predict_q(observation), dtype=float)
        else: q=np.asarray(self.q_network(observation), dtype=float)
        masked=np.where(mask, q, -np.inf); action_id=int(np.argmax(masked)) if np.any(mask) else 0
        cmd=self.actions.command_for(action_id,self.platform)
        return PolicyDecision(self.algorithm_name, q, None, None, action_id, cmd, 0.0, None, None)
