"""Independent Rainbow DQN trainer path preserving discrete replay semantics."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.air_combat_rl.algorithms.rainbow.network import RainbowQNetwork
from src.air_combat_rl.algorithms.rainbow.replay import PrioritizedReplayBuffer, Transition
from src.air_combat_rl.algorithms.rainbow.policy import RainbowDQNPolicyAdapter

@dataclass(frozen=True, slots=True)
class RainbowTrainerConfig:
    batch_size: int = 32
    gamma: float = 0.99
    learning_rate: float = 1e-3
    target_update_interval: int = 100
    replay_capacity: int = 100_000

@dataclass(slots=True)
class RainbowTrainingStats:
    global_step: int; replay_size: int; loss: float

class RainbowDQNTrainer:
    def __init__(self, env, q_network: RainbowQNetwork, target_network: RainbowQNetwork | None = None, replay_buffer: PrioritizedReplayBuffer | None = None, config: RainbowTrainerConfig | None = None, seed: int | None = None) -> None:
        self.env = env; self.q_network = q_network; self.target_network = target_network or RainbowQNetwork(q_network.obs_dim, q_network.action_dim, seed=seed); self.target_network.copy_from(q_network)
        self.replay_buffer = replay_buffer or PrioritizedReplayBuffer(capacity=(config or RainbowTrainerConfig()).replay_capacity); self.config = config or RainbowTrainerConfig(); self.rng = np.random.default_rng(seed); self.global_step = 0; self.policy = RainbowDQNPolicyAdapter(env.actions, env.platform, q_network)
    def collect_step(self, observation):
        decision = self.policy.act(observation); result = self.env.step(decision.executed_action_id)
        self.replay_buffer.add(Transition(observation, decision.executed_action_id, result.reward, result.observation, result.terminated, result.truncated))
        self.global_step += 1
        return result
    def train_one_update(self) -> RainbowTrainingStats:
        if len(self.replay_buffer) == 0: return RainbowTrainingStats(self.global_step, 0, 0.0)
        batch, indices, weights = self.replay_buffer.sample(self.config.batch_size, self.rng); losses=[]; td_errors=[]
        for tr, w in zip(batch, weights):
            obs = np.asarray(tr.observation, float); nxt = np.asarray(tr.next_observation, float); q = self.q_network.predict_q(obs); target_q = self.target_network.predict_q(nxt)
            target = tr.reward + (0.0 if tr.terminated else self.config.gamma * float(np.max(target_q)))
            td = q[tr.action] - target; losses.append(float(w * td * td)); td_errors.append(abs(float(td)))
            self.q_network.weights[:, tr.action] -= self.config.learning_rate * w * td * obs
            self.q_network.bias[tr.action] -= self.config.learning_rate * w * td
        self.replay_buffer.update_priorities(indices, np.asarray(td_errors) + 1e-6)
        if self.global_step % self.config.target_update_interval == 0: self.target_network.copy_from(self.q_network)
        return RainbowTrainingStats(self.global_step, len(self.replay_buffer), float(np.mean(losses)))
