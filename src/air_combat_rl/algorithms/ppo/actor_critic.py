"""NumPy MLP actor-critic for projected continuous PPO."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

LOG_2PI = float(np.log(2.0 * np.pi))

@dataclass(slots=True)
class GaussianSample:
    raw_action: np.ndarray
    squashed_action: np.ndarray
    log_prob: float
    entropy: float
    value: float
    mean: np.ndarray
    std: np.ndarray

class PPOActorCritic:
    """Feed-forward actor-critic: 3-D Gaussian actor and scalar critic."""
    def __init__(self, obs_dim: int, action_dim: int = 3, hidden_sizes: tuple[int, ...] = (64, 64), seed: int | None = None) -> None:
        if action_dim != 3:
            raise ValueError("projected PPO action_dim must be 3 for [nx,nf,gamma_s]")
        self.obs_dim = obs_dim; self.action_dim = action_dim; self.rng = np.random.default_rng(seed)
        dims = (obs_dim, *hidden_sizes); self.actor_w=[]; self.actor_b=[]; self.critic_w=[]; self.critic_b=[]
        for a, b in zip(dims[:-1], dims[1:]):
            self.actor_w.append(self.rng.normal(0, np.sqrt(2 / a), (a, b))); self.actor_b.append(np.zeros(b))
            self.critic_w.append(self.rng.normal(0, np.sqrt(2 / a), (a, b))); self.critic_b.append(np.zeros(b))
        last = dims[-1]
        self.mean_w = self.rng.normal(0, 0.01, (last, action_dim)); self.mean_b = np.zeros(action_dim); self.log_std = np.zeros(action_dim)
        self.value_w = self.rng.normal(0, 0.01, (last, 1)); self.value_b = np.zeros(1)
    def _mlp(self, x, weights, biases):
        h = np.asarray(x, dtype=float)
        if h.ndim == 1: h = h.reshape(1, -1)
        for w, b in zip(weights, biases): h = np.tanh(h @ w + b)
        return h
    def forward(self, observation):
        single = np.asarray(observation).ndim == 1
        h = self._mlp(observation, self.actor_w, self.actor_b); mean = h @ self.mean_w + self.mean_b; log_std = np.broadcast_to(np.clip(self.log_std, -20, 2), mean.shape)
        v = (self._mlp(observation, self.critic_w, self.critic_b) @ self.value_w + self.value_b).reshape(-1)
        if single: return mean[0], log_std[0], float(v[0])
        return mean, log_std, v
    def log_prob_from_raw(self, raw, mean, log_std):
        raw = np.asarray(raw, float); mean = np.asarray(mean, float); log_std = np.asarray(log_std, float); std = np.exp(log_std)
        gaussian = -0.5 * (((raw - mean) / std) ** 2 + 2 * log_std + LOG_2PI)
        squashed = np.tanh(raw); correction = np.log(np.clip(1.0 - squashed ** 2, 1e-6, None))
        return np.sum(gaussian - correction, axis=-1)
    def entropy_from_log_std(self, log_std):
        return np.sum(log_std + 0.5 * (1.0 + LOG_2PI), axis=-1)
    def evaluate_actions(self, observations, raw_actions):
        mean, log_std, values = self.forward(observations)
        return self.log_prob_from_raw(raw_actions, mean, log_std), self.entropy_from_log_std(log_std), values
    def act(self, observation, deterministic: bool = False) -> GaussianSample:
        mean, log_std, value = self.forward(observation); std = np.exp(log_std)
        raw = mean if deterministic else mean + self.rng.normal(size=3) * std
        squashed = np.tanh(raw); lp = float(self.log_prob_from_raw(raw, mean, log_std)); entropy = float(self.entropy_from_log_std(log_std))
        return GaussianSample(raw, squashed, lp, entropy, value, mean, std)
    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "rng"} | {"rng_state": self.rng.bit_generator.state}
    def trainable_vectors(self):
        return [self.mean_b, self.log_std, self.value_b]
