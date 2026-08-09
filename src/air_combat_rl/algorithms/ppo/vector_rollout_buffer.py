"""Fixed-shape [time, environment] PPO rollout storage."""

from __future__ import annotations

import numpy as np


class VectorRolloutBuffer:
    def __init__(
        self, steps: int, num_envs: int, obs_dim: int, action_dim: int = 3
    ) -> None:
        if min(steps, num_envs, obs_dim, action_dim) <= 0:
            raise ValueError("rollout buffer dimensions must be positive")
        shape = (steps, num_envs)
        self.steps = steps
        self.num_envs = num_envs
        self.observations = np.zeros((*shape, obs_dim), np.float32)
        self.raw_actions = np.zeros((*shape, action_dim), np.float32)
        self.bounded_actions = np.zeros((*shape, action_dim), np.float32)
        self.log_probs = np.zeros(shape, np.float32)
        self.values = np.zeros(shape, np.float32)
        self.next_values = np.zeros(shape, np.float32)
        self.rewards = np.zeros(shape, np.float32)
        self.terminated = np.zeros(shape, bool)
        self.truncated = np.zeros(shape, bool)
        self.advantages = np.zeros(shape, np.float32)
        self.returns = np.zeros(shape, np.float32)
        self.executed_action_ids = np.zeros(shape, np.int64)
        self.projection_distances = np.zeros(shape, np.float32)

    def compute_gae(self, gamma=0.99, gae_lambda=0.95, bootstrap_truncated=True):
        """Compute per-environment GAE without leaking across episode boundaries."""
        last_gae = np.zeros(self.num_envs, np.float32)
        for time_index in reversed(range(self.steps)):
            terminated = self.terminated[time_index]
            truncated = self.truncated[time_index]
            bootstrap_mask = (~terminated).astype(np.float32)
            if not bootstrap_truncated:
                bootstrap_mask *= (~truncated).astype(np.float32)
            delta = (
                self.rewards[time_index]
                + gamma * self.next_values[time_index] * bootstrap_mask
                - self.values[time_index]
            )
            last_gae = delta + gamma * gae_lambda * bootstrap_mask * last_gae
            # A truncation may bootstrap its final observation value, but must never
            # propagate advantages from the reset episode that follows it.
            last_gae = np.where(terminated | truncated, delta, last_gae)
            self.advantages[time_index] = last_gae
        self.returns = self.advantages + self.values
        return self.advantages, self.returns
