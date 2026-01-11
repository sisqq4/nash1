
"""Simple experience replay buffer for DQN-style agents."""

from __future__ import annotations

from typing import Tuple
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)

        self.obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.capacity,), dtype=np.int64)
        self.rew_buf = np.zeros((self.capacity,), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity,), dtype=np.float32)

        self.size = 0
        self.ptr = 0

    def store(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.next_obs_buf[idx] = next_obs
        self.act_buf[idx] = int(act)
        self.rew_buf[idx] = float(rew)
        self.done_buf[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        assert self.size > 0, "Buffer is empty"
        batch_size = min(batch_size, self.size)
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs_buf[idxs],
            self.act_buf[idxs],
            self.rew_buf[idxs],
            self.next_obs_buf[idxs],
            self.done_buf[idxs],
        )

    def get_state(self) -> dict:
        return {
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "obs_buf": self.obs_buf.copy(),
            "next_obs_buf": self.next_obs_buf.copy(),
            "act_buf": self.act_buf.copy(),
            "rew_buf": self.rew_buf.copy(),
            "done_buf": self.done_buf.copy(),
            "size": self.size,
            "ptr": self.ptr,
        }

    def load_state(self, state: dict) -> None:
        self.capacity = int(state.get("capacity", self.capacity))
        self.obs_dim = int(state.get("obs_dim", self.obs_dim))
        self.obs_buf = np.array(state["obs_buf"], dtype=np.float32)
        self.next_obs_buf = np.array(state["next_obs_buf"], dtype=np.float32)
        self.act_buf = np.array(state["act_buf"], dtype=np.int64)
        self.rew_buf = np.array(state["rew_buf"], dtype=np.float32)
        self.done_buf = np.array(state["done_buf"], dtype=np.float32)
        self.size = int(state.get("size", 0))
        self.ptr = int(state.get("ptr", 0))