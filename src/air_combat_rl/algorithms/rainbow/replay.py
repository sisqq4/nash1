"""Independent replay buffers for Rainbow DQN compatibility."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True, slots=True)
class Transition:
    observation: object; action: int; reward: float; next_observation: object; terminated: bool; truncated: bool

@dataclass(slots=True)
class ReplayBuffer:
    capacity: int = 100_000
    storage: list[Transition] = field(default_factory=list)
    position: int = 0
    def add(self, transition: Transition) -> None:
        if len(self.storage) < self.capacity: self.storage.append(transition)
        else: self.storage[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size: int, rng: np.random.Generator | None = None):
        if not self.storage: raise ValueError("cannot sample from empty replay buffer")
        rng = rng or np.random.default_rng(); idx = rng.choice(len(self.storage), size=min(batch_size, len(self.storage)), replace=False)
        return [self.storage[int(i)] for i in idx], idx, np.ones(len(idx))
    def __len__(self): return len(self.storage)

@dataclass(slots=True)
class PrioritizedReplayBuffer(ReplayBuffer):
    alpha: float = 0.6
    beta: float = 0.4
    priorities: list[float] = field(default_factory=list)
    def add(self, transition: Transition, priority: float | None = None) -> None:
        max_p = max(self.priorities, default=1.0) if priority is None else float(priority)
        if len(self.storage) < self.capacity:
            self.storage.append(transition); self.priorities.append(max_p)
        else:
            self.storage[self.position] = transition; self.priorities[self.position] = max_p
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size: int, rng: np.random.Generator | None = None):
        if not self.storage: raise ValueError("cannot sample from empty replay buffer")
        rng = rng or np.random.default_rng(); p = np.asarray(self.priorities[:len(self.storage)], float) ** self.alpha; p = p / p.sum()
        n = min(batch_size, len(self.storage)); idx = rng.choice(len(self.storage), size=n, replace=False, p=p)
        weights = (len(self.storage) * p[idx]) ** (-self.beta); weights = weights / weights.max()
        return [self.storage[int(i)] for i in idx], idx, weights
    def update_priorities(self, indices, priorities) -> None:
        for i, p in zip(indices, priorities): self.priorities[int(i)] = max(float(p), 1e-6)
