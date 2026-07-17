"""Scene-agnostic policy interface for discrete PPO implementations."""
from __future__ import annotations
from typing import Protocol
import numpy as np

class DiscretePolicy(Protocol):
    def act(self, observation: np.ndarray, action_mask: np.ndarray | None = None) -> int:
        """Return a discrete action without depending on scenario internals."""
