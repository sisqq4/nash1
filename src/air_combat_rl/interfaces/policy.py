"""Scene-agnostic policy interfaces."""
from __future__ import annotations

from typing import Protocol, Sequence


class DiscreteActionPolicy(Protocol):
    """Policy interface shared by learning policies and rule-machine baselines."""

    def reset(self, seed: int | None = None) -> None:
        """Reset any policy-internal recurrent or dwell state."""

    def act(self, observation: Sequence[float], action_mask: Sequence[bool] | None = None) -> int:
        """Return a discrete action id from the environment observation and mask."""
