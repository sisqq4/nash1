"""Scene-agnostic policy interfaces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Any


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Uniform action decision record for discrete DQN and projected PPO."""

    algorithm_name: str
    raw_action: Any
    bounded_action: Any | None
    projected_action: Any | None
    executed_action_id: int
    executed_command: Any
    projection_distance: float
    value: float | None = None
    log_prob: float | None = None


class DiscreteActionPolicy(Protocol):
    """Policy interface shared by learning policies and rule-machine baselines."""

    def reset(self, seed: int | None = None) -> None:
        """Reset any policy-internal recurrent or dwell state."""

    def act(self, observation: Sequence[float], action_mask: Sequence[bool] | None = None) -> int:
        """Return a discrete action id from the environment observation and mask."""


class DecisionPolicy(Protocol):
    """Uniform policy adapter returning a full PolicyDecision."""

    def act(self, observation: Sequence[float], deterministic: bool = False) -> PolicyDecision:
        """Return a uniform decision record."""
