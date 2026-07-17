from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(slots=True)
class RolloutBuffer:
    observations: list[object] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def clear(self) -> None:
        self.observations.clear(); self.actions.clear(); self.log_probs.clear(); self.values.clear(); self.rewards.clear(); self.dones.clear()
