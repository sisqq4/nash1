from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    total_reward: float
    terminal: float = 0.0
    threat_relief: float = 0.0
    distance_change: float = 0.0
    encirclement: float = 0.0
    altitude: float = 0.0
    speed: float = 0.0
    action_smoothness: float = 0.0

class RewardComposer:
    def combine(self, **components: float) -> RewardBreakdown:
        return RewardBreakdown(total_reward=sum(components.values()), **components)
