"""Simulation clock and time-scale contracts."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationClock:
    physics_dt: float = 0.005
    policy_dt: float = 0.1

    def __post_init__(self) -> None:
        if self.physics_dt <= 0 or self.policy_dt <= 0:
            raise ValueError("physics_dt and policy_dt must be positive")
        ratio = self.policy_dt / self.physics_dt
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("policy_dt must be an integer multiple of physics_dt")

    @property
    def substeps_per_policy_step(self) -> int:
        return int(round(self.policy_dt / self.physics_dt))
