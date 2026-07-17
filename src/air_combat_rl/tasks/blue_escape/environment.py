"""Gymnasium-style blue escape task wrapper around SimulationWorld."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from air_combat_rl.domain.outcomes import Outcome
from air_combat_rl.simulation.world import SimulationWorld
from air_combat_rl.tasks.blue_escape.action_catalog import ActionCatalog

@dataclass(frozen=True, slots=True)
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]

class BlueEscapeEnv:
    def __init__(self, world: SimulationWorld, actions: ActionCatalog, platform: str, max_time_s: float = 60.0) -> None:
        self.world = world
        self.actions = actions
        self.platform = platform
        self.max_time_s = max_time_s
        self.last_outcome = Outcome.RUNNING

    def step(self, action_id: int) -> StepResult:
        command = self.actions.command_for(action_id, self.platform)
        snapshot, events = self.world.step_policy_interval(command)
        outcome = Outcome.RUNNING
        if any(event.kind == "ground_collision" for event in events):
            outcome = Outcome.CRASH
        elif snapshot.time_s >= self.max_time_s:
            outcome = Outcome.TIMEOUT
        self.last_outcome = outcome
        terminated = outcome in {Outcome.CRASH, Outcome.HIT, Outcome.SUCCESS, Outcome.EXHAUSTED}
        truncated = outcome == Outcome.TIMEOUT
        reward = 0.0
        return StepResult(self._observe(snapshot), reward, terminated, truncated, {"outcome": outcome, "events": events})

    def _observe(self, snapshot) -> np.ndarray:
        blue = snapshot.blue.kinematics
        return np.array([blue.position.x, blue.position.z, blue.position.y, blue.speed, blue.angles.gamma, blue.angles.psi], dtype=np.float32)
