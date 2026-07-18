"""Gymnasium-style blue escape task wrapper around SimulationWorld."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from air_combat_rl.domain.outcomes import Outcome
from air_combat_rl.simulation.world import SimulationWorld
from air_combat_rl.tasks.blue_escape.action_catalog import ActionCatalog
from air_combat_rl.tasks.blue_escape.action_hold import HeldAction
from air_combat_rl.tasks.blue_escape.observation_builder import ObservationBuilder, ObservationConfig
from air_combat_rl.tasks.blue_escape.rewards.components import EscapeReward, RewardConfig

@dataclass(frozen=True, slots=True)
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]

class BlueEscapeEnv:
    def __init__(self, world: SimulationWorld, actions: ActionCatalog, platform: str, max_time_s: float = 60.0, max_policy_steps: int | None = None, m_max: int = 4) -> None:
        self.world = world; self.actions = actions; self.platform = platform
        self.max_time_s = max_time_s; self.max_policy_steps = max_policy_steps
        self.last_outcome = Outcome.RUNNING; self.held_action = HeldAction(); self.policy_steps = 0
        self.observations = ObservationBuilder(ObservationConfig(m_max=m_max)); self.reward_model = EscapeReward(RewardConfig())
        self.reward_model.reset(self.world.snapshot())

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
        self.policy_steps = 0; self.last_outcome = Outcome.RUNNING; self.reward_model.reset(self.world.snapshot())
        obs, mask = self.observations.build(self.world.blue, self.world.snapshot().missiles, 0)
        return obs, {"missile_mask": mask, "action_mask": np.asarray(self.actions.action_mask(self.platform), dtype=bool)}

    def step(self, action_id: int) -> StepResult:
        self.held_action.select(action_id, self.platform, self.actions, self.world.clock)
        snapshot, events = self.world.step_held_policy_interval(self.held_action)
        self.policy_steps += 1
        outcome = "running"
        if any(event.kind == "hit" for event in events): outcome = "hit"
        elif any(event.kind == "ground_collision" for event in events): outcome = "crash"
        elif snapshot.missiles and not any(m.alive and m.locked for m in snapshot.missiles): outcome = "exhausted"
        elif self.world.all_live_threats_safely_passed(): outcome = "success"
        elif snapshot.time_s >= self.max_time_s: outcome = "timeout"
        if self.max_policy_steps is not None and self.policy_steps >= self.max_policy_steps and outcome == "running": outcome = "timeout"
        self.last_outcome = _to_outcome(outcome)
        terminated = outcome in {"hit", "crash", "success", "exhausted"}
        truncated = outcome == "timeout"
        reward, comps = self.reward_model.compute(snapshot, outcome, action_id)
        obs, mask = self.observations.build(snapshot.blue, snapshot.missiles, action_id)
        info = {"outcome": outcome, "events": events, "reward_components": comps, "missile_mask": mask, "action_mask": np.asarray(self.actions.action_mask(self.platform), dtype=bool), "substeps": self.world.substeps_last_interval, "time_s": snapshot.time_s}
        return StepResult(obs, reward, terminated, truncated, info)


def _to_outcome(outcome: str) -> Outcome:
    if outcome == "running":
        return Outcome.RUNNING
    if outcome == "crash":
        return Outcome.CRASH
    if outcome == "hit":
        return Outcome.HIT
    if outcome == "success":
        return Outcome.SUCCESS
    if outcome == "exhausted":
        return Outcome.EXHAUSTED
    return Outcome.TIMEOUT
