"""Gymnasium-style blue escape task wrapper around SimulationWorld."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np
from air_combat_rl.domain.outcomes import Outcome
from air_combat_rl.simulation.world import SimulationWorld
from air_combat_rl.tasks.blue_escape.action_catalog import ActionCatalog, SAFE_FALLBACK_ACTION_ID
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

class DiscreteSpace:
    def __init__(self, n: int) -> None:
        self.n = n

class BlueEscapeEnv:
    def __init__(self, world: SimulationWorld, actions: ActionCatalog, platform: str, max_time_s: float = 60.0, max_policy_steps: int | None = None, m_max: int = 4, world_factory=None, initial_seed: int = 0, reward_config: RewardConfig | None = None, platform_config=None) -> None:
        self.world = world; self._initial_world = copy.deepcopy(world); self.actions = actions; self.platform = platform; self.action_space = DiscreteSpace(29)
        self.max_time_s = max_time_s; self.max_policy_steps = max_policy_steps
        self.last_outcome = Outcome.RUNNING; self.held_action = HeldAction(); self.policy_steps = 0
        self.observations = ObservationBuilder(ObservationConfig(m_max=m_max)); self.reward_model = EscapeReward(reward_config or RewardConfig())
        self.platform_config = platform_config
        self.reward_model.reset(self.world.snapshot())
        self._world_factory = world_factory; self._seed = int(initial_seed)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
        if seed is not None: self._seed = int(seed)
        self.world = self._world_factory(self._seed) if self._world_factory is not None else copy.deepcopy(self._initial_world)
        self.held_action = HeldAction(); self.policy_steps = 0; self.last_outcome = Outcome.RUNNING; self.reward_model.reset(self.world.snapshot())
        obs, mask = self.observations.build(self.world.blue, self.world.snapshot().missiles, 0)
        return obs, {"missile_mask": mask, "action_mask": np.asarray(self.actions.action_mask(self.platform), dtype=bool), "initial_missile_count": len(self.world.missiles)}

    def step(self, action_id: int) -> StepResult:
        threat_detected = self.world.blue_detects_threat()
        requested_action_id = int(action_id)
        if not threat_detected:
            action_id = SAFE_FALLBACK_ACTION_ID
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
        distances = list(getattr(self.world, "min_missile_distances", []))
        info = {"outcome": outcome, "events": events, "reward_components": comps, "missile_mask": mask, "action_mask": np.asarray(self.actions.action_mask(self.platform), dtype=bool), "substeps": self.world.substeps_last_interval, "time_s": snapshot.time_s, "altitude_y_m": snapshot.blue.kinematics.position.y, "min_sampled_distance_m": min(distances) if distances else None, "alive_missile_count": sum(1 for missile in snapshot.missiles if missile.alive), "locked_missile_count": sum(1 for missile in snapshot.missiles if missile.locked), "threat_detected": threat_detected, "requested_action_id": requested_action_id, "executed_action_id": action_id}
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
