"""Projected continuous-action wrapper over the discrete BlueEscapeEnv."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from air_combat_rl.tasks.blue_escape.continuous.projector import ContinuousCommandProjector
from air_combat_rl.tasks.blue_escape.continuous.mapper import NearestManeuverMapper

@dataclass(frozen=True, slots=True)
class BoxSpace:
    low: np.ndarray; high: np.ndarray; shape: tuple[int, ...]

class ProjectedContinuousActionWrapper:
    def __init__(self, base_env, projector: ContinuousCommandProjector | None = None, mapper: NearestManeuverMapper | None = None) -> None:
        self.base_env = base_env; self.projector = projector or ContinuousCommandProjector(); self.mapper = mapper or NearestManeuverMapper(base_env.actions)
        self.action_space = BoxSpace(np.full(3, -1.0), np.full(3, 1.0), (3,)); self.observation_space = getattr(base_env, "observation_space", None)
    def reset(self, seed: int | None = None): return self.base_env.reset(seed)
    def step(self, action):
        raw = np.asarray(action, dtype=float); bounded = self.projector.scale_from_unit(np.clip(raw, -1.0, 1.0))
        state = self.base_env.world.blue.kinematics; projected = self.projector.project(bounded, state, self.base_env.platform)
        mask = self.base_env.actions.action_mask(self.base_env.platform); valid = [i for i,m in enumerate(mask) if m]
        mapped = self.mapper.map(state, projected, self.base_env.platform, valid)
        result = self.base_env.step(mapped.action_id)
        result.info.update({"raw_continuous_action": raw, "bounded_continuous_action": bounded, "projected_continuous_action": projected, "executed_action_id": mapped.action_id, "executed_command": mapped.command, "projection_distance": mapped.distance, "valid_action_count": mapped.valid_action_count, "fallback_used": mapped.fallback_used})
        return result
