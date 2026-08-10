"""Projected continuous-action wrapper over the discrete BlueEscapeEnv."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from air_combat_rl.tasks.blue_escape.continuous.projector import ContinuousCommandProjector, ContinuousProjectionConfig
from air_combat_rl.tasks.blue_escape.continuous.mapper import NearestManeuverMapper

@dataclass(frozen=True, slots=True)
class BoxSpace:
    low: np.ndarray; high: np.ndarray; shape: tuple[int, ...]

class ProjectedContinuousActionWrapper:
    def __init__(self, base_env, projector: ContinuousCommandProjector | None = None, mapper: NearestManeuverMapper | None = None) -> None:
        if projector is None and getattr(base_env, "platform_config", None) is not None:
            p = base_env.platform_config
            projector = ContinuousCommandProjector(config=ContinuousProjectionConfig(
                zdj_max_g=p.max_g if p.name == "zdj" else 9.0,
                yjj_max_g=p.max_g if p.name == "yjj" else 3.0,
                min_speed_mps=p.min_speed, low_speed_mps=p.low_speed,
                high_speed_mps=p.high_speed, max_speed_mps=p.max_speed,
                min_altitude_m=p.min_altitude, low_altitude_m=p.low_altitude,
            ))
        self.base_env = base_env; self.projector = projector or ContinuousCommandProjector(); self.mapper = mapper or NearestManeuverMapper(base_env.actions)
        self.action_space = BoxSpace(np.full(3, -1.0), np.full(3, 1.0), (3,)); self.observation_space = getattr(base_env, "observation_space", None)
    def reset(self, seed: int | None = None): return self.base_env.reset(seed)
    def step(self, action):
        raw = np.asarray(action, dtype=float); bounded = self.projector.scale_from_unit(np.clip(raw, -1.0, 1.0))
        state = self.base_env.world.blue.kinematics; projected = self.projector.project(bounded, state, self.base_env.platform)
        mask = self.base_env.actions.action_mask(self.base_env.platform); valid = [i for i,m in enumerate(mask) if m]
        mapped = self.mapper.map(state, projected, self.base_env.platform, valid)
        result = self.base_env.step(mapped.action_id)
        executed_action_id = int(result.info.get("executed_action_id", mapped.action_id))
        executed_action = self.base_env.actions.action(executed_action_id)
        result.info.update({"raw_continuous_action": raw, "bounded_continuous_action": raw.clip(-1.0, 1.0), "continuous_command": bounded, "projected_continuous_action": projected, "requested_projected_action_id": mapped.action_id, "executed_action_id": executed_action_id, "executed_action_name": executed_action.name, "executed_command": executed_action.command, "projection_distance": mapped.distance, "valid_action_count": mapped.valid_action_count, "fallback_used": mapped.fallback_used or executed_action_id != mapped.action_id})
        return result
