"""Fixed-size blue escape observations with padded missile slots."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from air_combat_rl.core.coordinates import VecXZY, normalize_angle_rad
from air_combat_rl.core.math3d import flight_velocity, global_to_local
from air_combat_rl.domain.states import AircraftState, MissileState


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    m_max: int = 4
    position_scale_m: float = 100_000.0
    distance_scale_m: float = 100_000.0
    speed_scale_mps: float = 2_000.0


class ObservationBuilder:
    def __init__(self, config: ObservationConfig = ObservationConfig()) -> None:
        self.config = config

    @property
    def size(self) -> int:
        return 8 + self.config.m_max * 11

    def build(self, blue: AircraftState, missiles: tuple[MissileState, ...] | list[MissileState], current_action: int) -> tuple[np.ndarray, np.ndarray]:
        b = blue.kinematics
        obs = [
            b.position.x / self.config.position_scale_m,
            b.position.z / self.config.position_scale_m,
            b.position.y / self.config.position_scale_m,
            b.position.y / self.config.position_scale_m,
            b.speed / self.config.speed_scale_mps,
            b.angles.gamma / math.pi,
            normalize_angle_rad(b.angles.psi) / math.pi,
            current_action / 28.0,
        ]
        active = [m for m in missiles if m.alive and m.locked]
        active.sort(key=lambda m: _distance(blue, m))
        mask = np.zeros(self.config.m_max, dtype=bool)
        for slot in range(self.config.m_max):
            if slot >= len(active):
                obs.extend([0.0] * 11)
                continue
            m = active[slot]
            mask[slot] = True
            rel = VecXZY(m.kinematics.position.x - b.position.x, m.kinematics.position.z - b.position.z, m.kinematics.position.y - b.position.y)
            rel_local = global_to_local(rel, b.angles)
            bv = flight_velocity(b); mv = flight_velocity(m.kinematics)
            rv = VecXZY(mv.x - bv.x, mv.z - bv.z, mv.y - bv.y)
            rv_local = global_to_local(rv, b.angles)
            d = max(rel.norm(), 1e-6)
            closing = -((rel.x * rv.x + rel.z * rv.z + rel.y * rv.y) / d)
            bearing = math.atan2(rel_local.z, rel_local.x)
            elevation = math.atan2(rel_local.y, math.hypot(rel_local.x, rel_local.z))
            threat = max(0.0, closing / self.config.speed_scale_mps) / max(d / self.config.distance_scale_m, 0.01)
            obs.extend([
                rel_local.x / self.config.distance_scale_m,
                rel_local.z / self.config.distance_scale_m,
                rel_local.y / self.config.distance_scale_m,
                rv_local.x / self.config.speed_scale_mps,
                rv_local.z / self.config.speed_scale_mps,
                rv_local.y / self.config.speed_scale_mps,
                d / self.config.distance_scale_m,
                closing / self.config.speed_scale_mps,
                bearing / math.pi,
                elevation / math.pi,
                min(threat, 10.0) / 10.0,
            ])
        return np.asarray(obs, dtype=np.float32), mask


def _distance(blue: AircraftState, missile: MissileState) -> float:
    p = blue.kinematics.position; q = missile.kinematics.position
    return math.sqrt((p.x-q.x)**2 + (p.z-q.z)**2 + (p.y-q.y)**2)
