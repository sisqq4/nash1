from __future__ import annotations
from dataclasses import dataclass
from air_combat_rl.domain.states import AircraftState, MissileState

@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    time_s: float
    blue: AircraftState
    missiles: tuple[MissileState, ...]
