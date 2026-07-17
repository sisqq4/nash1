from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class SimulationEvent:
    time_s: float
    kind: str
    entity_id: str
    details: dict[str, object]
