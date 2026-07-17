"""Physical maneuver commands accepted by the simulation kernel."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class ManeuverCommand:
    nx: float
    nf: float
    gamma_s: float

    @property
    def ny(self) -> float:
        return self.nf * math.cos(self.gamma_s)

    @property
    def nz(self) -> float:
        return self.nf * math.sin(self.gamma_s)
