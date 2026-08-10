"""Continuous command physical scaling and state/platform projection."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from src.air_combat_rl.domain.commands import ManeuverCommand
from src.air_combat_rl.domain.states import FlightState

@dataclass(frozen=True, slots=True)
class ContinuousActionBounds:
    nx_min: float = -3.0; nx_max: float = 9.0
    nf_min: float = 0.0; nf_max: float = 9.0
    gamma_s_min: float = -math.pi; gamma_s_max: float = math.pi

@dataclass(frozen=True, slots=True)
class ContinuousProjectionConfig:
    zdj_max_g: float = 9.0
    yjj_max_g: float = 3.0
    min_speed_mps: float = 120.0
    low_speed_mps: float = 180.0
    high_speed_mps: float = 650.0
    max_speed_mps: float = 750.0
    min_altitude_m: float = 300.0
    low_altitude_m: float = 1000.0
    max_dive_gamma_s_rad: float = math.radians(100.0)

class ContinuousCommandProjector:
    """Apply range, platform, speed, altitude and safety constraints outside PPO."""
    def __init__(self, bounds: ContinuousActionBounds | None = None, config: ContinuousProjectionConfig | None = None) -> None:
        self.bounds = bounds or ContinuousActionBounds(); self.config = config or ContinuousProjectionConfig()

    def scale_from_unit(self, action: np.ndarray | list[float] | tuple[float, ...]) -> ManeuverCommand:
        arr = np.asarray(action, dtype=float)
        if arr.shape != (3,): raise ValueError("continuous action must have shape (3,)")
        arr = np.clip(arr, -1.0, 1.0)
        b = self.bounds
        nx = b.nx_min + (arr[0] + 1.0) * 0.5 * (b.nx_max - b.nx_min)
        nf = b.nf_min + (arr[1] + 1.0) * 0.5 * (b.nf_max - b.nf_min)
        gs = b.gamma_s_min + (arr[2] + 1.0) * 0.5 * (b.gamma_s_max - b.gamma_s_min)
        return ManeuverCommand(float(nx), float(nf), float(gs))

    def project(self, command: ManeuverCommand, state: FlightState, platform: str) -> ManeuverCommand:
        b, c = self.bounds, self.config
        max_g = c.yjj_max_g if platform == "yjj" else c.zdj_max_g
        nx = min(max(command.nx, b.nx_min), min(b.nx_max, max_g))
        nf = min(max(command.nf, b.nf_min), min(b.nf_max, max_g))
        gamma_s = min(max(command.gamma_s, b.gamma_s_min), b.gamma_s_max)
        if state.speed <= c.min_speed_mps: nx = max(nx, 0.0); nf = min(nf, max(1.0, max_g * 0.5))
        elif state.speed <= c.low_speed_mps: nx = max(nx, -0.5)
        if state.speed >= c.max_speed_mps:
            nx = min(nx, -0.5)
        elif state.speed >= c.high_speed_mps:
            nx = min(nx, 0.0)
        altitude = state.position.y
        if altitude <= c.min_altitude_m: gamma_s = 0.0; nf = max(1.5, min(nf, max_g))
        elif altitude <= c.low_altitude_m and math.cos(gamma_s) < 0.0:
            gamma_s = math.copysign(c.max_dive_gamma_s_rad, gamma_s)
        return ManeuverCommand(float(nx), float(nf), float(gamma_s))
