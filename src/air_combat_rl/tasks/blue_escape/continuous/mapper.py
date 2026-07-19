"""Map projected continuous commands to nearest legal 29-action maneuver."""
from __future__ import annotations
from dataclasses import dataclass
import math
from air_combat_rl.core.units import STANDARD_GRAVITY
from air_combat_rl.domain.commands import ManeuverCommand
from air_combat_rl.domain.states import FlightState
from air_combat_rl.tasks.blue_escape.action_catalog import ActionCatalog, SAFE_FALLBACK_ACTION_ID

@dataclass(frozen=True, slots=True)
class EffectVector: dV_dt: float; dgamma_dt: float; dpsi_dt: float
@dataclass(frozen=True, slots=True)
class NearestManeuverConfig:
    w_v: float = 1.0; w_gamma: float = 1.0; w_psi: float = 1.0
    scale_v: float = 30.0; scale_gamma: float = 0.5; scale_psi: float = 0.5
    min_speed: float = 1e-3; min_abs_cos_gamma: float = 1e-3
@dataclass(frozen=True, slots=True)
class MappingResult:
    action_id: int; command: ManeuverCommand; distance: float; fallback_used: bool; valid_action_count: int

class NearestManeuverMapper:
    def __init__(self, actions: ActionCatalog, config: NearestManeuverConfig | None = None) -> None:
        self.actions = actions; self.config = config or NearestManeuverConfig()
    def effect(self, state: FlightState, command: ManeuverCommand) -> EffectVector:
        c = self.config; g = state.angles.gamma; speed = max(abs(state.speed), c.min_speed)
        cos_g = math.cos(g); cos_g = math.copysign(max(abs(cos_g), c.min_abs_cos_gamma), cos_g if cos_g else 1.0)
        return EffectVector(STANDARD_GRAVITY*(command.nx-math.sin(g)), STANDARD_GRAVITY/speed*(command.nf*math.cos(command.gamma_s)-math.cos(g)), STANDARD_GRAVITY*command.nf*math.sin(command.gamma_s)/(speed*cos_g))
    def distance(self, a: EffectVector, b: EffectVector) -> float:
        c = self.config
        return c.w_v*((a.dV_dt-b.dV_dt)/c.scale_v)**2 + c.w_gamma*((a.dgamma_dt-b.dgamma_dt)/c.scale_gamma)**2 + c.w_psi*((a.dpsi_dt-b.dpsi_dt)/c.scale_psi)**2
    def map(self, state: FlightState, command: ManeuverCommand, platform: str, valid_actions: list[int] | tuple[int, ...] | None = None) -> MappingResult:
        if valid_actions is None:
            mask = self.actions.action_mask(platform); valid_actions = [i for i,m in enumerate(mask) if m]
        if not valid_actions:
            fb = self.actions.command_for(SAFE_FALLBACK_ACTION_ID, platform)
            return MappingResult(SAFE_FALLBACK_ACTION_ID, fb, 0.0, True, 0)
        target = self.effect(state, command); best_id = min(valid_actions, key=lambda i: (self.distance(self.effect(state, self.actions.action(i).command), target), i))
        best_cmd = self.actions.command_for(best_id, platform)
        return MappingResult(best_id, best_cmd, self.distance(self.effect(state, best_cmd), target), False, len(valid_actions))
