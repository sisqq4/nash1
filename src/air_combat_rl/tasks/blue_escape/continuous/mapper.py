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
    w_nx: float = 1.0; w_nf: float = 1.0; w_gamma_s: float = 1.0
    range_nx: float = 18.0; range_nf: float = 9.0; range_gamma_s: float = math.pi
    min_speed: float = 1e-3; min_abs_cos_gamma: float = 1e-3

@dataclass(frozen=True, slots=True)
class MappingResult:
    action_id: int; action_name: str; command: ManeuverCommand; distance: float; fallback_used: bool; valid_action_count: int; continuous_command: ManeuverCommand | None = None

class NearestManeuverMapper:
    def __init__(self, actions: ActionCatalog, config: NearestManeuverConfig | None = None) -> None:
        self.actions = actions; self.config = config or NearestManeuverConfig()
    def effect(self, state: FlightState, command: ManeuverCommand) -> EffectVector:
        c = self.config; g = state.angles.gamma; speed = max(abs(state.speed), c.min_speed)
        cos_g = math.cos(g); cos_g = math.copysign(max(abs(cos_g), c.min_abs_cos_gamma), cos_g if cos_g else 1.0)
        return EffectVector(STANDARD_GRAVITY*(command.nx-math.sin(g)), STANDARD_GRAVITY/speed*(command.nf*math.cos(command.gamma_s)-math.cos(g)), STANDARD_GRAVITY*command.nf*math.sin(command.gamma_s)/(speed*cos_g))
    @staticmethod
    def angular_distance(a: float, b: float) -> float:
        return math.atan2(math.sin(a - b), math.cos(a - b))
    def command_distance(self, a: ManeuverCommand, b: ManeuverCommand) -> float:
        c = self.config
        return (c.w_nx*((a.nx-b.nx)/c.range_nx)**2 + c.w_nf*((a.nf-b.nf)/c.range_nf)**2 + c.w_gamma_s*(self.angular_distance(a.gamma_s,b.gamma_s)/c.range_gamma_s)**2)
    def distance(self, a: EffectVector, b: EffectVector) -> float:
        raise NotImplementedError("use command_distance for projected PPO action projection")
    def map(self, state: FlightState, command: ManeuverCommand, platform: str, valid_actions: list[int] | tuple[int, ...] | None = None) -> MappingResult:
        if valid_actions is None:
            mask = self.actions.action_mask(platform); valid_actions = [i for i,m in enumerate(mask) if m]
        if not valid_actions:
            fb_action = self.actions.action(SAFE_FALLBACK_ACTION_ID); fb = self.actions.command_for(SAFE_FALLBACK_ACTION_ID, platform)
            return MappingResult(SAFE_FALLBACK_ACTION_ID, fb_action.name, fb, 0.0, True, 0, command)
        best_id = min(valid_actions, key=lambda i: (self.command_distance(self.actions.action(i).command, command), i))
        best_action = self.actions.action(best_id); best_cmd = self.actions.command_for(best_id, platform)
        return MappingResult(best_id, best_action.name, best_cmd, self.command_distance(best_cmd, command), False, len(valid_actions), command)
