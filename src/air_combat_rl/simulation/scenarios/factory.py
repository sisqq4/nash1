"""Config-driven scenario construction for blue escape experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
import math, random
from pathlib import Path
import yaml
from air_combat_rl.core.coordinates import VecXZY
from air_combat_rl.core.timebase import SimulationClock
from air_combat_rl.domain.states import AircraftState, FlightPathAngles, KinematicState, MissileState
from air_combat_rl.simulation.world import SimulationWorld, WorldConfig


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    mode: str = "fixed_1v1"
    seed: int = 0
    blue_altitude_m: tuple[float, float] = (8000.0, 12000.0)
    blue_speed_mps: float = 300.0
    missile_count: int = 1
    terminal_distance_m: float = 30000.0
    terminal_speed_mps: float = 1800.0
    powered_speed_mps: float = 250.0
    missile_altitude_m: float = 10000.0
    missile_launch_climb_angle_deg: float = 20.0
    max_episode_time_s: float = 60.0
    physics_dt: float = 0.005
    policy_dt: float = 0.1
    world: WorldConfig = field(default_factory=WorldConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ScenarioConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("scenario config must be a YAML mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed - {"name"}
        if unknown:
            raise ValueError(f"unknown scenario fields: {sorted(unknown)}")
        if "blue_altitude_m" in data:
            values = data["blue_altitude_m"]
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                raise ValueError("blue_altitude_m must contain [minimum, maximum]")
            data["blue_altitude_m"] = tuple(float(value) for value in values)
        if isinstance(data.get("world"), dict):
            data["world"] = WorldConfig(**data["world"])
        result = cls(**{key: value for key, value in data.items() if key in allowed})
        if result.physics_dt <= 0 or result.policy_dt <= 0:
            raise ValueError("physics_dt and policy_dt must be positive")
        if result.physics_dt > result.policy_dt:
            raise ValueError("physics_dt must not exceed policy_dt")
        if result.missile_count <= 0:
            raise ValueError("missile_count must be positive")
        if not -90.0 < result.missile_launch_climb_angle_deg < 90.0:
            raise ValueError("missile_launch_climb_angle_deg must be between -90 and 90 degrees")
        return result


def build_scenario(config: ScenarioConfig) -> SimulationWorld:
    rng = random.Random(config.seed)
    mode = config.mode
    n = config.missile_count
    powered = True
    dist = 20000.0
    mspeed = 900.0
    if mode in {"1v2"}: n = 2
    if mode in {"1vN"}: n = max(1, config.missile_count)
    if mode == "high_threat_1v1": dist = 12000.0; mspeed = 1200.0
    if mode == "terminal_intercept": dist = config.terminal_distance_m; mspeed = config.terminal_speed_mps
    if mode == "powered_launch": mspeed = config.powered_speed_mps; dist = 15000.0
    alt = rng.uniform(*config.blue_altitude_m) if mode == "randomized_1v1" else sum(config.blue_altitude_m)/2.0
    blue = AircraftState(KinematicState(VecXZY(0.0, 0.0, alt), config.blue_speed_mps, FlightPathAngles(0.0, 0.0)), True, "zdj")
    missiles=[]
    for i in range(n):
        angle = 0.0 if n == 1 else (2*math.pi*i/n)
        pos = VecXZY(dist*math.cos(angle), dist*math.sin(angle), config.missile_altitude_m)
        psi = math.atan2(-pos.z, -pos.x)
        gamma = math.radians(config.missile_launch_climb_angle_deg)
        missiles.append(MissileState(KinematicState(pos, mspeed, FlightPathAngles(gamma, psi)), True, True, powered, 0.0))
    return SimulationWorld(blue=blue, missiles=missiles, clock=SimulationClock(config.physics_dt, config.policy_dt), config=config.world)
