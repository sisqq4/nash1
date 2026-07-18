"""Config-driven scenario construction for blue escape experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
import math, random
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
    max_episode_time_s: float = 60.0
    physics_dt: float = 0.005
    policy_dt: float = 0.1
    world: WorldConfig = field(default_factory=WorldConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ScenarioConfig":
        data: dict[str, object] = {}
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                value = value.strip()
                if value.startswith("[") and value.endswith("]"):
                    data[key] = tuple(float(x.strip()) for x in value[1:-1].split(",") if x.strip())
                elif value.replace(".", "", 1).isdigit():
                    data[key] = float(value) if "." in value else int(value)
                else:
                    data[key] = value
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def build_scenario(config: ScenarioConfig) -> SimulationWorld:
    rng = random.Random(config.seed)
    mode = config.mode
    n = config.missile_count
    powered = False
    dist = 20000.0
    mspeed = 900.0
    if mode in {"1v2"}: n = 2
    if mode in {"1vN"}: n = max(1, config.missile_count)
    if mode == "high_threat_1v1": dist = 12000.0; mspeed = 1200.0
    if mode == "terminal_intercept": dist = config.terminal_distance_m; mspeed = config.terminal_speed_mps
    if mode == "powered_launch": powered = True; mspeed = config.powered_speed_mps; dist = 15000.0
    alt = rng.uniform(*config.blue_altitude_m) if mode == "randomized_1v1" else sum(config.blue_altitude_m)/2.0
    blue = AircraftState(KinematicState(VecXZY(0.0, 0.0, alt), config.blue_speed_mps, FlightPathAngles(0.0, 0.0)), True, "zdj")
    missiles=[]
    for i in range(n):
        angle = 0.0 if n == 1 else (2*math.pi*i/n)
        pos = VecXZY(dist*math.cos(angle), dist*math.sin(angle), config.missile_altitude_m)
        psi = math.atan2(-pos.z, -pos.x)
        missiles.append(MissileState(KinematicState(pos, mspeed, FlightPathAngles(0.0, psi)), True, True, powered, 0.0))
    return SimulationWorld(blue=blue, missiles=missiles, clock=SimulationClock(config.physics_dt, config.policy_dt), config=config.world)
