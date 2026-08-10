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
    blue_heading_deg: float = 0.0
    missile_count: int = 1
    terminal_distance_m: float = 30000.0
    terminal_speed_mps: float = 1800.0
    powered_speed_mps: float = 250.0
    missile_altitude_m: float = 10000.0
    missile_launch_climb_angle_deg: float = 20.0
    missile_spawn_distance_m: tuple[float, float] = (20000.0, 20000.0)
    missile_spawn_bearing_deg: tuple[float, float] = (0.0, 360.0)
    missile_spawn_altitude_m: tuple[float, float] | None = None
    missile_first_launch_time_s: float = 0.0
    missile_launch_interval_s: float = 0.0
    blue_detection_range_m: float = 30000.0
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
        for range_field in ("blue_altitude_m", "missile_spawn_distance_m",
                            "missile_spawn_bearing_deg", "missile_spawn_altitude_m"):
            if range_field not in data or data[range_field] is None:
                continue
            values = data[range_field]
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                raise ValueError(f"{range_field} must contain [minimum, maximum]")
            data[range_field] = tuple(float(value) for value in values)
        if isinstance(data.get("world"), dict):
            data["world"] = WorldConfig(**data["world"])
        result = cls(**{key: value for key, value in data.items() if key in allowed})
        if result.physics_dt <= 0 or result.policy_dt <= 0:
            raise ValueError("physics_dt and policy_dt must be positive")
        if result.physics_dt > result.policy_dt:
            raise ValueError("physics_dt must not exceed policy_dt")
        if result.missile_count <= 0:
            raise ValueError("missile_count must be positive")
        if result.missile_spawn_distance_m[0] < 0 or result.missile_spawn_distance_m[0] > result.missile_spawn_distance_m[1]:
            raise ValueError("missile_spawn_distance_m must be a non-negative ordered range")
        if result.missile_spawn_bearing_deg[0] > result.missile_spawn_bearing_deg[1]:
            raise ValueError("missile_spawn_bearing_deg must be an ordered range")
        if result.missile_spawn_altitude_m is not None and result.missile_spawn_altitude_m[0] > result.missile_spawn_altitude_m[1]:
            raise ValueError("missile_spawn_altitude_m must be an ordered range")
        if result.missile_first_launch_time_s < 0 or result.missile_launch_interval_s < 0:
            raise ValueError("missile launch timing must be non-negative")
        if result.blue_detection_range_m <= 0:
            raise ValueError("blue_detection_range_m must be positive")
        numeric_values = (
            result.blue_heading_deg, *result.missile_spawn_distance_m,
            *result.missile_spawn_bearing_deg, result.missile_first_launch_time_s,
            result.missile_launch_interval_s, result.blue_detection_range_m,
        )
        if result.missile_spawn_altitude_m is not None:
            numeric_values += result.missile_spawn_altitude_m
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("scenario launch and detection parameters must be finite")
        if not -90.0 < result.missile_launch_climb_angle_deg < 90.0:
            raise ValueError("missile_launch_climb_angle_deg must be between -90 and 90 degrees")
        return result


def build_scenario(config: ScenarioConfig) -> SimulationWorld:
    rng = random.Random(config.seed)
    mode = config.mode
    n = config.missile_count
    powered = True
    default_dist = 20000.0
    mspeed = 900.0
    if mode in {"1v2"}: n = 2
    if mode in {"1vN"}: n = max(1, config.missile_count)
    if mode == "high_threat_1v1": default_dist = 12000.0; mspeed = 1200.0
    if mode == "terminal_intercept": default_dist = config.terminal_distance_m; mspeed = config.terminal_speed_mps
    if mode == "powered_launch": mspeed = config.powered_speed_mps; default_dist = 15000.0
    alt = rng.uniform(*config.blue_altitude_m) if mode == "randomized_1v1" else sum(config.blue_altitude_m)/2.0
    blue = AircraftState(KinematicState(VecXZY(0.0, 0.0, alt), config.blue_speed_mps, FlightPathAngles(0.0, math.radians(config.blue_heading_deg))), True, "zdj")
    missiles=[]
    for i in range(n):
        distance_range = config.missile_spawn_distance_m
        # Preserve legacy mode-specific distances unless an explicit region is configured.
        dist = default_dist if distance_range == (20000.0, 20000.0) and default_dist != 20000.0 else rng.uniform(*distance_range)
        bearing_min, bearing_max = config.missile_spawn_bearing_deg
        angle = (
            0.0 if n == 1 else 2 * math.pi * i / n
        ) if (bearing_min, bearing_max) == (0.0, 360.0) else math.radians(
            rng.uniform(bearing_min, bearing_max)
        )
        altitude_range = config.missile_spawn_altitude_m
        missile_altitude = rng.uniform(*altitude_range) if altitude_range else config.missile_altitude_m
        pos = VecXZY(dist*math.cos(angle), dist*math.sin(angle), missile_altitude)
        psi = math.atan2(-pos.z, -pos.x)
        gamma = math.radians(config.missile_launch_climb_angle_deg)
        launch_time = config.missile_first_launch_time_s + i * config.missile_launch_interval_s
        missiles.append(MissileState(KinematicState(pos, mspeed, FlightPathAngles(gamma, psi)), True, True, powered, 0.0, launch_time))
    return SimulationWorld(blue=blue, missiles=missiles, clock=SimulationClock(config.physics_dt, config.policy_dt), config=config.world, blue_detection_range_m=config.blue_detection_range_m)
