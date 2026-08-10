"""Config-driven scenario construction for blue escape experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
import math, random
from pathlib import Path
import yaml
from src.air_combat_rl.core.coordinates import VecXZY
from src.air_combat_rl.core.timebase import SimulationClock
from src.air_combat_rl.domain.states import AircraftState, FlightPathAngles, KinematicState, MissileState
from src.air_combat_rl.simulation.world import SimulationWorld, WorldConfig


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    mode: str = "fixed_1v1"
    seed: int = 0
    blue_altitude_m: tuple[float, float] = (8000.0, 12000.0)
    blue_speed_mps: float | tuple[float, float] = (300.0, 400.0)
    blue_heading_deg: float | tuple[float, float] = (-180.0, 180.0)
    missile_count: int = 1
    terminal_distance_m: float = 30000.0
    terminal_speed_mps: float = 1800.0
    powered_speed_mps: float = 250.0
    missile_altitude_m: float = 9000.0
    missile_initial_mach: tuple[float, float] = (0.6, 0.9)
    reference_sound_speed_mps: float = 295.0
    missile_launch_climb_angle_deg: float = 20.0
    missile_spawn_distance_m: tuple[float, float] = (140000.0, 160000.0)
    missile_spawn_bearing_deg: tuple[float, float] = (-30.0, 30.0)
    missile_spawn_altitude_m: tuple[float, float] | None = (8000.0, 10000.0)
    missile_heading_offset_deg: tuple[float, float] = (-15.0, 15.0)
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
        for range_field in ("blue_altitude_m", "missile_initial_mach", "missile_spawn_distance_m",
                            "missile_spawn_bearing_deg", "missile_spawn_altitude_m", "missile_heading_offset_deg"):
            if range_field not in data or data[range_field] is None:
                continue
            values = data[range_field]
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                raise ValueError(f"{range_field} must contain [minimum, maximum]")
            data[range_field] = tuple(float(value) for value in values)
        for scalar_or_range in ("blue_speed_mps", "blue_heading_deg"):
            if scalar_or_range not in data or not isinstance(data[scalar_or_range], (list, tuple)):
                continue
            values = data[scalar_or_range]
            if len(values) != 2:
                raise ValueError(f"{scalar_or_range} must be a scalar or [minimum, maximum]")
            data[scalar_or_range] = tuple(float(value) for value in values)
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
        if result.blue_altitude_m[0] > result.blue_altitude_m[1]:
            raise ValueError("blue_altitude_m must be an ordered range")
        if result.missile_heading_offset_deg[0] > result.missile_heading_offset_deg[1]:
            raise ValueError("missile_heading_offset_deg must be an ordered range")
        if result.missile_first_launch_time_s < 0 or result.missile_launch_interval_s < 0:
            raise ValueError("missile launch timing must be non-negative")
        if result.blue_detection_range_m <= 0:
            raise ValueError("blue_detection_range_m must be positive")
        numeric_values = (
            *result.blue_altitude_m,
            *result.missile_spawn_distance_m,
            *result.missile_spawn_bearing_deg, result.missile_first_launch_time_s,
            result.missile_launch_interval_s, result.blue_detection_range_m,
            result.reference_sound_speed_mps, *result.missile_initial_mach,
            *result.missile_heading_offset_deg, result.terminal_distance_m,
            result.terminal_speed_mps, result.powered_speed_mps,
            result.missile_altitude_m, result.max_episode_time_s,
        )
        for value in (result.blue_speed_mps, result.blue_heading_deg):
            numeric_values += tuple(value) if isinstance(value, tuple) else (value,)
        if result.missile_spawn_altitude_m is not None:
            numeric_values += result.missile_spawn_altitude_m
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("scenario launch and detection parameters must be finite")
        if not -90.0 < result.missile_launch_climb_angle_deg < 90.0:
            raise ValueError("missile_launch_climb_angle_deg must be between -90 and 90 degrees")
        if result.missile_initial_mach[0] <= 0 or result.missile_initial_mach[0] > result.missile_initial_mach[1]:
            raise ValueError("missile_initial_mach must be a positive ordered range")
        if result.reference_sound_speed_mps <= 0:
            raise ValueError("reference_sound_speed_mps must be positive")
        if result.blue_altitude_m[0] < 0:
            raise ValueError("blue_altitude_m must be non-negative")
        if result.missile_spawn_altitude_m is not None and result.missile_spawn_altitude_m[0] < 0:
            raise ValueError("missile_spawn_altitude_m must be non-negative")
        if min(result.terminal_distance_m, result.terminal_speed_mps, result.powered_speed_mps,
               result.missile_altitude_m, result.max_episode_time_s) <= 0:
            raise ValueError("scenario distances, speeds, altitudes, and duration must be positive")
        if isinstance(result.blue_speed_mps, tuple):
            if result.blue_speed_mps[0] <= 0 or result.blue_speed_mps[0] > result.blue_speed_mps[1]:
                raise ValueError("blue_speed_mps must be a positive ordered range")
        elif result.blue_speed_mps <= 0:
            raise ValueError("blue_speed_mps must be positive")
        return result


def build_scenario(config: ScenarioConfig) -> SimulationWorld:
    rng = random.Random(config.seed)
    mode = config.mode
    n = config.missile_count
    powered = True
    default_dist = None
    mspeed = None
    if mode in {"1v2"}: n = 2
    if mode in {"1vN"}: n = max(1, config.missile_count)
    if mode == "high_threat_1v1": default_dist = 12000.0; mspeed = 1200.0
    if mode == "terminal_intercept": default_dist = config.terminal_distance_m; mspeed = config.terminal_speed_mps
    if mode == "powered_launch": mspeed = config.powered_speed_mps; default_dist = 15000.0
    alt = rng.uniform(*config.blue_altitude_m)
    blue_speed = _sample(rng, config.blue_speed_mps)
    blue_heading = _sample(rng, config.blue_heading_deg)
    blue = AircraftState(KinematicState(VecXZY(0.0, 0.0, alt), blue_speed, FlightPathAngles(0.0, math.radians(blue_heading))), True, "zdj")
    missiles=[]
    for i in range(n):
        distance_range = config.missile_spawn_distance_m
        # sqrt sampling makes points uniform by area in the annular sector.
        dist = default_dist if default_dist is not None else math.sqrt(rng.uniform(distance_range[0] ** 2, distance_range[1] ** 2))
        bearing_min, bearing_max = config.missile_spawn_bearing_deg
        angle = math.radians(rng.uniform(bearing_min, bearing_max))
        altitude_range = config.missile_spawn_altitude_m
        missile_altitude = rng.uniform(*altitude_range) if altitude_range else config.missile_altitude_m
        pos = VecXZY(dist*math.cos(angle), dist*math.sin(angle), missile_altitude)
        psi = math.atan2(-pos.z, -pos.x) + math.radians(rng.uniform(*config.missile_heading_offset_deg))
        gamma = math.radians(config.missile_launch_climb_angle_deg)
        initial_speed = mspeed if mspeed is not None else rng.uniform(*config.missile_initial_mach) * config.reference_sound_speed_mps
        launch_time = config.missile_first_launch_time_s + i * config.missile_launch_interval_s
        missiles.append(MissileState(KinematicState(pos, initial_speed, FlightPathAngles(gamma, psi)), True, True, powered, 0.0, launch_time))
    return SimulationWorld(blue=blue, missiles=missiles, clock=SimulationClock(config.physics_dt, config.policy_dt), config=config.world, blue_detection_range_m=config.blue_detection_range_m)


def _sample(rng: random.Random, value: float | tuple[float, float]) -> float:
    return rng.uniform(*value) if isinstance(value, tuple) else float(value)
