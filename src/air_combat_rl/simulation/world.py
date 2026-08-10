"""Simulation world orchestration that accepts only physical commands."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

from src.air_combat_rl.core.coordinates import VecXZY
from src.air_combat_rl.core.math3d import flight_velocity, global_to_local
from src.air_combat_rl.core.timebase import SimulationClock
from src.air_combat_rl.core.units import STANDARD_GRAVITY
from src.air_combat_rl.domain.collision.closest_approach import segment_closest_approach
from src.air_combat_rl.domain.commands import ManeuverCommand
from src.air_combat_rl.domain.dynamics.aircraft_3dof import integrate_aircraft
from src.air_combat_rl.domain.dynamics.missile_3dof import MissileCommand, integrate_missile, proportional_navigation_command
from src.air_combat_rl.domain.events import SimulationEvent
from src.air_combat_rl.domain.propulsion.drag import DragResult, missile_drag
from src.air_combat_rl.domain.states import AircraftState, KinematicState, MissileState
from src.air_combat_rl.simulation.snapshot import WorldSnapshot


@dataclass(frozen=True, slots=True)
class WorldConfig:
    navigation_constant: float = 3.5
    kill_radius_m: float = 5.0
    missile_stall_speed_mps: float = 250.0
    max_guidance_time_s: float = 180.0
    missile_dry_mass_kg: float = 120.0
    missile_propellant_mass_kg: float = 45.0
    missile_boost_time_s: float = 7.0
    missile_boost_target_mach: float = 6.0
    reference_sound_speed_mps: float = 295.0
    missile_boost_climb_angle_deg: float = 20.0
    missile_reference_area_m2: float = 0.028
    missile_induced_drag_factor: float = 0.08
    missile_drag_coefficient: float | None = None
    missile_max_load_g: float = 35.0
    seeker_acquisition_fov_deg: float = 35.0
    seeker_tracking_fov_deg: float = 60.0
    seeker_lock_hold_time_s: float = 0.75
    passed_distance_growth_m: float = 600.0
    passed_receding_speed_mps: float = 40.0
    success_distance_m: float = 30_000.0
    blue_min_speed_mps: float = 100.0
    blue_max_speed_mps: float = 600.0
    blue_min_altitude_m: float = 8_000.0
    blue_max_altitude_m: float = 12_000.0

    def __post_init__(self) -> None:
        positive = {
            "navigation_constant": self.navigation_constant,
            "kill_radius_m": self.kill_radius_m,
            "max_guidance_time_s": self.max_guidance_time_s,
            "missile_dry_mass_kg": self.missile_dry_mass_kg,
            "missile_boost_target_mach": self.missile_boost_target_mach,
            "reference_sound_speed_mps": self.reference_sound_speed_mps,
            "missile_reference_area_m2": self.missile_reference_area_m2,
            "missile_max_load_g": self.missile_max_load_g,
            "seeker_acquisition_fov_deg": self.seeker_acquisition_fov_deg,
            "seeker_tracking_fov_deg": self.seeker_tracking_fov_deg,
            "passed_distance_growth_m": self.passed_distance_growth_m,
            "passed_receding_speed_mps": self.passed_receding_speed_mps,
            "success_distance_m": self.success_distance_m,
        }
        if any(not math.isfinite(value) or value <= 0 for value in positive.values()):
            raise ValueError("missile and world positive parameters must be finite and positive")
        nonnegative = (self.missile_propellant_mass_kg, self.missile_boost_time_s,
                       self.missile_induced_drag_factor, self.seeker_lock_hold_time_s,
                       self.missile_stall_speed_mps)
        if any(not math.isfinite(value) or value < 0 for value in nonnegative):
            raise ValueError("missile non-negative parameters must be finite and non-negative")
        if not 0 < self.seeker_acquisition_fov_deg <= self.seeker_tracking_fov_deg <= 180:
            raise ValueError("seeker FOV must satisfy 0 < acquisition <= tracking <= 180")
        if not self.blue_min_speed_mps < self.blue_max_speed_mps:
            raise ValueError("blue minimum speed must be below maximum speed")
        if not self.blue_min_altitude_m < self.blue_max_altitude_m:
            raise ValueError("blue minimum altitude must be below maximum altitude")
        envelope = (self.blue_min_speed_mps, self.blue_max_speed_mps,
                    self.blue_min_altitude_m, self.blue_max_altitude_m,
                    self.missile_boost_climb_angle_deg)
        if not all(math.isfinite(value) for value in envelope):
            raise ValueError("flight-envelope parameters must be finite")
        if self.blue_min_speed_mps < 0 or self.blue_min_altitude_m < 0:
            raise ValueError("blue minimum speed and altitude must be non-negative")
        if not -90.0 < self.missile_boost_climb_angle_deg < 90.0:
            raise ValueError("missile boost climb angle must be between -90 and 90 degrees")
        if self.missile_drag_coefficient is not None and (
            not math.isfinite(self.missile_drag_coefficient) or self.missile_drag_coefficient < 0
        ):
            raise ValueError("fixed missile drag coefficient must be finite and non-negative")


@dataclass(slots=True)
class MissileRuntimeState:
    seeker_mode: str = "boost"
    last_lock_time_s: float | None = None
    estimated_target: KinematicState | None = None
    fuel_mass_kg: float = 45.0
    mass_kg: float = 165.0
    pn_load: tuple[float, float] = (0.0, 0.0)
    gravity_compensation_load: tuple[float, float] = (0.0, 0.0)
    final_load: tuple[float, float, float] = (0.0, 0.0, 0.0)
    latest_drag: DragResult | None = None
    failure_reason: str | None = None


@dataclass(slots=True)
class SimulationWorld:
    blue: AircraftState
    missiles: list[MissileState] = field(default_factory=list)
    clock: SimulationClock = field(default_factory=SimulationClock)
    config: WorldConfig = field(default_factory=WorldConfig)
    time_s: float = 0.0
    substeps_last_interval: int = 0
    min_missile_distances: list[float] = field(default_factory=list)
    closest_approach_passed: list[bool] = field(default_factory=list)
    blue_detection_range_m: float = 30000.0
    launched_missiles: set[int] = field(default_factory=set)
    missile_runtime: list[MissileRuntimeState] = field(default_factory=list)

    def __post_init__(self) -> None:
        count = len(self.missiles)
        if not self.min_missile_distances or len(self.min_missile_distances) != count:
            self.min_missile_distances = [math.inf] * count
        if not self.closest_approach_passed or len(self.closest_approach_passed) != count:
            self.closest_approach_passed = [False] * count
        if not self.missile_runtime or len(self.missile_runtime) != count:
            total_mass = self.config.missile_dry_mass_kg + self.config.missile_propellant_mass_kg
            self.missile_runtime = [MissileRuntimeState(
                seeker_mode="boost" if missile.powered else "locked",
                estimated_target=self.blue.kinematics,
                last_lock_time_s=self.time_s if missile.locked and not missile.powered else None,
                fuel_mass_kg=self.config.missile_propellant_mass_kg if missile.powered else 0.0,
                mass_kg=total_mass if missile.powered else self.config.missile_dry_mass_kg,
            ) for missile in self.missiles]

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(self.time_s, self.blue, tuple(self.missiles))

    def missile_is_launched(self, missile: MissileState) -> bool:
        return self.time_s + 1.0e-9 >= missile.launch_time_s

    def blue_detects_threat(self) -> bool:
        return any(self.missile_is_launched(missile) and missile.alive
                   and _distance(missile.kinematics, self.blue.kinematics) <= self.blue_detection_range_m
                   for missile in self.missiles)

    def step_policy_interval(self, command: ManeuverCommand) -> tuple[WorldSnapshot, tuple[SimulationEvent, ...]]:
        events: list[SimulationEvent] = []
        self.substeps_last_interval = 0
        for _ in range(self.clock.substeps_per_policy_step):
            events.extend(self._step_physics_substep(command))
            if self._blue_terminal_event(events):
                break
        return self.snapshot(), tuple(events)

    def step_held_policy_interval(self, held_action) -> tuple[WorldSnapshot, tuple[SimulationEvent, ...]]:
        events: list[SimulationEvent] = []
        self.substeps_last_interval = 0
        for _ in range(self.clock.substeps_per_policy_step):
            events.extend(self._step_physics_substep(held_action.consume_substep()))
            if self._blue_terminal_event(events):
                break
        return self.snapshot(), tuple(events)

    def all_live_threats_safely_passed(self) -> bool:
        if not self.missiles:
            return True
        return all(
            not missile.alive or (
                self.closest_approach_passed[i]
                and _distance(missile.kinematics, self.blue.kinematics) >= self.config.success_distance_m
                and _closing_speed(missile, self.blue) <= 0.0
            )
            for i, missile in enumerate(self.missiles)
        )

    def fail_all_missiles(self, reason: str) -> None:
        self.missiles = [replace(missile, alive=False) if missile.alive else missile for missile in self.missiles]
        for runtime in self.missile_runtime:
            if runtime.failure_reason is None:
                runtime.failure_reason = reason

    def _step_physics_substep(self, command: ManeuverCommand) -> list[SimulationEvent]:
        events: list[SimulationEvent] = []
        self.substeps_last_interval += 1
        blue_start = self.blue.kinematics
        blue_kinematics = integrate_aircraft(blue_start, command, self.clock.physics_dt)
        blue_kinematics = replace(
            blue_kinematics,
            speed=min(self.config.blue_max_speed_mps, max(self.config.blue_min_speed_mps, blue_kinematics.speed)),
            position=replace(blue_kinematics.position, y=min(self.config.blue_max_altitude_m, max(self.config.blue_min_altitude_m, blue_kinematics.position.y))),
        )
        self.blue = replace(self.blue, kinematics=blue_kinematics)

        updated: list[MissileState] = []
        for i, missile in enumerate(self.missiles):
            runtime = self.missile_runtime[i]
            if not missile.alive or not self.missile_is_launched(missile):
                updated.append(missile)
                continue
            if i not in self.launched_missiles:
                self.launched_missiles.add(i)
                events.append(SimulationEvent(self.time_s, "missile_launch", f"missile_{i}", {}))

            start = missile.kinematics
            age = missile.age_s + self.clock.physics_dt
            powered = missile.powered and age <= self.config.missile_boost_time_s
            if powered:
                mcmd = self._boost_command(start, age, runtime)
                runtime.seeker_mode = "boost"
            else:
                if missile.powered:
                    start = replace(start, speed=self.config.missile_boost_target_mach * self.config.reference_sound_speed_mps)
                    runtime.fuel_mass_kg = 0.0
                    runtime.mass_kg = self.config.missile_dry_mass_kg
                target, locked = self._seeker_target(start, runtime)
                pn = proportional_navigation_command(start, target, self.config.navigation_constant) if target is not None else MissileCommand(0.0, 0.0, 0.0)
                runtime.pn_load = (pn.nn, pn.ns)
                nn, ns = _limit_lateral_load(pn.nn, pn.ns, self.config.missile_max_load_g)
                drag = missile_drag(start.speed, start.position.y, runtime.mass_kg,
                                    self.config.missile_reference_area_m2,
                                    self.config.missile_induced_drag_factor,
                                    math.hypot(nn, ns), self.config.missile_drag_coefficient)
                runtime.latest_drag = drag
                mcmd = MissileCommand(-drag.acceleration_mps2 / STANDARD_GRAVITY, nn, ns)
                missile = replace(missile, locked=locked)
            runtime.final_load = (mcmd.nx, mcmd.nn, mcmd.ns)
            kin = integrate_missile(start, mcmd, self.clock.physics_dt)
            if powered and age + 1.0e-9 >= self.config.missile_boost_time_s:
                kin = replace(kin, speed=self.config.missile_boost_target_mach * self.config.reference_sound_speed_mps)
                powered = False
                runtime.fuel_mass_kg = 0.0
                runtime.mass_kg = self.config.missile_dry_mass_kg
            alive = True
            reason = None

            approach = segment_closest_approach(start.position, kin.position, blue_start.position, blue_kinematics.position)
            d = _distance(kin, blue_kinematics)
            self.min_missile_distances[i] = min(self.min_missile_distances[i], approach.distance_m)
            if approach.distance_m <= self.config.kill_radius_m:
                alive = False
                reason = "hit"
                self.blue = replace(self.blue, alive=False)
                events.append(SimulationEvent(self.time_s + approach.fraction * self.clock.physics_dt, "hit", f"missile_{i}", {"distance_m": approach.distance_m}))
            elif kin.position.y <= 0.0:
                alive = False; reason = "ground_collision"
                events.append(SimulationEvent(self.time_s, "missile_ground_collision", f"missile_{i}", {}))
            elif age >= self.config.max_guidance_time_s:
                alive = False; reason = "guidance_timeout"
                events.append(SimulationEvent(self.time_s, "missile_timeout", f"missile_{i}", {"age_s": age}))
            elif not powered and kin.speed < self.config.missile_stall_speed_mps:
                alive = False; reason = "stall"
                events.append(SimulationEvent(self.time_s, "missile_stall", f"missile_{i}", {"speed": kin.speed}))
            elif (d >= self.min_missile_distances[i] + self.config.passed_distance_growth_m
                  and _closing_speed_kinematics(kin, blue_kinematics) <= -self.config.passed_receding_speed_mps):
                alive = False; reason = "passed_target"
                self.closest_approach_passed[i] = True
                events.append(SimulationEvent(self.time_s, "missile_passed", f"missile_{i}", {"min_distance_m": self.min_missile_distances[i], "distance_m": d}))
            if reason is not None:
                runtime.failure_reason = reason
            updated.append(replace(missile, kinematics=kin, alive=alive, age_s=age, powered=powered))

        self.missiles = updated
        self.time_s += self.clock.physics_dt
        return events

    def _boost_command(self, state: KinematicState, age: float, runtime: MissileRuntimeState) -> MissileCommand:
        remaining = max(self.config.missile_boost_time_s - age + self.clock.physics_dt, self.clock.physics_dt)
        target_speed = self.config.missile_boost_target_mach * self.config.reference_sound_speed_mps
        target_gamma = math.radians(self.config.missile_boost_climb_angle_deg)
        normal_g = math.cos(state.angles.gamma) + (target_gamma - state.angles.gamma) * max(state.speed, 1.0) / (STANDARD_GRAVITY * remaining)
        burn_fraction = min(1.0, age / max(self.config.missile_boost_time_s, self.clock.physics_dt))
        runtime.fuel_mass_kg = self.config.missile_propellant_mass_kg * (1.0 - burn_fraction)
        runtime.mass_kg = self.config.missile_dry_mass_kg + runtime.fuel_mass_kg
        drag = missile_drag(state.speed, state.position.y, runtime.mass_kg,
                            self.config.missile_reference_area_m2,
                            self.config.missile_induced_drag_factor,
                            abs(normal_g), self.config.missile_drag_coefficient)
        runtime.latest_drag = drag
        tangential_g = ((target_speed - state.speed) / remaining + drag.acceleration_mps2) / STANDARD_GRAVITY + math.sin(state.angles.gamma)
        runtime.gravity_compensation_load = (math.cos(state.angles.gamma), 0.0)
        return MissileCommand(tangential_g, normal_g, 0.0)

    def _seeker_target(self, missile: KinematicState, runtime: MissileRuntimeState) -> tuple[KinematicState | None, bool]:
        rel = VecXZY(self.blue.kinematics.position.x - missile.position.x,
                     self.blue.kinematics.position.z - missile.position.z,
                     self.blue.kinematics.position.y - missile.position.y)
        local = global_to_local(rel, missile.angles)
        off_boresight = math.degrees(math.atan2(math.hypot(local.z, local.y), local.x))
        fov = self.config.seeker_tracking_fov_deg if runtime.last_lock_time_s is not None else self.config.seeker_acquisition_fov_deg
        if off_boresight <= fov:
            runtime.estimated_target = self.blue.kinematics
            runtime.last_lock_time_s = self.time_s
            runtime.seeker_mode = "locked"
            return self.blue.kinematics, True
        if runtime.last_lock_time_s is not None and self.time_s - runtime.last_lock_time_s <= self.config.seeker_lock_hold_time_s:
            runtime.seeker_mode = "lock_hold"
            runtime.estimated_target = _extrapolate(runtime.estimated_target, self.clock.physics_dt)
            return runtime.estimated_target, False
        runtime.seeker_mode = "inertial"
        runtime.estimated_target = _extrapolate(runtime.estimated_target, self.clock.physics_dt)
        return runtime.estimated_target, False

    @staticmethod
    def _blue_terminal_event(events: list[SimulationEvent]) -> bool:
        return any(event.kind == "hit" for event in events)


def _limit_lateral_load(nn: float, ns: float, limit: float) -> tuple[float, float]:
    magnitude = math.hypot(nn, ns)
    if magnitude <= limit:
        return nn, ns
    scale = limit / magnitude
    return nn * scale, ns * scale


def _extrapolate(state: KinematicState | None, dt: float) -> KinematicState | None:
    if state is None:
        return None
    velocity = flight_velocity(state)
    return replace(state, position=VecXZY(state.position.x + velocity.x * dt,
                                          state.position.z + velocity.z * dt,
                                          state.position.y + velocity.y * dt))


def _distance(a: KinematicState, b: KinematicState) -> float:
    return math.sqrt((a.position.x-b.position.x)**2 + (a.position.z-b.position.z)**2 + (a.position.y-b.position.y)**2)


def _closing_speed_kinematics(missile: KinematicState, blue: KinematicState) -> float:
    rel = VecXZY(blue.position.x - missile.position.x, blue.position.z - missile.position.z, blue.position.y - missile.position.y)
    mv, bv = flight_velocity(missile), flight_velocity(blue)
    rel_v = VecXZY(bv.x - mv.x, bv.z - mv.z, bv.y - mv.y)
    return -(rel.x * rel_v.x + rel.z * rel_v.z + rel.y * rel_v.y) / max(rel.norm(), 1.0e-6)


def _closing_speed(missile: MissileState, blue: AircraftState) -> float:
    return _closing_speed_kinematics(missile.kinematics, blue.kinematics)
