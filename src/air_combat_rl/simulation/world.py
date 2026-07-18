"""Simulation world orchestration that accepts only physical commands."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from air_combat_rl.core.math3d import flight_velocity
from air_combat_rl.core.timebase import SimulationClock
from air_combat_rl.domain.commands import ManeuverCommand
from air_combat_rl.domain.dynamics.aircraft_3dof import integrate_aircraft
from air_combat_rl.domain.dynamics.missile_3dof import integrate_missile, proportional_navigation_command
from air_combat_rl.domain.events import SimulationEvent
from air_combat_rl.domain.states import AircraftState, MissileState
from air_combat_rl.simulation.snapshot import WorldSnapshot


@dataclass(frozen=True, slots=True)
class WorldConfig:
    navigation_constant: float = 4.5
    kill_radius_m: float = 5.0
    missile_stall_speed_mps: float = 250.0
    max_guidance_time_s: float = 120.0
    missile_boost_time_s: float = 7.0
    missile_boost_nx_g: float = 6.0
    closest_approach_event_threshold_m: float = 1.0
    success_distance_m: float = 30_000.0


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

    def __post_init__(self) -> None:
        if not self.min_missile_distances or len(self.min_missile_distances) != len(self.missiles):
            self.min_missile_distances = [math.inf] * len(self.missiles)
        if not self.closest_approach_passed or len(self.closest_approach_passed) != len(self.missiles):
            self.closest_approach_passed = [False] * len(self.missiles)

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(self.time_s, self.blue, tuple(self.missiles))

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
            command = held_action.consume_substep()
            events.extend(self._step_physics_substep(command))
            if self._blue_terminal_event(events):
                break
        return self.snapshot(), tuple(events)

    def all_live_threats_safely_passed(self) -> bool:
        if not self.missiles:
            return True
        for i, missile in enumerate(self.missiles):
            if not (missile.alive and missile.locked):
                continue
            distance = _distance(missile.kinematics, self.blue.kinematics)
            if not self.closest_approach_passed[i] or distance < self.config.success_distance_m or _closing_speed(missile, self.blue) > 0.0:
                return False
        return True

    def _step_physics_substep(self, command: ManeuverCommand) -> list[SimulationEvent]:
        events: list[SimulationEvent] = []
        self.substeps_last_interval += 1
        self.blue = replace(self.blue, kinematics=integrate_aircraft(self.blue.kinematics, command, self.clock.physics_dt))
        updated: list[MissileState] = []
        for i, missile in enumerate(self.missiles):
            if not missile.alive:
                updated.append(missile)
                continue
            age = missile.age_s + self.clock.physics_dt
            mcmd = proportional_navigation_command(missile.kinematics, self.blue.kinematics, self.config.navigation_constant, 0.0)
            if missile.powered and age <= self.config.missile_boost_time_s:
                mcmd = replace(mcmd, nx=self.config.missile_boost_nx_g)
            kin = integrate_missile(missile.kinematics, mcmd, self.clock.physics_dt)
            alive = missile.alive
            if kin.position.y <= 0.0:
                alive = False; events.append(SimulationEvent(self.time_s, "missile_ground_collision", f"missile_{i}", {}))
            if kin.speed < self.config.missile_stall_speed_mps:
                alive = False; events.append(SimulationEvent(self.time_s, "missile_stall", f"missile_{i}", {"speed": kin.speed}))
            if age >= self.config.max_guidance_time_s:
                alive = False; events.append(SimulationEvent(self.time_s, "missile_timeout", f"missile_{i}", {"age_s": age}))
            d = _distance(kin, self.blue.kinematics)
            if d <= self.config.kill_radius_m:
                alive = False; self.blue = replace(self.blue, alive=False); events.append(SimulationEvent(self.time_s, "hit", f"missile_{i}", {"distance_m": d}))
            prev = self.min_missile_distances[i]
            if prev < math.inf and d > prev + self.config.closest_approach_event_threshold_m:
                if not self.closest_approach_passed[i]:
                    events.append(SimulationEvent(self.time_s, "closest_approach_passed", f"missile_{i}", {"min_distance_m": prev, "distance_m": d}))
                self.closest_approach_passed[i] = True
            self.min_missile_distances[i] = min(prev, d)
            updated.append(replace(missile, kinematics=kin, alive=alive, age_s=age, powered=missile.powered and age <= self.config.missile_boost_time_s))
        self.missiles = updated
        self.time_s += self.clock.physics_dt
        if self.blue.kinematics.position.y <= 0.0:
            self.blue = replace(self.blue, alive=False)
            events.append(SimulationEvent(self.time_s, "ground_collision", "blue", {}))
        return events

    @staticmethod
    def _blue_terminal_event(events: list[SimulationEvent]) -> bool:
        return any(event.kind in {"ground_collision", "hit"} for event in events)


def _distance(a, b) -> float:
    return math.sqrt((a.position.x-b.position.x)**2 + (a.position.z-b.position.z)**2 + (a.position.y-b.position.y)**2)


def _closing_speed(missile: MissileState, blue: AircraftState) -> float:
    rel_x = blue.kinematics.position.x - missile.kinematics.position.x
    rel_z = blue.kinematics.position.z - missile.kinematics.position.z
    rel_y = blue.kinematics.position.y - missile.kinematics.position.y
    distance = max(math.sqrt(rel_x * rel_x + rel_z * rel_z + rel_y * rel_y), 1.0e-6)
    mv = flight_velocity(missile.kinematics)
    bv = flight_velocity(blue.kinematics)
    rel_vx = bv.x - mv.x
    rel_vz = bv.z - mv.z
    rel_vy = bv.y - mv.y
    return -((rel_x * rel_vx + rel_z * rel_vz + rel_y * rel_vy) / distance)
