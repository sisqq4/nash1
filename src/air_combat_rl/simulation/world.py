"""Simulation world orchestration that accepts only physical commands."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from air_combat_rl.core.timebase import SimulationClock
from air_combat_rl.domain.commands import ManeuverCommand
from air_combat_rl.domain.dynamics.aircraft_3dof import integrate_aircraft
from air_combat_rl.domain.events import SimulationEvent
from air_combat_rl.domain.states import AircraftState, MissileState
from air_combat_rl.simulation.snapshot import WorldSnapshot


@dataclass(slots=True)
class SimulationWorld:
    blue: AircraftState
    missiles: list[MissileState] = field(default_factory=list)
    clock: SimulationClock = field(default_factory=SimulationClock)
    time_s: float = 0.0

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(self.time_s, self.blue, tuple(self.missiles))

    def step_policy_interval(self, command: ManeuverCommand) -> tuple[WorldSnapshot, tuple[SimulationEvent, ...]]:
        events: list[SimulationEvent] = []
        for _ in range(self.clock.substeps_per_policy_step):
            self.blue = replace(
                self.blue,
                kinematics=integrate_aircraft(self.blue.kinematics, command, self.clock.physics_dt),
            )
            self.time_s += self.clock.physics_dt
            if self.blue.kinematics.position.y <= 0.0:
                self.blue = replace(self.blue, alive=False)
                events.append(SimulationEvent(self.time_s, "ground_collision", "blue", {}))
                break
        return self.snapshot(), tuple(events)
