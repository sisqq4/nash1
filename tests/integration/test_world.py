from air_combat_rl.core.coordinates import VecXZY
from air_combat_rl.core.timebase import SimulationClock
from air_combat_rl.domain.commands import ManeuverCommand
from air_combat_rl.domain.states import AircraftState, FlightPathAngles, KinematicState
from air_combat_rl.simulation.world import SimulationWorld


def test_world_steps_one_policy_interval_with_physical_command():
    blue = AircraftState(KinematicState(VecXZY(0.0, 0.0, 1000.0), 200.0, FlightPathAngles(0.0, 0.0)), True, "zdj")
    world = SimulationWorld(blue=blue, clock=SimulationClock())
    snapshot, events = world.step_policy_interval(ManeuverCommand(0.0, 0.0, 0.0))
    assert round(snapshot.time_s, 3) == 0.1
    assert snapshot.blue.kinematics.position.x > 0.0
    assert events == ()
