import math
from pathlib import Path

from src.air_combat_rl.core.timebase import SimulationClock
from src.air_combat_rl.domain.dynamics.aircraft_3dof import aircraft_derivative
from src.air_combat_rl.domain.states import FlightPathAngles, FlightState, VecXZY
from src.air_combat_rl.tasks.blue_escape.action_catalog import SAFE_FALLBACK_ACTION_ID, ActionCatalog
from src.air_combat_rl.tasks.blue_escape.action_hold import HeldAction

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ActionCatalog.from_yaml(str(ROOT / "configs/actions/blue_29.yaml"))
BASE_STATE = FlightState(VecXZY(0.0, 0.0, 1000.0), 200.0, FlightPathAngles(0.0, 0.0))


def derivative(action_id):
    return aircraft_derivative(BASE_STATE, CATALOG.command_for(action_id, "zdj"))


def test_action_ids_are_stable_and_complete():
    assert [CATALOG.action(i).action_id for i in range(29)] == list(range(29))
    assert CATALOG.action(0).name == "constant_speed_forward"
    assert CATALOG.action(28).name == "right_dive_9g"


def test_longitudinal_effects_match_commands():
    for action_id in range(29):
        action = CATALOG.action(action_id)
        d = derivative(action_id)
        if action.expected_longitudinal_effect == "accelerate":
            assert d.speed > 0
        elif action.expected_longitudinal_effect == "decelerate":
            assert d.speed < 0


def test_lateral_effects_match_heading_direction():
    for action_id in range(29):
        action = CATALOG.action(action_id)
        d = derivative(action_id)
        if action.expected_lateral_effect == "left":
            assert d.angles.psi < 0, action.name
        elif action.expected_lateral_effect == "right":
            assert d.angles.psi > 0, action.name


def test_vertical_effects_match_flight_path_direction():
    for action_id in range(29):
        action = CATALOG.action(action_id)
        d = derivative(action_id)
        if action.expected_vertical_effect == "climb":
            assert d.angles.gamma > 0, action.name
        elif action.expected_vertical_effect == "dive":
            assert d.angles.gamma < 0, action.name


def test_compound_actions_have_both_expected_directions():
    compound_ids = list(range(17, 29))
    for action_id in compound_ids:
        action = CATALOG.action(action_id)
        d = derivative(action_id)
        assert (d.angles.psi < 0) == (action.expected_lateral_effect == "left")
        assert (d.angles.psi > 0) == (action.expected_lateral_effect == "right")
        assert (d.angles.gamma > 0) == (action.expected_vertical_effect == "climb")
        assert (d.angles.gamma < 0) == (action.expected_vertical_effect == "dive")


def test_platform_action_masks_respect_overload_limits():
    zdj_mask = CATALOG.action_mask("zdj")
    yjj_mask = CATALOG.action_mask("yjj")
    assert all(zdj_mask)
    for action_id, allowed in enumerate(yjj_mask):
        action = CATALOG.action(action_id)
        assert allowed == (max(abs(action.nx), abs(action.nf)) <= 3.0)


def test_unknown_platform_mask_falls_back_to_safe_constant_speed():
    mask = CATALOG.action_mask("unknown")
    assert sum(mask) == 1
    assert mask[SAFE_FALLBACK_ACTION_ID]
    assert CATALOG.command_for(28, "unknown") == CATALOG.command_for(SAFE_FALLBACK_ACTION_ID, "zdj")


def test_action_hold_keeps_selected_command_for_policy_interval():
    clock = SimulationClock(physics_dt=0.005, policy_dt=0.1)
    held = HeldAction()
    command = held.select(2, "zdj", CATALOG, clock)
    assert held.remaining_substeps == 20
    for remaining in range(19, -1, -1):
        assert held.consume_substep() == command
        assert held.remaining_substeps == remaining


def test_action_hold_uses_fallback_for_masked_action():
    held = HeldAction()
    command = held.select(2, "yjj", CATALOG, SimulationClock())
    assert held.action_id == SAFE_FALLBACK_ACTION_ID
    assert math.isclose(command.nx, 0.0)
