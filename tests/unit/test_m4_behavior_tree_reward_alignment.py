from air_combat_rl.core.coordinates import VecXZY
from air_combat_rl.domain.states import AircraftState, FlightPathAngles, KinematicState, MissileState
from air_combat_rl.simulation.snapshot import WorldSnapshot
from air_combat_rl.controllers.baselines.blue_behavior_tree import (
    A_ACCEL,
    A_CLIMB,
    A_DIVE,
    A_LEFT_DIVE,
    A_LEFT_TURN,
    A_RIGHT_DIVE,
    A_RIGHT_TURN,
    BlueBehaviorTreeConfig,
    BlueBehaviorTreePolicy,
)
from air_combat_rl.tasks.blue_escape.rewards.components import EscapeReward, RewardConfig


def blue(y=10000.0, v=300.0, gamma=0.0, psi=0.0):
    return AircraftState(KinematicState(VecXZY(0, 0, y), v, FlightPathAngles(gamma, psi)), True, "zdj")


def missile(x=30000.0, z=0.0, y=10000.0, v=900.0):
    return MissileState(KinematicState(VecXZY(x, z, y), v, FlightPathAngles(0.0, 3.14159)), True, True, False, 0.0)


def snapshot(b, missiles=()):
    return WorldSnapshot(0.0, b, tuple(missiles))


def test_behavior_tree_prioritizes_low_altitude_then_low_speed():
    policy = BlueBehaviorTreePolicy(BlueBehaviorTreeConfig(seed=1))
    assert policy.select_action(snapshot(blue(y=500.0, v=300.0))) == A_CLIMB
    assert policy.select_action(snapshot(blue(y=5000.0, v=100.0))) == A_DIVE
    assert policy.select_action(snapshot(blue(y=1500.0, v=100.0))) == A_ACCEL


def test_behavior_tree_beam_and_break_states_from_threat_range():
    policy = BlueBehaviorTreePolicy(BlueBehaviorTreeConfig(seed=2, beam_dwell_steps=1, break_dwell_steps=1))
    beam_action = policy.select_action(snapshot(blue(), [missile(x=15000.0, z=0.0)]))
    assert beam_action in {A_LEFT_TURN, A_RIGHT_TURN, A_ACCEL}
    assert policy.state == "beam"
    break_action = policy.select_action(snapshot(blue(), [missile(x=5000.0, z=0.0)]))
    assert break_action in {A_LEFT_DIVE, A_RIGHT_DIVE}
    assert policy.state == "break"


def test_reward_terminal_values_match_rule_document_and_nonterminal_has_components():
    reward = EscapeReward(RewardConfig())
    snap = snapshot(blue(), [missile(x=20000.0)])
    reward.reset(snap)
    hit, hit_components = reward.compute(snap, "hit", 0)
    assert hit == -10.0 and hit_components["terminal"] == -10.0
    reward.reset(snap)
    crash, crash_components = reward.compute(snap, "crash", 0)
    assert crash == -20.0 and crash_components["terminal"] == -20.0
    reward.reset(snap)
    exhausted, exhausted_components = reward.compute(snap, "exhausted", 0)
    assert exhausted == 10.0 and exhausted_components["terminal"] == 10.0
    reward.reset(snap)
    running, running_components = reward.compute(snap, "running", 0)
    assert running_components["terminal"] == 0.0
    assert set(running_components) == {"terminal", "separation", "threat", "height", "encirclement", "ground", "smooth"}


def test_behavior_tree_act_uses_rl_observation_and_action_mask_shape():
    from air_combat_rl.tasks.blue_escape.observation_builder import ObservationBuilder

    policy = BlueBehaviorTreePolicy(BlueBehaviorTreeConfig(seed=3))
    obs, _ = ObservationBuilder().build(blue(y=500.0, v=300.0), [missile(x=15000.0)], 0)
    action_mask = [True] * 29
    assert policy.act(obs, action_mask) == A_CLIMB
    action_mask[A_CLIMB] = False
    assert policy.act(obs, action_mask) == 0
