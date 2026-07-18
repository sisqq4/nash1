import numpy as np
from air_combat_rl.core.coordinates import VecXZY
from air_combat_rl.core.timebase import SimulationClock
from air_combat_rl.domain.states import AircraftState, FlightPathAngles, KinematicState, MissileState
from air_combat_rl.simulation.scenarios.factory import ScenarioConfig, build_scenario
from air_combat_rl.simulation.world import SimulationWorld, WorldConfig
from air_combat_rl.tasks.blue_escape.action_catalog import ActionCatalog
from air_combat_rl.tasks.blue_escape.environment import BlueEscapeEnv
from air_combat_rl.tasks.blue_escape.observation_builder import ObservationBuilder, ObservationConfig
from air_combat_rl.tasks.blue_escape.rewards.components import EscapeReward, RewardConfig


def catalog(): return ActionCatalog.from_yaml("configs/actions/blue_29.yaml")
def blue(y=10000.0): return AircraftState(KinematicState(VecXZY(0,0,y),300,FlightPathAngles(0,0)), True, "zdj")
def missile(x=30000,z=0,y=10000,v=900,powered=False, psi=3.14159): return MissileState(KinematicState(VecXZY(x,z,y),v,FlightPathAngles(0,psi)), True, True, powered, 0)

def test_reset_reproducible_and_20_substeps_info():
    w1 = build_scenario(ScenarioConfig(mode="randomized_1v1", seed=7))
    w2 = build_scenario(ScenarioConfig(mode="randomized_1v1", seed=7))
    e1 = BlueEscapeEnv(w1, catalog(), "zdj"); e2 = BlueEscapeEnv(w2, catalog(), "zdj")
    assert np.allclose(e1.reset()[0], e2.reset()[0])
    r = e1.step(0)
    assert r.info["substeps"] == 20
    assert e1.held_action.remaining_substeps == 0
    assert "reward_components" in r.info and "missile_mask" in r.info and "action_mask" in r.info

def test_substep_hit_ends_early():
    world = SimulationWorld(blue=blue(), missiles=[missile(x=1, v=300)], clock=SimulationClock(), config=WorldConfig(kill_radius_m=50))
    r = BlueEscapeEnv(world, catalog(), "zdj").step(0)
    assert r.terminated and r.info["outcome"] == "hit" and r.info["substeps"] < 20

def test_y_altitude_ground_collision():
    world = SimulationWorld(blue=blue(y=0.1), clock=SimulationClock())
    r = BlueEscapeEnv(world, catalog(), "zdj").step(16)
    assert r.terminated and r.info["outcome"] == "crash"

def test_single_and_multi_missile_observation_padding_mask():
    b=blue(); builder=ObservationBuilder(ObservationConfig(m_max=3))
    obs, mask = builder.build(b, [missile()], 0)
    assert obs.shape == (8 + 3*11,) and mask.tolist() == [True, False, False]
    obs2, mask2 = builder.build(b, [missile(x=40000), missile(x=20000,z=1000)], 0)
    assert obs2.shape == obs.shape and mask2.tolist() == [True, True, False]

def test_powered_launch_and_terminal_intercept_scenarios():
    powered = build_scenario(ScenarioConfig(mode="powered_launch"))
    terminal = build_scenario(ScenarioConfig(mode="terminal_intercept"))
    assert powered.missiles[0].powered and 240 <= powered.missiles[0].kinematics.speed <= 300
    assert terminal.missiles[0].kinematics.speed == 1800.0
    assert round(terminal.missiles[0].kinematics.position.x) == 30000

def test_reward_direction_and_terminal_scale():
    snap_far = SimulationWorld(blue=blue(), missiles=[missile(x=40000)]).snapshot()
    snap_near = SimulationWorld(blue=blue(), missiles=[missile(x=20000)]).snapshot()
    rew = EscapeReward(); rew.reset(snap_far)
    r_near, comps_near = rew.compute(snap_near, "running", 0)
    rew.reset(snap_near)
    r_far, comps_far = rew.compute(snap_far, "running", 0)
    assert comps_far["separation"] > comps_near["separation"]
    assert r_far > r_near
    assert RewardConfig().terminal_success == 10.0
    assert RewardConfig().terminal_timeout == 0.0
    assert abs(comps_far["threat"]) < RewardConfig().terminal_success

def test_success_exhausted_and_timeout_conditions():
    no_threat = BlueEscapeEnv(SimulationWorld(blue=blue(), clock=SimulationClock()), catalog(), "zdj").step(0)
    assert no_threat.terminated and no_threat.info["outcome"] == "success"
    exhausted = SimulationWorld(blue=blue(), missiles=[MissileState(missile().kinematics, False, True, False, 0)], clock=SimulationClock())
    res = BlueEscapeEnv(exhausted, catalog(), "zdj").step(0)
    assert res.terminated and res.info["outcome"] == "exhausted"
    active = SimulationWorld(blue=blue(), missiles=[missile(x=40000, v=900)], clock=SimulationClock())
    timeout = BlueEscapeEnv(active, catalog(), "zdj", max_policy_steps=1).step(0)
    assert timeout.truncated and timeout.info["outcome"] == "timeout"

def test_closest_approach_event_does_not_invalidate_threat_immediately():
    world = SimulationWorld(blue=blue(), missiles=[missile(x=-10, v=900, psi=3.14159)], clock=SimulationClock(), config=WorldConfig(kill_radius_m=1.0, success_distance_m=1_000_000.0))
    res = BlueEscapeEnv(world, catalog(), "zdj").step(0)
    assert any(event.kind == "closest_approach_passed" for event in res.info["events"])
    assert world.missiles[0].alive
    assert res.info["outcome"] == "running"
