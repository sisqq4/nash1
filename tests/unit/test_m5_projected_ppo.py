import math
import numpy as np

from src.air_combat_rl.algorithms.ppo.actor_critic import PPOActorCritic
from src.air_combat_rl.algorithms.ppo.loss import clipped_surrogate_loss
from src.air_combat_rl.algorithms.ppo.rollout_buffer import RolloutBuffer
from src.air_combat_rl.algorithms.rainbow.policy import RainbowDQNPolicyAdapter
from src.air_combat_rl.domain.commands import ManeuverCommand
from src.air_combat_rl.domain.states import FlightState, FlightPathAngles
from src.air_combat_rl.core.coordinates import VecXZY
from src.air_combat_rl.tasks.blue_escape.action_catalog import ActionCatalog
from src.air_combat_rl.tasks.blue_escape.continuous import ContinuousCommandProjector, NearestManeuverMapper


def catalog(): return ActionCatalog.from_yaml("configs/actions/blue_29.yaml")
def state(speed=300.0, y=5000.0, gamma=0.0): return FlightState(VecXZY(0,0,y), speed, FlightPathAngles(gamma,0.0))


def test_ppo_actor_outputs_three_dimensional_tanh_corrected_distribution():
    ac = PPOActorCritic(obs_dim=5, seed=1)
    sample = ac.act(np.zeros(5))
    assert sample.raw_action.shape == (3,)
    assert sample.squashed_action.shape == (3,)
    assert np.all(sample.squashed_action <= 1.0) and np.all(sample.squashed_action >= -1.0)
    naive = -0.5 * np.sum(sample.raw_action**2 + math.log(2 * math.pi))
    assert sample.log_prob != naive


def test_projector_platform_speed_and_altitude_constraints():
    p = ContinuousCommandProjector()
    assert p.project(ManeuverCommand(9,9,0), state(), "zdj").nf == 9
    assert p.project(ManeuverCommand(9,9,0), state(), "yjj").nf == 3
    low_speed = p.project(ManeuverCommand(-3,9,0), state(speed=100), "zdj")
    assert low_speed.nx >= 0 and low_speed.nf <= 4.5
    low_alt = p.project(ManeuverCommand(0,1,math.pi), state(y=100), "zdj")
    assert low_alt.gamma_s == 0.0 and low_alt.nf >= 1.5


def test_mapper_uses_effect_distance_tie_break_and_fallback():
    actions = catalog(); mapper = NearestManeuverMapper(actions); s = state()
    eff = mapper.effect(s, ManeuverCommand(0,1,0))
    assert abs(eff.dV_dt) < 1e-9 and abs(eff.dgamma_dt) < 1e-9 and abs(eff.dpsi_dt) < 1e-9
    mapped = mapper.map(s, ManeuverCommand(0,1,0), "zdj", [0, 1])
    assert mapped.action_id == 0
    tied = mapper.map(s, actions.action(1).command, "zdj", [2, 1])
    assert tied.action_id == 1
    fb = mapper.map(s, ManeuverCommand(1,1,0), "zdj", [])
    assert fb.fallback_used and fb.action_id == 0


def test_rollout_tracks_continuous_and_discrete_actions_and_gae_loss():
    buf = RolloutBuffer()
    buf.add(np.zeros(2), np.zeros(3), 1.0, 0.2, -0.1, False, False, True, projected_action=ManeuverCommand(0,1,0), executed_action_id=7, projection_distance=0.3)
    adv, ret = buf.compute_returns_and_advantages(last_value=0.0)
    assert buf.executed_action_ids == [7] and buf.projection_distances == [0.3]
    assert adv.shape == (1,) and ret.shape == (1,)
    stats = clipped_surrogate_loss([-0.1], [-0.2], [1.0], [0.1], [1.0], [0.5])
    assert stats.clip_fraction == 0.0 and stats.approx_kl > 0.0


def test_rainbow_adapter_keeps_discrete_execution_path():
    class Q:
        def predict_q(self, obs):
            q = np.zeros(29); q[3] = 5; return q
    d = RainbowDQNPolicyAdapter(catalog(), "zdj", Q()).act(np.zeros(4))
    assert d.algorithm_name == "rainbow_dqn"
    assert d.executed_action_id == 3
    assert d.bounded_action is None and d.projected_action is None

from src.air_combat_rl.core.timebase import SimulationClock
from src.air_combat_rl.domain.states import AircraftState, KinematicState, MissileState
from src.air_combat_rl.simulation.world import SimulationWorld
from src.air_combat_rl.tasks.blue_escape.environment import BlueEscapeEnv
from src.air_combat_rl.tasks.blue_escape.continuous import ProjectedContinuousActionWrapper
from src.air_combat_rl.algorithms.ppo.trainer import PPOProjectedTrainer, PPOTrainerConfig
from src.air_combat_rl.algorithms.rainbow.trainer import RainbowDQNTrainer, RainbowTrainerConfig
from src.air_combat_rl.algorithms.rainbow.network import RainbowQNetwork
from src.air_combat_rl.algorithms.rainbow.replay import PrioritizedReplayBuffer
from src.air_combat_rl.training.runner import build_algorithm_runtime


def make_env(platform="zdj"):
    b = AircraftState(KinematicState(VecXZY(0,0,10000),300,FlightPathAngles(0,0)), True, platform)
    m = MissileState(KinematicState(VecXZY(30000,0,10000),900,FlightPathAngles(0,math.pi)), True, True, False, 0)
    return BlueEscapeEnv(SimulationWorld(blue=b, missiles=[m], clock=SimulationClock()), catalog(), platform, max_policy_steps=3)


def test_projected_wrapper_calls_discrete_env_and_logs_mapping():
    env = make_env(); wrapped = ProjectedContinuousActionWrapper(env)
    res = wrapped.step(np.array([0.2, 0.3, -0.1]))
    assert isinstance(res.info["executed_action_id"], int)
    assert "raw_continuous_action" in res.info and "projection_distance" in res.info
    assert res.info["valid_action_count"] > 0


def test_high_speed_projection_limits_acceleration():
    p = ContinuousCommandProjector()
    cmd = p.project(ManeuverCommand(9, 2, 0), state(speed=800), "zdj")
    assert cmd.nx <= -0.5
    cmd2 = p.project(ManeuverCommand(9, 2, 0), state(speed=700), "zdj")
    assert cmd2.nx <= 0.0


def test_ppo_trainer_uses_continuous_rollout_buffer_and_updates():
    env = ProjectedContinuousActionWrapper(make_env())
    ac = PPOActorCritic(obs_dim=len(env.reset()[0]), seed=4)
    trainer = PPOProjectedTrainer(env, ac, PPOTrainerConfig(rollout_steps=4, learning_rate=1e-3))
    buf = trainer.collect_rollout()
    assert np.asarray(buf.actions).shape == (4, 3)
    before = ac.mean_b.copy(); stats = trainer.train_one_update()
    assert trainer.global_step == 4 and stats.executed_action_frequency
    assert not np.allclose(before, ac.mean_b)


def test_rainbow_trainer_has_independent_prioritized_replay_and_updates():
    env = make_env(); obs = env.reset()[0]
    q = RainbowQNetwork(len(obs), seed=5)
    trainer = RainbowDQNTrainer(env, q, config=RainbowTrainerConfig(batch_size=1, learning_rate=1e-3), seed=5)
    res = trainer.collect_step(obs)
    assert isinstance(trainer.replay_buffer, PrioritizedReplayBuffer)
    before = q.bias.copy(); stats = trainer.train_one_update()
    assert stats.replay_size == 1 and not np.allclose(before, q.bias)


def test_config_selects_independent_trainers_and_no_shared_buffer():
    ppo_rt = build_algorithm_runtime({"algorithm": {"name": "ppo_projected"}, "ppo": {"rollout_steps": 2}, "seed": 1}, make_env())
    dqn_rt = build_algorithm_runtime({"algorithm": {"name": "rainbow_dqn"}, "rainbow": {"batch_size": 1}, "seed": 1}, make_env())
    assert ppo_rt.name == "ppo_projected" and dqn_rt.name == "rainbow_dqn"
    assert isinstance(ppo_rt.trainer.buffer, RolloutBuffer)
    assert isinstance(dqn_rt.trainer.replay_buffer, PrioritizedReplayBuffer)
    assert ppo_rt.trainer.buffer is not dqn_rt.trainer.replay_buffer

import pickle
from src.air_combat_rl.algorithms.rainbow.checkpoint import load_rainbow_checkpoint
from src.air_combat_rl.algorithms.ppo.checkpoint import load_ppo_checkpoint, save_ppo_checkpoint


def test_rainbow_legacy_checkpoint_loads_and_ppo_rejects_it(tmp_path):
    path = tmp_path / "legacy_rainbow.pkl"
    with open(path, "wb") as f:
        pickle.dump({"weights": [1, 2, 3]}, f)
    assert load_rainbow_checkpoint(path)["weights"] == [1, 2, 3]
    try:
        load_ppo_checkpoint(path)
    except ValueError:
        pass
    else:
        raise AssertionError("PPO loader must reject legacy Rainbow checkpoint")


def test_ppo_checkpoint_saves_required_metadata(tmp_path):
    ac = PPOActorCritic(obs_dim=3, seed=8)
    path = tmp_path / "ppo.pkl"
    save_ppo_checkpoint(path, ac, optimizer_state={"lr": 1e-3}, obs_normalization={"mean": 0}, global_step=7, curriculum_stage="m5", config={"algorithm": {"name": "ppo_projected"}})
    payload = load_ppo_checkpoint(path)
    assert payload["metadata"]["algorithm_name"] == "ppo_projected"
    assert payload["metadata"]["action_dimension"] == 3
    assert payload["global_step"] == 7 and payload["config"]["algorithm"]["name"] == "ppo_projected"
