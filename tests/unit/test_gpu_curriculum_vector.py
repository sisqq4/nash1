import numpy as np
import yaml

from air_combat_rl.algorithms.ppo.vector_rollout_buffer import VectorRolloutBuffer
from air_combat_rl.training.curriculum import CurriculumConfig, CurriculumScheduler
from air_combat_rl.vector import EnvSpec, SerialVectorEnv, SubprocessVectorEnv


def _spec(seed, *, scenario="configs/scenario/fixed_1v1.yaml", max_policy_steps=1):
    return EnvSpec(
        scenario,
        "configs/actions/blue_29.yaml",
        "zdj",
        seed,
        max_policy_steps=max_policy_steps,
    )


def test_curriculum_weighted_schedule_advances_and_restores(tmp_path):
    path = tmp_path / "curriculum.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "test",
                "metric_window_episodes": 2,
                "stages": [
                    {
                        "name": "easy",
                        "minimum_episodes": 2,
                        "scenarios": [
                            {"path": "configs/scenario/fixed_1v1.yaml", "weight": 1}
                        ],
                        "advancement": {
                            "metric": "success_rate",
                            "threshold": 1.0,
                            "consecutive_windows": 1,
                        },
                    },
                    {
                        "name": "hard",
                        "scenarios": [
                            {
                                "path": "configs/scenario/high_threat_1v1.yaml",
                                "weight": 1,
                            }
                        ],
                    },
                ],
            }
        )
    )
    config = CurriculumConfig.from_yaml(path)
    scheduler = CurriculumScheduler(
        config,
        actions_path="configs/actions/blue_29.yaml",
        platform="zdj",
        base_seed=3,
        max_policy_steps=1,
    )
    first = scheduler.next_spec(0)
    scheduler.record_episode("success", 1)
    scheduler.record_episode("success", 1)
    event = scheduler.maybe_advance(20)
    assert first.stage_name == "easy"
    assert event["to_stage"] == "hard"

    state = scheduler.state_dict()
    restored = CurriculumScheduler(
        config,
        actions_path="configs/actions/blue_29.yaml",
        platform="zdj",
        base_seed=3,
        max_policy_steps=1,
    )
    restored.load_state_dict(state)
    assert restored.stage.name == "hard"
    assert restored.next_spec(1).stage_name == "hard"


def test_curriculum_success_metric_does_not_count_exhaustion_as_success(tmp_path):
    path = tmp_path / "curriculum.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "test",
                "metric_window_episodes": 1,
                "stages": [
                    {
                        "name": "easy",
                        "scenarios": [
                            {"path": "configs/scenario/fixed_1v1.yaml", "weight": 1}
                        ],
                    }
                ],
            }
        )
    )
    scheduler = CurriculumScheduler(
        CurriculumConfig.from_yaml(path),
        actions_path="configs/actions/blue_29.yaml",
        platform="zdj",
        base_seed=0,
    )
    scheduler.record_episode("exhausted")
    assert scheduler.metrics()["success_rate"] == 0
    assert scheduler.metrics()["escape_completion_rate"] == 1


def test_serial_vector_env_batches_and_resets_one_slot():
    specs = [
        _spec(index, scenario="configs/scenario/randomized_1v1.yaml")
        for index in range(2)
    ]
    env = SerialVectorEnv(specs)
    try:
        observations, _ = env.reset()
        result = env.step(np.zeros((2, 3), np.float32))
        assert observations.shape[0] == 2
        assert result.observations.shape == observations.shape
        assert result.truncated.tolist() == [True, True]

        replacement = _spec(99, scenario="configs/scenario/high_threat_1v1.yaml")
        reset_observation, _ = env.reset_at(0, replacement)
        assert reset_observation.shape == observations[0].shape
        assert env.specs[0].seed == 99
    finally:
        env.close()


def test_randomized_environment_reset_seed_rebuilds_world():
    spec = EnvSpec(
        "configs/scenario/randomized_1v1.yaml",
        "configs/actions/blue_29.yaml",
        "zdj",
        1,
        max_policy_steps=1,
        projected_continuous=False,
    )
    env = SerialVectorEnv([spec])
    try:
        first, _ = env.envs[0].reset(1)
        same, _ = env.envs[0].reset(1)
        different, _ = env.envs[0].reset(2)
        assert np.array_equal(first, same)
        assert not np.array_equal(first, different)
    finally:
        env.close()


def test_spawn_vector_env_batches_worker_results_and_honors_active_mask():
    env = SubprocessVectorEnv([_spec(0), _spec(1)], start_method="spawn")
    try:
        observations, _ = env.reset()
        result = env.step(np.zeros((2, 3), np.float32), active_mask=[True, False])
        assert result.observations.shape == observations.shape
        assert result.truncated.tolist() == [True, False]
        assert result.rewards[1] == 0
        assert np.array_equal(result.observations[1], observations[1])
    finally:
        env.close()


def test_vector_env_active_mask_does_not_step_finished_slot():
    env = SerialVectorEnv([_spec(0, max_policy_steps=2), _spec(1, max_policy_steps=2)])
    try:
        observations, _ = env.reset()
        first = env.step(np.zeros((2, 3), np.float32), active_mask=[True, False])
        assert first.rewards[1] == 0
        assert first.infos[1]["initial_missile_count"] == 1
        assert np.array_equal(first.observations[1], observations[1])
    finally:
        env.close()


def test_vector_gae_bootstraps_timeout_without_cross_episode_leakage():
    buffer = VectorRolloutBuffer(2, 2, 3)
    buffer.rewards[:] = [[1, 1], [1, 1]]
    buffer.values[:] = 0
    buffer.next_values[:] = 0
    buffer.terminated[0, 0] = True
    buffer.truncated[0, 1] = True
    buffer.next_values[0, 1] = 2
    buffer.compute_gae(gamma=1, gae_lambda=1, bootstrap_truncated=True)
    assert buffer.advantages[0, 0] == 1
    assert buffer.advantages[0, 1] == 3


def test_vector_environment_applies_projection_configuration():
    projection = {
        "metric": "weighted_normalized_command_distance",
        "weights": {"nx": 2.0, "nf": 3.0, "gamma_s": 4.0},
        "ranges": {"nx": 12.0, "nf": 6.0, "gamma_s": 2.0},
    }
    spec = EnvSpec(
        "configs/scenario/fixed_1v1.yaml",
        "configs/actions/blue_29.yaml",
        "zdj",
        0,
        projection_config=projection,
    )
    env = SerialVectorEnv([spec])
    try:
        config = env.envs[0].mapper.config
        assert (config.w_nx, config.w_nf, config.w_gamma_s) == (2.0, 3.0, 4.0)
        assert (config.range_nx, config.range_nf, config.range_gamma_s) == (
            12.0,
            6.0,
            2.0,
        )
    finally:
        env.close()
