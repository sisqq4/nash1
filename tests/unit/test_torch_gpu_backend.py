# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")

from src.air_combat_rl.algorithms.ppo.torch_actor_critic import (
    TorchPPOActorCritic,
)  # noqa: E402
from src.air_combat_rl.algorithms.ppo.vector_rollout_buffer import (
    VectorRolloutBuffer,
)  # noqa: E402
from src.air_combat_rl.evaluation.parallel_evaluator import (
    run_parallel_torch_evaluation,
)  # noqa: E402
from src.air_combat_rl.training.torch_runner import run_torch_training  # noqa: E402


def test_torch_actor_is_batched_and_all_parameters_receive_gradients():
    model = TorchPPOActorCritic(5, hidden_sizes=(16, 16))
    observations = torch.zeros((4, 5))
    sample = model.act(observations)
    assert sample.raw_action.shape == (4, 3)
    assert sample.value.shape == (4,)

    log_prob, entropy, value = model.evaluate_actions(observations, sample.raw_action)
    loss = -(log_prob.mean() + 0.01 * entropy.mean()) + value.square().mean()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_vector_gae_does_not_leak_across_episode_boundaries():
    buffer = VectorRolloutBuffer(2, 2, 3)
    buffer.rewards[:] = [[1, 1], [1, 1]]
    buffer.values[:] = 0
    buffer.next_values[:] = 0
    buffer.terminated[0, 0] = True
    buffer.compute_gae(gamma=1, gae_lambda=1)
    assert buffer.advantages[0, 0] == pytest.approx(1)
    assert buffer.advantages[0, 1] == pytest.approx(2)


def test_torch_training_checkpoint_resume_and_parallel_evaluation(tmp_path):
    runtime_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = {
        "algorithm": {"name": "ppo_projected", "backend": "torch"},
        "device": {
            "type": runtime_device,
            "amp": torch.cuda.is_available(),
            "deterministic": True,
        },
        "vector_env": {"num_envs": 2, "backend": "serial"},
        "model": {"hidden_sizes": [8, 8]},
        "ppo": {
            "rollout_steps": 2,
            "update_epochs": 1,
            "minibatch_size": 4,
            "learning_rate": 0.001,
        },
        "projection": {
            "metric": "weighted_normalized_command_distance",
            "weights": {"nx": 1.0, "nf": 1.0, "gamma_s": 1.0},
            "ranges": {"nx": 18.0, "nf": 9.0, "gamma_s": 3.141592653589793},
        },
    }
    run_dir = tmp_path / "train"
    result = run_torch_training(
        algorithm_config=config,
        actions="configs/actions/blue_29.yaml",
        platform="zdj",
        seed=0,
        output_dir=run_dir,
        total_steps=4,
        checkpoint_interval=4,
        device=runtime_device,
        num_envs=2,
        env_backend="serial",
        scenario="configs/scenario/fixed_1v1.yaml",
        max_policy_steps=1,
        amp=torch.cuda.is_available(),
    )
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    assert result["global_step"] == 4
    assert checkpoint.is_file()

    resumed = run_torch_training(
        algorithm_config=config,
        actions="configs/actions/blue_29.yaml",
        platform="zdj",
        seed=0,
        output_dir=run_dir,
        total_steps=8,
        checkpoint_interval=4,
        device=runtime_device,
        num_envs=2,
        env_backend="serial",
        scenario="configs/scenario/fixed_1v1.yaml",
        max_policy_steps=1,
        resume=checkpoint,
        amp=torch.cuda.is_available(),
    )
    assert resumed["global_step"] == 8

    algorithm_path = tmp_path / "algorithm.yaml"
    import yaml

    algorithm_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    evaluation_dir = tmp_path / "evaluation"
    evaluated = run_parallel_torch_evaluation(
        scenarios=["configs/scenario/fixed_1v1.yaml"],
        actions="configs/actions/blue_29.yaml",
        algorithm_config_path=algorithm_path,
        checkpoint=checkpoint,
        platform="zdj",
        episodes=2,
        seeds=[0],
        deterministic=True,
        output_dir=evaluation_dir,
        max_policy_steps=1,
        device=runtime_device,
        num_envs=2,
        env_backend="serial",
    )
    assert evaluated["episodes"] == 2
    assert (evaluation_dir / "metrics.json").is_file()
    assert (evaluation_dir / "report.md").is_file()
