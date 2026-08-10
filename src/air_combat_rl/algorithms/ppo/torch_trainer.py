"""GPU learner with vectorized CPU environment collection."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time

import numpy as np

from src.air_combat_rl.algorithms.common.device import require_torch
from src.air_combat_rl.algorithms.ppo.vector_rollout_buffer import VectorRolloutBuffer

torch = require_torch()


@dataclass(frozen=True, slots=True)
class TorchPPOConfig:
    rollout_steps: int = 256
    update_epochs: int = 10
    minibatch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    bootstrap_truncated: bool = True

    def __post_init__(self) -> None:
        if (
            self.rollout_steps <= 0
            or self.update_epochs <= 0
            or self.minibatch_size <= 0
        ):
            raise ValueError(
                "rollout_steps, update_epochs, and minibatch_size must be positive"
            )
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")


class TorchPPOTrainer:
    """Collect vector rollouts on CPUs and optimize one shared GPU model."""

    def __init__(
        self, vector_env, model, device, config, scheduler=None, amp=False
    ) -> None:
        self.env = vector_env
        self.model = model
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.amp = bool(amp and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.global_step = 0
        self.update_index = 0
        self.observations = None
        self.infos = None
        self.episode_returns = np.zeros(vector_env.num_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(vector_env.num_envs, dtype=np.int64)
        self.completed_episodes: list[dict] = []
        self.curriculum_events: list[dict] = []
        self.action_catalog_version = None

    def _tensor(self, array):
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def initialize(self) -> None:
        if self.observations is None:
            self.observations, self.infos = self.env.reset()

    def collect_rollout(self, rollout_steps: int | None = None):
        self.initialize()
        num_envs = self.env.num_envs
        observation_dim = self.observations.shape[1]
        rollout_steps = (
            self.config.rollout_steps if rollout_steps is None else int(rollout_steps)
        )
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        buffer = VectorRolloutBuffer(
            rollout_steps,
            num_envs,
            observation_dim,
        )
        start = time.perf_counter()

        for time_index in range(rollout_steps):
            buffer.observations[time_index] = self.observations
            with torch.inference_mode():
                sample = self.model.act(self._tensor(self.observations))
            actions = sample.squashed_action.cpu().numpy()
            buffer.raw_actions[time_index] = sample.raw_action.cpu().numpy()
            buffer.bounded_actions[time_index] = actions
            buffer.log_probs[time_index] = sample.log_prob.cpu().numpy()
            buffer.values[time_index] = sample.value.cpu().numpy()

            result = self.env.step(actions)
            final_observations = result.observations.copy()
            with torch.inference_mode():
                buffer.next_values[time_index] = (
                    self.model(self._tensor(final_observations))[2].cpu().numpy()
                )
            buffer.rewards[time_index] = result.rewards
            buffer.terminated[time_index] = result.terminated
            buffer.truncated[time_index] = result.truncated
            buffer.executed_action_ids[time_index] = [
                int(info.get("executed_action_id", -1)) for info in result.infos
            ]
            buffer.projection_distances[time_index] = [
                float(info.get("projection_distance", 0)) for info in result.infos
            ]

            self.episode_returns += result.rewards
            self.episode_lengths += 1
            next_observations = result.observations.copy()
            next_infos = list(result.infos)
            done_indices = np.flatnonzero(result.terminated | result.truncated)

            completed_now = []
            for env_index in done_indices:
                info = result.infos[env_index]
                record = {
                    "episode_id": self.env.specs[env_index].episode_id,
                    "scenario": self.env.specs[env_index].scenario_path,
                    "stage": self.env.specs[env_index].stage_name,
                    "seed": self.env.specs[env_index].seed,
                    "reward": float(self.episode_returns[env_index]),
                    "length": int(self.episode_lengths[env_index]),
                    "outcome": str(info.get("outcome", "running")),
                }
                completed_now.append(record)
                self.completed_episodes.append(record)

            if self.scheduler and completed_now:
                for record in completed_now:
                    self.scheduler.record_episode(record["outcome"], record["reward"])
                event = self.scheduler.maybe_advance(
                    self.global_step + (time_index + 1) * num_envs
                )
                if event:
                    self.curriculum_events.append(event)

            for env_index in done_indices:
                if self.scheduler:
                    spec = self.scheduler.next_spec(int(env_index))
                else:
                    old = self.env.specs[env_index]
                    spec = replace(
                        old,
                        seed=old.seed + 1_000_003,
                        episode_id=old.episode_id + num_envs,
                    )
                reset_observation, reset_info = self.env.reset_at(int(env_index), spec)
                next_observations[env_index] = reset_observation
                next_infos[env_index] = reset_info
                self.episode_returns[env_index] = 0
                self.episode_lengths[env_index] = 0

            self.observations = next_observations
            self.infos = next_infos

        self.global_step += rollout_steps * num_envs
        buffer.compute_gae(
            self.config.gamma,
            self.config.gae_lambda,
            self.config.bootstrap_truncated,
        )
        return buffer, {"rollout_seconds": time.perf_counter() - start}

    def train_one_update(self, buffer):
        start = time.perf_counter()
        batch_size = buffer.steps * buffer.num_envs
        indices = np.arange(batch_size)
        observations = buffer.observations.reshape(batch_size, -1)
        raw_actions = buffer.raw_actions.reshape(batch_size, -1)
        old_log_probs = buffer.log_probs.reshape(-1)
        advantages = buffer.advantages.reshape(-1)
        returns = buffer.returns.reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        metric_rows = []
        stop_early = False

        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for begin in range(0, batch_size, self.config.minibatch_size):
                minibatch_indices = indices[begin : begin + self.config.minibatch_size]
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp,
                ):
                    new_log_probs, entropy, values = self.model.evaluate_actions(
                        self._tensor(observations[minibatch_indices]),
                        self._tensor(raw_actions[minibatch_indices]),
                    )
                    old = self._tensor(old_log_probs[minibatch_indices])
                    minibatch_advantages = self._tensor(advantages[minibatch_indices])
                    minibatch_returns = self._tensor(returns[minibatch_indices])
                    log_ratio = new_log_probs - old
                    ratio = log_ratio.exp()
                    policy_loss = -torch.minimum(
                        ratio * minibatch_advantages,
                        torch.clamp(
                            ratio,
                            1 - self.config.clip_range,
                            1 + self.config.clip_range,
                        )
                        * minibatch_advantages,
                    ).mean()
                    value_loss = torch.nn.functional.mse_loss(values, minibatch_returns)
                    loss = (
                        policy_loss
                        + self.config.vf_coef * value_loss
                        - self.config.ent_coef * entropy.mean()
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite PPO loss")

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                approximate_kl = float(((ratio - 1) - log_ratio).mean().detach().cpu())
                clip_fraction = float(
                    ((ratio - 1).abs() > self.config.clip_range)
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                )
                metric_rows.append(
                    (
                        float(policy_loss.detach().cpu()),
                        float(value_loss.detach().cpu()),
                        float(entropy.mean().detach().cpu()),
                        approximate_kl,
                        clip_fraction,
                        float(gradient_norm.detach().cpu()),
                    )
                )
                if (
                    self.config.target_kl is not None
                    and approximate_kl > self.config.target_kl
                ):
                    stop_early = True
                    break
            if stop_early:
                break

        self.update_index += 1
        metric_array = np.asarray(metric_rows, dtype=float)
        action_ids, action_counts = np.unique(
            buffer.executed_action_ids, return_counts=True
        )
        return {
            "global_step": self.global_step,
            "update_index": self.update_index,
            "policy_loss": float(metric_array[:, 0].mean()),
            "value_loss": float(metric_array[:, 1].mean()),
            "entropy": float(metric_array[:, 2].mean()),
            "approx_kl": float(metric_array[:, 3].mean()),
            "clip_fraction": float(metric_array[:, 4].mean()),
            "gradient_norm": float(metric_array[:, 5].mean()),
            "projection_distance_mean": float(buffer.projection_distances.mean()),
            "projection_distance_max": float(buffer.projection_distances.max()),
            "executed_action_frequency": {
                str(action_id): int(count)
                for action_id, count in zip(action_ids, action_counts)
            },
            "optimization_seconds": time.perf_counter() - start,
            "batch_size": batch_size,
            "curriculum_stage": (
                self.scheduler.stage.name if self.scheduler else "single"
            ),
        }
