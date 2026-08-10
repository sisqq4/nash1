"""Projected continuous PPO trainer with an independent rollout buffer."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from air_combat_rl.algorithms.ppo.actor_critic import PPOActorCritic
from air_combat_rl.algorithms.ppo.loss import clipped_surrogate_loss
from air_combat_rl.algorithms.ppo.rollout_buffer import RolloutBuffer

@dataclass(frozen=True, slots=True)
class PPOTrainerConfig:
    rollout_steps: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-3
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    gradient_epsilon: float = 1e-3

@dataclass(slots=True)
class PPOTrainingStats:
    global_step: int
    policy_loss: float
    value_loss: float
    entropy_loss: float
    approx_kl: float
    clip_fraction: float
    projection_distance_mean: float
    projection_distance_max: float
    fallback_count: int
    action_switch_rate: float
    executed_action_frequency: dict[int, int]
    continuous_action_mean: list[float]
    continuous_action_std: list[float]
    saturation_rate: list[float]
    exact_projection_rate: float
    action_mask_valid_count_mean: float
    explained_variance: float
    gradient_norm: float

@dataclass(slots=True)
class PPOActionLogger:
    raw_actions: list[np.ndarray] = field(default_factory=list)
    squashed_actions: list[np.ndarray] = field(default_factory=list)
    projection_distances: list[float] = field(default_factory=list)
    executed_action_ids: list[int] = field(default_factory=list)
    fallback_count: int = 0
    def add(self, raw, squashed, info):
        self.raw_actions.append(np.asarray(raw, float)); self.squashed_actions.append(np.asarray(squashed, float))
        self.projection_distances.append(float(info.get("projection_distance", 0.0)))
        self.executed_action_ids.append(int(info.get("executed_action_id", -1)))
        self.fallback_count += int(bool(info.get("fallback_used", False)))
    def summarize(self):
        raw = np.asarray(self.raw_actions) if self.raw_actions else np.zeros((0, 3)); sq = np.asarray(self.squashed_actions) if self.squashed_actions else np.zeros((0, 3))
        ids = self.executed_action_ids; freq = {i: ids.count(i) for i in sorted(set(ids)) if i >= 0}
        switches = sum(a != b for a, b in zip(ids[:-1], ids[1:])) / max(len(ids) - 1, 1) if len(ids) > 1 else 0.0
        return {
            "continuous_action_mean": raw.mean(axis=0).tolist() if len(raw) else [0.0, 0.0, 0.0],
            "continuous_action_std": raw.std(axis=0).tolist() if len(raw) else [0.0, 0.0, 0.0],
            "saturation_rate": (np.mean(np.abs(sq) > 0.99, axis=0).tolist() if len(sq) else [0.0, 0.0, 0.0]),
            "exact_projection_rate": float(np.mean(np.asarray(self.projection_distances) <= 1e-12)) if self.projection_distances else 0.0,
            "projection_distance_mean": float(np.mean(self.projection_distances)) if self.projection_distances else 0.0,
            "projection_distance_max": float(np.max(self.projection_distances)) if self.projection_distances else 0.0,
            "executed_action_frequency": freq,
            "action_switch_rate": float(switches),
            "fallback_count": self.fallback_count,
        }

class PPOProjectedTrainer:
    """Collects continuous PPO rollouts and updates actor-critic parameters independently of DQN."""
    def __init__(self, env, actor_critic: PPOActorCritic, config: PPOTrainerConfig | None = None) -> None:
        self.env = env; self.actor_critic = actor_critic; self.config = config or PPOTrainerConfig(); self.buffer = RolloutBuffer(); self.global_step = 0; self.completed_outcomes = []
    def collect_rollout(self, reset_seed: int | None = None) -> RolloutBuffer:
        self.buffer.clear(); obs, _ = self.env.reset(reset_seed); episode_start = True; logger = PPOActionLogger()
        for _ in range(self.config.rollout_steps):
            sample = self.actor_critic.act(obs)
            result = self.env.step(sample.squashed_action)
            info = result.info; logger.add(sample.raw_action, sample.squashed_action, info)
            self.buffer.add(obs, sample.raw_action, result.reward, sample.value, sample.log_prob, result.terminated, result.truncated, episode_start, projected_action=info.get("projected_continuous_action"), executed_action_id=info.get("executed_action_id", -1), projection_distance=info.get("projection_distance", 0.0), action_mask=info.get("action_mask"), next_observation=result.observation, bounded_action=sample.squashed_action, continuous_command=info.get("continuous_command"), projected_command=info.get("executed_command"))
            obs = result.observation; episode_start = result.terminated or result.truncated; self.global_step += 1
            if episode_start:
                self.completed_outcomes.append(info.get("outcome")); obs, _ = self.env.reset()
        self._last_logger = logger
        last_value = self.actor_critic.act(obs, deterministic=True).value
        self.buffer.compute_returns_and_advantages(last_value, self.config.gamma, self.config.gae_lambda)
        return self.buffer
    def _objective(self):
        obs = np.asarray(self.buffer.observations, float); actions = np.asarray(self.buffer.actions, float)
        new_lp, ent, values = self.actor_critic.evaluate_actions(obs, actions)
        return clipped_surrogate_loss(self.buffer.log_probs, new_lp, self.buffer.advantages, values, self.buffer.returns, ent, self.config.clip_range, self.config.vf_coef, self.config.ent_coef)
    def train_one_update(self) -> PPOTrainingStats:
        if not self.buffer.observations: self.collect_rollout()
        base = self._objective(); params = self.actor_critic.trainable_vectors(); grads = []
        for param in params:
            grad = np.zeros_like(param)
            it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
            for x in it:
                idx = it.multi_index; orig = float(param[idx]); eps = self.config.gradient_epsilon
                param[idx] = orig + eps; plus = self._objective().total_loss
                param[idx] = orig - eps; minus = self._objective().total_loss
                param[idx] = orig; grad[idx] = (plus - minus) / (2.0 * eps)
            grads.append(grad)
        norm = float(np.sqrt(sum(np.sum(g*g) for g in grads))); scale = min(1.0, self.config.max_grad_norm / (norm + 1e-8))
        for param, grad in zip(params, grads): param -= self.config.learning_rate * scale * grad
        post = self._objective(); summary = self._last_logger.summarize() if hasattr(self, "_last_logger") else PPOActionLogger().summarize()
        if self.config.target_kl is not None and post.approx_kl > self.config.target_kl: pass
        ev = 0.0 if self.buffer.returns is None or np.var(self.buffer.returns) < 1e-12 else float(1.0 - np.var(self.buffer.returns - np.asarray(self.buffer.values))/np.var(self.buffer.returns))
        mask_counts=[int(np.sum(m)) for m in self.buffer.action_masks if m is not None]
        return PPOTrainingStats(self.global_step, post.policy_loss, post.value_loss, -post.entropy_loss, post.approx_kl, post.clip_fraction, summary["projection_distance_mean"], summary["projection_distance_max"], summary["fallback_count"], summary["action_switch_rate"], summary["executed_action_frequency"], summary["continuous_action_mean"], summary["continuous_action_std"], summary["saturation_rate"], summary["exact_projection_rate"], float(np.mean(mask_counts)) if mask_counts else 0.0, ev, norm)
