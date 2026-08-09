"""Batched PyTorch actor-critic for projected continuous PPO."""

from __future__ import annotations

from dataclasses import dataclass

from air_combat_rl.algorithms.common.device import require_torch

torch = require_torch()
nn = torch.nn
Normal = torch.distributions.Normal


@dataclass(slots=True)
class TorchGaussianSample:
    raw_action: object
    squashed_action: object
    log_prob: object
    entropy: object
    value: object
    mean: object
    std: object


def _mlp(input_dim: int, hidden_sizes, output_dim: int):
    layers = []
    previous = input_dim
    for hidden in hidden_sizes:
        if int(hidden) <= 0:
            raise ValueError("hidden layer sizes must be positive")
        layers.extend([nn.Linear(previous, int(hidden)), nn.Tanh()])
        previous = int(hidden)
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class TorchPPOActorCritic(nn.Module):
    """Independent Gaussian actor and scalar critic supporting batch inputs."""

    def __init__(
        self, obs_dim: int, action_dim: int = 3, hidden_sizes=(256, 256)
    ) -> None:
        super().__init__()
        if action_dim != 3:
            raise ValueError("projected PPO action_dim must be 3")
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = _mlp(obs_dim, hidden_sizes, action_dim)
        self.critic = _mlp(obs_dim, hidden_sizes, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, observations):
        mean = self.actor(observations)
        log_std = self.log_std.clamp(-20, 2).expand_as(mean)
        value = self.critic(observations).squeeze(-1)
        return mean, log_std, value

    @staticmethod
    def log_prob_from_raw(raw_action, mean, log_std):
        distribution = Normal(mean, log_std.exp())
        gaussian_log_prob = distribution.log_prob(raw_action).sum(-1)
        # Stable form of log(1 - tanh(x)^2), used by SAC-style squashed Gaussians.
        correction = 2 * (
            torch.log(
                torch.tensor(2.0, dtype=raw_action.dtype, device=raw_action.device)
            )
            - raw_action
            - torch.nn.functional.softplus(-2 * raw_action)
        )
        return gaussian_log_prob - correction.sum(-1)

    def act(self, observations, deterministic: bool = False) -> TorchGaussianSample:
        mean, log_std, value = self(observations)
        distribution = Normal(mean, log_std.exp())
        raw_action = mean if deterministic else distribution.rsample()
        return TorchGaussianSample(
            raw_action,
            torch.tanh(raw_action),
            self.log_prob_from_raw(raw_action, mean, log_std),
            distribution.entropy().sum(-1),
            value,
            mean,
            log_std.exp(),
        )

    def evaluate_actions(self, observations, raw_actions):
        mean, log_std, value = self(observations)
        distribution = Normal(mean, log_std.exp())
        return (
            self.log_prob_from_raw(raw_actions, mean, log_std),
            distribution.entropy().sum(-1),
            value,
        )
