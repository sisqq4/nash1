"""Synchronous vector environment used for debugging and deterministic tests."""

from __future__ import annotations

import numpy as np

from src.air_combat_rl.vector.env_factory import make_env
from src.air_combat_rl.vector.types import EnvSpec, VectorStepResult


class SerialVectorEnv:
    """Run multiple environments in one process behind a batched interface."""

    def __init__(self, specs: list[EnvSpec]) -> None:
        if not specs:
            raise ValueError("at least one environment is required")
        self.specs = list(specs)
        self.envs = [make_env(spec) for spec in specs]
        self._observations: list[np.ndarray | None] = [None] * len(specs)
        self._infos: list[dict] = [{} for _ in specs]

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    def reset(self) -> tuple[np.ndarray, list[dict]]:
        pairs = [env.reset(spec.seed) for env, spec in zip(self.envs, self.specs)]
        self._observations = [pair[0] for pair in pairs]
        self._infos = [pair[1] for pair in pairs]
        return np.stack(self._observations).astype(np.float32), list(self._infos)

    def reset_at(self, index: int, spec: EnvSpec) -> tuple[np.ndarray, dict]:
        self.specs[index] = spec
        self.envs[index] = make_env(spec)
        observation, info = self.envs[index].reset(spec.seed)
        self._observations[index] = observation
        self._infos[index] = info
        return observation, info

    def step(self, actions, active_mask=None) -> VectorStepResult:
        if len(actions) != self.num_envs:
            raise ValueError("action batch size does not match num_envs")
        active = (
            np.ones(self.num_envs, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if active.shape != (self.num_envs,):
            raise ValueError("active_mask must have shape [num_envs]")
        if any(observation is None for observation in self._observations):
            raise RuntimeError("vector environment must be reset before step")

        observations = list(self._observations)
        infos = list(self._infos)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        for index, (env, action) in enumerate(zip(self.envs, actions)):
            if not active[index]:
                continue
            result = env.step(action)
            observations[index] = result.observation
            infos[index] = result.info
            rewards[index] = result.reward
            terminated[index] = result.terminated
            truncated[index] = result.truncated

        self._observations = observations
        self._infos = infos
        return VectorStepResult(
            np.stack(observations).astype(np.float32),
            rewards,
            terminated,
            truncated,
            infos,
        )

    def close(self) -> None:
        self.envs.clear()
