"""Serializable vector-environment records."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class EnvSpec:
    scenario_path: str
    actions_path: str
    platform: str
    seed: int
    max_policy_steps: int | None = None
    projected_continuous: bool = True
    stage_name: str = "single"
    episode_id: int = 0
    projection_config: dict | None = None
    platform_config_path: str | None = None
    reward_config_path: str | None = None


@dataclass(slots=True)
class VectorStepResult:
    observations: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: list[dict]
