"""Weighted, checkpointable curriculum and scenario scheduler."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import random

import yaml

from air_combat_rl.vector.types import EnvSpec

SUPPORTED_METRICS = {
    "success_rate",
    "escape_completion_rate",
    "survival_rate",
    "mean_reward",
}


@dataclass(frozen=True, slots=True)
class ScenarioChoice:
    path: str
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class AdvancementRule:
    metric: str = "success_rate"
    threshold: float = 1.0
    consecutive_windows: int = 1
    source: str = "training"


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    name: str
    scenarios: tuple[ScenarioChoice, ...]
    minimum_episodes: int = 0
    maximum_steps: int | None = None
    advancement: AdvancementRule = field(default_factory=AdvancementRule)


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    name: str
    stages: tuple[CurriculumStage, ...]
    metric_window_episodes: int = 200
    minimum_stage_episodes: int = 0
    allow_regression: bool = False

    @classmethod
    def from_yaml(cls, path) -> "CurriculumConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("curriculum config must be a YAML mapping")
        window = int(data.get("metric_window_episodes", 200))
        minimum_stage_episodes = int(data.get("minimum_stage_episodes", 0))
        if window <= 0:
            raise ValueError("metric_window_episodes must be positive")
        if minimum_stage_episodes < 0:
            raise ValueError("minimum_stage_episodes must be non-negative")

        stages = []
        for raw in data.get("stages", []):
            choices = tuple(
                ScenarioChoice(str(choice["path"]), float(choice.get("weight", 1.0)))
                for choice in raw.get("scenarios", [])
            )
            if (
                not choices
                or any(choice.weight < 0 for choice in choices)
                or sum(choice.weight for choice in choices) <= 0
            ):
                raise ValueError(
                    f"curriculum stage {raw.get('name')} has invalid scenario weights"
                )
            rule = AdvancementRule(**raw.get("advancement", {}))
            if rule.metric not in SUPPORTED_METRICS:
                raise ValueError(f"unsupported curriculum metric: {rule.metric}")
            if rule.source != "training":
                raise ValueError(
                    "only training-window curriculum advancement is currently supported"
                )
            if rule.consecutive_windows <= 0:
                raise ValueError("consecutive_windows must be positive")
            minimum_episodes = int(raw.get("minimum_episodes", 0))
            maximum_steps = raw.get("maximum_steps")
            if minimum_episodes < 0 or (
                maximum_steps is not None and int(maximum_steps) <= 0
            ):
                raise ValueError(
                    f"curriculum stage {raw.get('name')} has invalid limits"
                )
            stages.append(
                CurriculumStage(
                    str(raw["name"]),
                    choices,
                    minimum_episodes,
                    None if maximum_steps is None else int(maximum_steps),
                    rule,
                )
            )
        if not stages:
            raise ValueError("curriculum must define at least one stage")
        return cls(
            str(data.get("name", "curriculum")),
            tuple(stages),
            window,
            minimum_stage_episodes,
            bool(data.get("allow_regression", False)),
        )


class CurriculumScheduler:
    """Assign weighted scenarios and advance through performance stages."""

    def __init__(
        self,
        config: CurriculumConfig,
        *,
        actions_path: str,
        platform: str,
        base_seed: int,
        max_policy_steps=None,
        projection_config=None,
    ) -> None:
        self.config = config
        self.actions_path = actions_path
        self.platform = platform
        self.base_seed = int(base_seed)
        self.max_policy_steps = max_policy_steps
        self.projection_config = projection_config
        self.stage_index = 0
        self.stage_start_step = 0
        self.stage_episodes = 0
        self.total_episodes = 0
        self.consecutive_windows = 0
        self.next_episode_id = 0
        self.rng = random.Random(base_seed)
        self.recent = deque(maxlen=config.metric_window_episodes)
        self.scenario_counts: dict[str, int] = {}

    @property
    def stage(self) -> CurriculumStage:
        return self.config.stages[self.stage_index]

    def next_spec(self, worker_index=0) -> EnvSpec:
        choices = self.stage.scenarios
        selected = self.rng.choices(
            choices, weights=[choice.weight for choice in choices], k=1
        )[0]
        episode_id = self.next_episode_id
        self.next_episode_id += 1
        seed = self.base_seed + episode_id * 1_000_003 + worker_index
        self.scenario_counts[selected.path] = (
            self.scenario_counts.get(selected.path, 0) + 1
        )
        return EnvSpec(
            selected.path,
            self.actions_path,
            self.platform,
            seed,
            self.max_policy_steps,
            True,
            self.stage.name,
            episode_id,
            self.projection_config,
        )

    def record_episode(self, outcome: str, reward: float = 0.0) -> None:
        self.total_episodes += 1
        self.stage_episodes += 1
        self.recent.append({"outcome": str(outcome), "reward": float(reward)})

    def metrics(self) -> dict:
        count = len(self.recent)
        outcomes = [result["outcome"] for result in self.recent]
        return {
            "episodes": count,
            "success_rate": (
                0.0
                if not count
                else sum(value == "success" for value in outcomes) / count
            ),
            "escape_completion_rate": (
                0.0
                if not count
                else sum(value in {"success", "exhausted"} for value in outcomes)
                / count
            ),
            "survival_rate": (
                0.0
                if not count
                else sum(value not in {"hit", "crash"} for value in outcomes) / count
            ),
            "mean_reward": (
                0.0
                if not count
                else sum(result["reward"] for result in self.recent) / count
            ),
        }

    def maybe_advance(self, global_step: int):
        if self.stage_index >= len(self.config.stages) - 1:
            return None
        minimum = max(self.config.minimum_stage_episodes, self.stage.minimum_episodes)
        rule = self.stage.advancement
        metrics = self.metrics()
        enough_samples = (
            self.stage_episodes >= minimum and len(self.recent) == self.recent.maxlen
        )
        passed = enough_samples and metrics[rule.metric] >= rule.threshold
        self.consecutive_windows = self.consecutive_windows + 1 if passed else 0
        forced = (
            self.stage.maximum_steps is not None
            and global_step - self.stage_start_step >= self.stage.maximum_steps
        )
        if self.consecutive_windows < rule.consecutive_windows and not forced:
            return None

        old_stage = self.stage.name
        self.stage_index += 1
        self.stage_start_step = int(global_step)
        self.stage_episodes = 0
        self.consecutive_windows = 0
        self.recent.clear()
        return {
            "event": "stage_transition",
            "global_step": int(global_step),
            "from_stage": old_stage,
            "to_stage": self.stage.name,
            "forced": forced,
            "metrics": metrics,
        }

    def state_dict(self) -> dict:
        return {
            "stage_index": self.stage_index,
            "stage_start_step": self.stage_start_step,
            "stage_episodes": self.stage_episodes,
            "total_episodes": self.total_episodes,
            "consecutive_windows": self.consecutive_windows,
            "next_episode_id": self.next_episode_id,
            "recent": list(self.recent),
            "scenario_counts": dict(self.scenario_counts),
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state) -> None:
        stage_index = int(state["stage_index"])
        if not 0 <= stage_index < len(self.config.stages):
            raise ValueError(
                "checkpoint curriculum stage is incompatible with the configured curriculum"
            )
        self.stage_index = stage_index
        for key in (
            "stage_start_step",
            "stage_episodes",
            "total_episodes",
            "consecutive_windows",
            "next_episode_id",
        ):
            setattr(self, key, int(state[key]))
        self.recent.clear()
        self.recent.extend(state.get("recent", []))
        self.scenario_counts = dict(state.get("scenario_counts", {}))
        self.rng.setstate(state["rng_state"])
