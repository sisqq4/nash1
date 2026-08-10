"""Environment factories shared by serial and subprocess workers."""

from src.air_combat_rl.runtime import build_blue_escape_env
from src.air_combat_rl.tasks.blue_escape.continuous.wrapper import (
    ProjectedContinuousActionWrapper,
)
from src.air_combat_rl.tasks.blue_escape.continuous.mapper import (
    NearestManeuverConfig,
    NearestManeuverMapper,
)
from src.air_combat_rl.vector.types import EnvSpec


def make_env(spec: EnvSpec):
    env, _ = build_blue_escape_env(
        spec.scenario_path,
        spec.actions_path,
        spec.platform,
        spec.seed,
        spec.max_policy_steps,
        spec.platform_config_path,
        spec.reward_config_path,
    )
    if not spec.projected_continuous:
        return env
    projection = spec.projection_config or {}
    metric = projection.get("metric", "weighted_normalized_command_distance")
    if metric != "weighted_normalized_command_distance":
        raise ValueError(f"unsupported projection metric: {metric}")
    weights = projection.get("weights", {})
    ranges = projection.get("ranges", {})
    mapper_config = NearestManeuverConfig(
        w_nx=float(weights.get("nx", 1.0)),
        w_nf=float(weights.get("nf", 1.0)),
        w_gamma_s=float(weights.get("gamma_s", 1.0)),
        range_nx=float(ranges.get("nx", 18.0)),
        range_nf=float(ranges.get("nf", 9.0)),
        range_gamma_s=float(ranges.get("gamma_s", 3.141592653589793)),
    )
    if (
        min(mapper_config.range_nx, mapper_config.range_nf, mapper_config.range_gamma_s)
        <= 0
    ):
        raise ValueError("projection ranges must be positive")
    mapper = NearestManeuverMapper(env.actions, mapper_config)
    return ProjectedContinuousActionWrapper(env, mapper=mapper)
