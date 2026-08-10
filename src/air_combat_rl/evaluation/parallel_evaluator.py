"""Parallel CPU simulation with batched GPU policy inference."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import csv
import json

import numpy as np
import yaml

from src.air_combat_rl.algorithms.common.device import (
    describe_device,
    require_torch,
    resolve_device,
)
from src.air_combat_rl.algorithms.ppo.torch_actor_critic import TorchPPOActorCritic
from src.air_combat_rl.evaluation.metrics import summarize_episodes
from src.air_combat_rl.evaluation.report import render_report
from src.air_combat_rl.io.trajectory_writer import normalize_json
from src.air_combat_rl.io.progress import ExperimentProgress
from src.air_combat_rl.tasks.blue_escape.action_catalog import ActionCatalog
from src.air_combat_rl.vector import EnvSpec, SerialVectorEnv, SubprocessVectorEnv

torch = require_torch()


def _new_diagnostics() -> dict:
    return {
        "projection_distances": [],
        "continuous_actions": [],
        "valid_action_counts": [],
        "saturation_flags": [],
        "projected_action_distribution": {},
        "continuous_to_discrete_mapping_frequency": {},
    }


def _validate_checkpoint(
    payload, model, action_catalog_version, projection_config
) -> None:
    if payload.get("schema_version") != 3:
        raise ValueError("unsupported torch checkpoint schema")
    if (
        payload.get("algorithm_name") != "ppo_projected"
        or payload.get("backend") != "torch"
    ):
        raise ValueError("incompatible torch projected PPO checkpoint")
    if (
        payload.get("observation_dim") != model.obs_dim
        or payload.get("action_dim") != model.action_dim
    ):
        raise ValueError("checkpoint model dimension mismatch")
    if payload.get("continuous_action_dim") != 3:
        raise ValueError("checkpoint continuous action dimension mismatch")
    if payload.get("action_bounds") != {"low": [-1.0] * 3, "high": [1.0] * 3}:
        raise ValueError("checkpoint continuous action bounds mismatch")
    if payload.get("action_catalog_version") != action_catalog_version:
        raise ValueError("checkpoint action catalog mismatch")
    if payload.get("projection_config", {}) != projection_config:
        raise ValueError("checkpoint projection config mismatch")


def run_parallel_torch_evaluation(
    *,
    scenarios,
    actions,
    algorithm_config_path,
    checkpoint,
    platform,
    episodes,
    seeds,
    deterministic,
    output_dir,
    max_policy_steps=None,
    device="auto",
    num_envs=8,
    env_backend="subprocess",
    start_method="spawn",
):
    if episodes <= 0:
        raise ValueError("episodes must be > 0")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if env_backend not in {"serial", "subprocess"}:
        raise ValueError(f"unsupported vector environment backend: {env_backend}")
    if checkpoint is None or not Path(checkpoint).is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")

    config = (
        yaml.safe_load(Path(algorithm_config_path).read_text(encoding="utf-8")) or {}
    )
    projection_config = config.get("projection", {})
    tasks = []
    for scenario in scenarios:
        for seed in seeds:
            for local_episode in range(episodes):
                tasks.append(
                    EnvSpec(
                        str(scenario),
                        str(actions),
                        platform,
                        int(seed) + local_episode * 1_000_003,
                        max_policy_steps,
                        True,
                        "evaluation",
                        len(tasks),
                        projection_config,
                    )
                )
    if not tasks:
        raise ValueError("scenario list is empty")

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory conflict: {output}")
    (output / "trajectories").mkdir(parents=True, exist_ok=True)

    requested_device = (
        device if device != "auto" else config.get("device", {}).get("type", "auto")
    )
    resolved_device = resolve_device(requested_device)
    action_catalog_version = ActionCatalog.from_yaml(actions).version
    worker_count = min(int(num_envs), len(tasks))
    initial = tasks[:worker_count]
    pending = deque(tasks[worker_count:])
    vector_env = (
        SerialVectorEnv(initial)
        if env_backend == "serial"
        else SubprocessVectorEnv(initial, start_method)
    )

    rows = []
    progress = ExperimentProgress(len(tasks), "evaluation", "episode")
    trajectory_files = {}
    step_file = (output / "evaluation_steps.jsonl").open("w", encoding="utf-8")
    try:
        observations, infos = vector_env.reset()
        model_config = config.get("model", {})
        model = TorchPPOActorCritic(
            observations.shape[1],
            3,
            tuple(model_config.get("hidden_sizes", [256, 256])),
        ).to(resolved_device)
        payload = torch.load(
            checkpoint, map_location=resolved_device, weights_only=False
        )
        _validate_checkpoint(
            payload, model, action_catalog_version, config.get("projection", {})
        )
        model.load_state_dict(payload["model_state_dict"])
        model.eval()

        active = np.ones(worker_count, dtype=bool)
        returns = np.zeros(worker_count, dtype=np.float64)
        lengths = np.zeros(worker_count, dtype=np.int64)
        minimum_altitudes = np.full(worker_count, np.inf)
        components = [defaultdict(float) for _ in range(worker_count)]
        diagnostics = [_new_diagnostics() for _ in range(worker_count)]
        missile_counts = [int(info.get("initial_missile_count", 0)) for info in infos]
        for spec in initial:
            trajectory_files[spec.episode_id] = (
                output / "trajectories" / f"episode_{spec.episode_id:06d}.jsonl"
            ).open("w", encoding="utf-8")

        while active.any():
            active_indices = np.flatnonzero(active)
            with torch.inference_mode():
                active_observations = torch.as_tensor(
                    observations[active_indices],
                    dtype=torch.float32,
                    device=resolved_device,
                )
                active_sample = model.act(
                    active_observations, deterministic=deterministic
                )
            actions_batch = np.zeros((worker_count, 3), dtype=np.float32)
            actions_batch[active_indices] = active_sample.squashed_action.cpu().numpy()
            result = vector_env.step(actions_batch, active_mask=active)

            for env_index in active_indices:
                spec = vector_env.specs[env_index]
                info = result.infos[env_index]
                bounded = actions_batch[env_index].tolist()
                returns[env_index] += result.rewards[env_index]
                lengths[env_index] += 1
                minimum_altitudes[env_index] = min(
                    minimum_altitudes[env_index],
                    float(info.get("altitude_y_m", np.inf)),
                )
                for key, value in (info.get("reward_components") or {}).items():
                    components[env_index][key] += float(value)

                action_id = int(info.get("executed_action_id", -1))
                diagnostic = diagnostics[env_index]
                diagnostic["projection_distances"].append(
                    float(info.get("projection_distance", 0))
                )
                diagnostic["continuous_actions"].append(bounded)
                diagnostic["valid_action_counts"].append(
                    int(info.get("valid_action_count", 0))
                )
                diagnostic["saturation_flags"].append(
                    any(abs(value) >= 0.99 for value in bounded)
                )
                action_key = str(action_id)
                action_counts = diagnostic["projected_action_distribution"]
                action_counts[action_key] = action_counts.get(action_key, 0) + 1
                mapping = f"{np.round(bounded, 3).tolist()}->{action_id}"
                mappings = diagnostic["continuous_to_discrete_mapping_frequency"]
                mappings[mapping] = mappings.get(mapping, 0) + 1

                record = {
                    "episode_index": spec.episode_id,
                    "step": int(lengths[env_index]),
                    "seed": spec.seed,
                    "scenario": spec.scenario_path,
                    "action_id": action_id,
                    "reward": float(result.rewards[env_index]),
                    "outcome": info.get("outcome"),
                    "time_s": info.get("time_s"),
                    "bounded_continuous_action": bounded,
                    "projection_distance": float(info.get("projection_distance", 0)),
                }
                line = json.dumps(normalize_json(record), sort_keys=True)
                step_file.write(line + "\n")
                trajectory_files[spec.episode_id].write(line + "\n")

                if not (result.terminated[env_index] or result.truncated[env_index]):
                    continue

                trajectory_files.pop(spec.episode_id).close()
                outcome = str(info.get("outcome", "running"))
                rows.append(
                    {
                        "episode_index": spec.episode_id,
                        "scenario": spec.scenario_path,
                        "platform": platform,
                        "missile_count": missile_counts[env_index],
                        "seed": spec.seed,
                        "algorithm": "ppo_projected",
                        "outcome": outcome,
                        "total_reward": float(returns[env_index]),
                        "duration_s": float(info.get("time_s", 0)),
                        "policy_steps": int(lengths[env_index]),
                        "min_sampled_distance_m": info.get("min_sampled_distance_m"),
                        "min_altitude_y_m": (
                            None
                            if not np.isfinite(minimum_altitudes[env_index])
                            else float(minimum_altitudes[env_index])
                        ),
                        "ground_collision_count": int(outcome == "crash"),
                        "missile_exhaustion_count": int(outcome == "exhausted"),
                        "reward_component_totals": dict(components[env_index]),
                        "projected_ppo": diagnostics[env_index],
                    }
                )
                progress.record_outcomes([outcome])
                progress.update(len(rows))

                if pending:
                    new_spec = pending.popleft()
                    reset_observation, reset_info = vector_env.reset_at(
                        env_index, new_spec
                    )
                    result.observations[env_index] = reset_observation
                    trajectory_files[new_spec.episode_id] = (
                        output
                        / "trajectories"
                        / f"episode_{new_spec.episode_id:06d}.jsonl"
                    ).open("w", encoding="utf-8")
                    returns[env_index] = 0
                    lengths[env_index] = 0
                    minimum_altitudes[env_index] = np.inf
                    components[env_index] = defaultdict(float)
                    diagnostics[env_index] = _new_diagnostics()
                    missile_counts[env_index] = int(
                        reset_info.get("initial_missile_count", 0)
                    )
                else:
                    active[env_index] = False
            observations = result.observations

        rows.sort(key=lambda row: row["episode_index"])
        metrics = summarize_episodes(rows)
        manifest = {
            "algorithm": "ppo_projected",
            "backend": "torch",
            "checkpoint": str(checkpoint),
            "scenarios": [str(scenario) for scenario in scenarios],
            "actions": str(actions),
            "action_catalog_version": action_catalog_version,
            "platform": platform,
            "episodes": episodes,
            "seeds": [int(seed) for seed in seeds],
            "deterministic": deterministic,
            "device": describe_device(resolved_device),
            "parallelism": {"num_envs": worker_count, "backend": env_backend},
            "fair_comparison": {
                "same_scenarios": True,
                "same_seeds": True,
                "same_episode_count": True,
                "same_platform": True,
                "same_actions": True,
                "same_observation_reward_termination": True,
            },
        }
        (output / "manifest.json").write_text(
            json.dumps(normalize_json(manifest), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        keys = list(rows[0]) if rows else []
        with (output / "episodes.csv").open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: (
                            json.dumps(normalize_json(value), sort_keys=True)
                            if isinstance(value, (dict, list))
                            else value
                        )
                        for key, value in row.items()
                    }
                )
        (output / "metrics.json").write_text(
            json.dumps(normalize_json(metrics), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output / "report.md").write_text(
            render_report(manifest, metrics, ["GPU批量策略推理，CPU并行场景仿真。"]),
            encoding="utf-8",
        )
        return {"output_dir": str(output), "episodes": len(rows), "metrics": metrics}
    finally:
        progress.close()
        for trajectory_file in trajectory_files.values():
            trajectory_file.close()
        step_file.close()
        vector_env.close()
