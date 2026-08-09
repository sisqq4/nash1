"""Batch evaluator for blue-escape algorithms and baselines."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import math
import random

import numpy as np
import yaml

from air_combat_rl.evaluation.metrics import summarize_episodes
from air_combat_rl.evaluation.report import render_report
from air_combat_rl.io.trajectory_writer import normalize_json
from air_combat_rl.runtime import build_algorithm_runtime, build_blue_escape_env


class EvaluationError(ValueError):
    """Raised when evaluation configuration or runtime validation fails."""


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _load_state(obj, state: dict) -> None:
    for key, value in (state or {}).items():
        if key == "rng_state":
            continue
        current = getattr(obj, key, None)
        if isinstance(current, list):
            for target, source in zip(current, value):
                target[...] = np.asarray(source, float)
        elif isinstance(current, np.ndarray):
            current[...] = np.asarray(value, float)
        elif hasattr(obj, key):
            setattr(obj, key, np.asarray(value, float) if isinstance(value, list) else value)


def _checkpoint_load(runtime, path: str | Path | None, alg: str, env) -> dict | None:
    if alg in {"constant", "random_valid"}:
        if path:
            raise EvaluationError("baselines must not provide checkpoint")
        return None
    checkpoint_path = Path(path or "")
    if not checkpoint_path.exists():
        raise EvaluationError(f"checkpoint does not exist: {checkpoint_path}")
    data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if data.get("algorithm_name") != alg:
        raise EvaluationError(f"checkpoint algorithm {data.get('algorithm_name')!r} != {alg!r}")
    obs_dim = len(env.reset(0)[0])
    if data.get("observation_dim") not in (None, obs_dim):
        raise EvaluationError("observation dimension mismatch")
    if data.get("action_catalog_version") not in (None, getattr(env.actions, "version", None)):
        raise EvaluationError("action catalog version mismatch")
    model = getattr(runtime.trainer, "actor_critic", getattr(runtime.trainer, "q_network", None))
    if data.get("action_dim") not in (None, getattr(model, "action_dim", None)):
        raise EvaluationError("action dimension mismatch")
    if alg == "ppo_projected":
        if data.get("continuous_action_dim") not in (None, 3):
            raise EvaluationError("projected PPO projection config mismatch")
        if data.get("action_bounds") not in (None, {"low": [-1, -1, -1], "high": [1, 1, 1]}):
            raise EvaluationError("projected PPO projection config mismatch")
    _load_state(model, data.get("model_state", {}))
    return data


class ConstantPolicy:
    algorithm_name = "constant"

    def __init__(self, env, action_id: int = 0) -> None:
        self.env = env
        self.action_id = int(action_id)

    def act(self, obs, deterministic: bool = False):
        from air_combat_rl.interfaces.policy import PolicyDecision

        mask = self.env.actions.action_mask(self.env.platform)
        if self.action_id < 0 or self.action_id >= len(mask) or not mask[self.action_id]:
            raise EvaluationError(f"illegal action {self.action_id}")
        command = self.env.actions.command_for(self.action_id, self.env.platform)
        return PolicyDecision("constant", None, None, None, self.action_id, command, 0.0, None, None)


class RandomValidPolicy:
    algorithm_name = "random_valid"

    def __init__(self, env, seed: int = 0) -> None:
        self.env = env
        self.rng = random.Random(seed)

    def act(self, obs, deterministic: bool = False):
        from air_combat_rl.interfaces.policy import PolicyDecision

        valid = [i for i, allowed in enumerate(self.env.actions.action_mask(self.env.platform)) if allowed]
        action_id = valid[0] if deterministic else self.rng.choice(valid)
        command = self.env.actions.command_for(action_id, self.env.platform)
        return PolicyDecision("random_valid", None, None, None, action_id, command, 0.0, None, None)


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _missile_outcomes(missiles) -> list[str]:
    outcomes = []
    for missile in missiles:
        if missile.alive and missile.locked:
            outcomes.append("alive_locked")
        elif missile.alive:
            outcomes.append("alive_unlocked")
        elif missile.locked:
            outcomes.append("dead_locked")
        else:
            outcomes.append("dead_unlocked")
    return outcomes


def _episode_record(
    env,
    scenario_path,
    alg: str,
    platform: str,
    seed: int,
    ep_index: int,
    steps: int,
    total_reward: float,
    outcome: str,
    reward_components: dict,
    proj_diag: dict,
    initial_missile_count: int,
    min_altitude_y_m: float,
    max_effective_threats: int,
) -> dict:
    snapshot = env.world.snapshot()
    missiles = list(snapshot.missiles)
    distances = [d for d in getattr(env.world, "min_missile_distances", []) if math.isfinite(float(d))]
    return {
        "episode_index": ep_index,
        "scenario": str(scenario_path),
        "platform": platform,
        "missile_count": initial_missile_count,
        "seed": seed,
        "algorithm": alg,
        "outcome": outcome,
        "total_reward": float(total_reward),
        "duration_s": float(snapshot.time_s),
        "policy_steps": int(steps),
        "min_sampled_distance_m": min(distances) if distances else None,
        "min_altitude_y_m": _finite_or_none(min_altitude_y_m),
        "ground_collision_count": 1 if outcome == "crash" else 0,
        "missile_exhaustion_count": 1 if outcome == "exhausted" else 0,
        "initial_missile_count": int(initial_missile_count),
        "final_alive_missile_count": int(sum(1 for missile in missiles if missile.alive)),
        "final_locked_missile_count": int(sum(1 for missile in missiles if missile.locked)),
        "missile_outcomes": _missile_outcomes(missiles),
        "max_simultaneous_effective_threats": int(max_effective_threats),
        "reward_component_totals": dict(reward_components),
        "threat_timeseries_ref": f"trajectories/episode_{ep_index:06d}.jsonl",
        "projected_ppo": proj_diag,
    }


def _projected_ppo_decision(runtime, observation, deterministic: bool):
    """Select a Projected PPO action using the same bounded action passed in training."""
    sample = runtime.trainer.actor_critic.act(observation, deterministic=deterministic)
    return sample, sample.squashed_action


def run_evaluation(
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
    constant_action_id: int = 0,
    max_policy_steps=None,
    device="auto",
    num_envs=8,
    env_backend="subprocess",
    start_method="spawn",
):
    cfg = load_yaml(algorithm_config_path) if algorithm_config_path else {"algorithm": {"name": "constant"}}
    if cfg.get("algorithm", {}).get("backend") == "torch":
        from air_combat_rl.evaluation.parallel_evaluator import run_parallel_torch_evaluation
        try:
            return run_parallel_torch_evaluation(scenarios=scenarios, actions=actions, algorithm_config_path=algorithm_config_path, checkpoint=checkpoint, platform=platform, episodes=episodes, seeds=seeds, deterministic=deterministic, output_dir=output_dir, max_policy_steps=max_policy_steps, device=device, num_envs=num_envs, env_backend=env_backend, start_method=start_method)
        except (ValueError, RuntimeError, OSError) as exc:
            raise EvaluationError(str(exc)) from exc
    if episodes <= 0:
        raise EvaluationError("episodes must be > 0")
    if not scenarios:
        raise EvaluationError("scenario list is empty")
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise EvaluationError(f"output directory conflict: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "trajectories").mkdir(exist_ok=True)

    alg = cfg.get("algorithm", {}).get("name", cfg.get("name", "ppo_projected"))
    all_rows = []

    with (output_path / "evaluation_steps.jsonl").open("w", encoding="utf-8") as step_file:
        ep_index = 0
        for scenario_path in scenarios:
            for seed in seeds:
                for local_ep in range(episodes):
                    actual_seed = int(seed) + local_ep * 1_000_003
                    env, _ = build_blue_escape_env(scenario_path, actions, platform, actual_seed, max_policy_steps)
                    initial_missile_count = len(env.world.missiles)
                    if alg in {"constant", "random_valid"}:
                        policy = ConstantPolicy(env, constant_action_id) if alg == "constant" else RandomValidPolicy(env, actual_seed)
                        _checkpoint_load(None, checkpoint, alg, env)
                        runtime = None
                    else:
                        cfg_for_episode = dict(cfg)
                        cfg_for_episode["seed"] = actual_seed
                        runtime = build_algorithm_runtime(cfg_for_episode, env)
                        _checkpoint_load(runtime, checkpoint, alg, env)
                        policy = runtime.policy

                    obs, info = env.reset(actual_seed)
                    done = False
                    total_reward = 0.0
                    steps = 0
                    component_totals = defaultdict(float)
                    min_altitude_y_m = float(env.world.blue.kinematics.position.y)
                    max_effective_threats = sum(1 for missile in env.world.missiles if missile.alive and missile.locked)
                    projected_diag = {
                        "projection_distances": [],
                        "projected_action_distribution": Counter(),
                        "continuous_actions": [],
                        "continuous_commands": [],
                        "projected_commands": [],
                        "valid_action_counts": [],
                        "saturation_flags": [],
                        "continuous_to_discrete_mapping_frequency": Counter(),
                    }
                    trajectory_path = output_path / "trajectories" / f"episode_{ep_index:06d}.jsonl"
                    with trajectory_path.open("w", encoding="utf-8") as trajectory_file:
                        while not done:
                            if alg == "ppo_projected":
                                sample, wrapper_action = _projected_ppo_decision(runtime, obs, deterministic)
                                result = runtime.env.step(wrapper_action)
                                info_after_step = result.info
                                action_id = int(info_after_step["executed_action_id"])
                                if not info["action_mask"][action_id]:
                                    raise EvaluationError(f"illegal action: {action_id}")
                                bounded = np.asarray(wrapper_action, float)
                                projection_distance = float(info_after_step.get("projection_distance", 0.0))
                                projected_diag["projection_distances"].append(projection_distance)
                                projected_diag["projected_action_distribution"][action_id] += 1
                                projected_diag["continuous_actions"].append(bounded.tolist())
                                continuous_command = info_after_step.get("continuous_command")
                                projected_command = info_after_step.get("executed_command")
                                projected_diag["continuous_commands"].append(
                                    [continuous_command.nx, continuous_command.nf, continuous_command.gamma_s]
                                )
                                projected_diag["projected_commands"].append(
                                    [projected_command.nx, projected_command.nf, projected_command.gamma_s]
                                )
                                projected_diag["valid_action_counts"].append(int(info_after_step.get("valid_action_count", 0)))
                                projected_diag["saturation_flags"].append(bool(np.any(np.abs(bounded) >= 0.99)))
                                mapping_key = f"{np.round(bounded, 3).tolist()}->{action_id}"
                                projected_diag["continuous_to_discrete_mapping_frequency"][mapping_key] += 1
                            else:
                                decision = policy.act(obs, deterministic=deterministic)
                                action_id = int(decision.executed_action_id)
                                if not info["action_mask"][action_id]:
                                    raise EvaluationError(f"illegal action: {action_id}")
                                result = env.step(action_id)
                                info_after_step = result.info

                            total_reward += float(result.reward)
                            steps += 1
                            obs = result.observation
                            info = result.info
                            done = result.terminated or result.truncated
                            min_altitude_y_m = min(min_altitude_y_m, float(env.world.blue.kinematics.position.y))
                            max_effective_threats = max(
                                max_effective_threats,
                                sum(1 for missile in env.world.missiles if missile.alive and missile.locked),
                            )
                            for key, value in (info_after_step.get("reward_components") or {}).items():
                                component_totals[key] += float(value)
                            step_record = {
                                "episode_index": ep_index,
                                "step": steps,
                                "seed": actual_seed,
                                "scenario": str(scenario_path),
                                "action_id": action_id,
                                "reward": float(result.reward),
                                "outcome": info_after_step.get("outcome"),
                                "time_s": info_after_step.get("time_s"),
                                "threat_count": sum(1 for missile in env.world.missiles if missile.alive and missile.locked),
                                "trajectory_ref": str(trajectory_path.relative_to(output_path)),
                            }
                            if alg == "ppo_projected":
                                step_record.update({
                                    "bounded_continuous_action": bounded.tolist(),
                                    "continuous_command_nx_nf_gamma_s": projected_diag["continuous_commands"][-1],
                                    "projected_command_nx_nf_gamma_s": projected_diag["projected_commands"][-1],
                                    "projection_distance": projection_distance,
                                })
                            line = json.dumps(normalize_json(step_record), sort_keys=True)
                            step_file.write(line + "\n")
                            trajectory_file.write(line + "\n")

                    row = _episode_record(
                        env,
                        scenario_path,
                        alg,
                        platform,
                        actual_seed,
                        ep_index,
                        steps,
                        total_reward,
                        info.get("outcome", "running"),
                        component_totals,
                        {key: (dict(value) if isinstance(value, Counter) else value) for key, value in projected_diag.items()},
                        initial_missile_count,
                        min_altitude_y_m,
                        max_effective_threats,
                    )
                    all_rows.append(row)
                    ep_index += 1

    metrics = summarize_episodes(all_rows)
    manifest = {
        "output_dir": str(output_path),
        "algorithm": alg,
        "algorithm_config": str(algorithm_config_path),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "scenarios": [str(scenario) for scenario in scenarios],
        "actions": str(actions),
        "action_catalog_version": getattr(
            build_blue_escape_env(scenarios[0], actions, platform, int(seeds[0]), max_policy_steps)[0].actions,
            "version",
            None,
        ),
        "platform": platform,
        "episodes": episodes,
        "seeds": [int(seed) for seed in seeds],
        "deterministic": deterministic,
        "fair_comparison": {
            "same_scenarios": True,
            "same_seeds": True,
            "same_episode_count": True,
            "same_platform": True,
            "same_actions": True,
            "same_observation_reward_termination": True,
        },
    }
    (output_path / "manifest.json").write_text(json.dumps(normalize_json(manifest), indent=2, sort_keys=True), encoding="utf-8")
    keys = list(all_rows[0].keys()) if all_rows else []
    with (output_path / "episodes.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: json.dumps(normalize_json(value), sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    (output_path / "metrics.json").write_text(json.dumps(normalize_json(metrics), indent=2, sort_keys=True), encoding="utf-8")
    (output_path / "report.md").write_text(render_report(manifest, metrics, ["本阶段不实现图表或ACMI。"]), encoding="utf-8")
    return {"output_dir": str(output_path), "episodes": len(all_rows), "metrics": metrics}
