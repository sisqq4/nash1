"""Application runner and checkpoints for vectorized PyTorch PPO training."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
import json
import os
import random
import tempfile
import hashlib

import numpy as np
import yaml

from src.air_combat_rl.algorithms.common.device import (
    describe_device,
    require_torch,
    resolve_device,
    seed_everything,
)
from src.air_combat_rl.algorithms.ppo.torch_actor_critic import TorchPPOActorCritic
from src.air_combat_rl.algorithms.ppo.torch_trainer import TorchPPOConfig, TorchPPOTrainer
from src.air_combat_rl.tasks.blue_escape.action_catalog import ActionCatalog
from src.air_combat_rl.training.curriculum import CurriculumConfig, CurriculumScheduler
from src.air_combat_rl.vector import EnvSpec, SerialVectorEnv, SubprocessVectorEnv
from src.air_combat_rl.io.progress import ExperimentProgress

torch = require_torch()
CHECKPOINT_SCHEMA_VERSION = 3


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _checkpoint_metadata(trainer, algorithm_config, curriculum_config):
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "algorithm_name": "ppo_projected",
        "backend": "torch",
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "grad_scaler_state_dict": trainer.scaler.state_dict(),
        "global_step": trainer.global_step,
        "update_index": trainer.update_index,
        "algorithm_config": algorithm_config,
        "observation_dim": trainer.model.obs_dim,
        "action_dim": trainer.model.action_dim,
        "continuous_action_dim": 3,
        "action_bounds": {"low": [-1.0] * 3, "high": [1.0] * 3},
        "action_catalog_version": trainer.action_catalog_version,
        "projection_config": algorithm_config.get("projection", {}),
        "curriculum_config": curriculum_config,
        "curriculum_signature": trainer.curriculum_signature,
        "curriculum_state": (
            trainer.scheduler.state_dict() if trainer.scheduler else None
        ),
        "env_specs": [asdict(spec) for spec in trainer.env.specs],
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
    }


def save_torch_checkpoint(
    path, trainer, algorithm_config, curriculum_config=None
) -> None:
    """Atomically save model, optimizer, RNG, and curriculum state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_metadata(trainer, algorithm_config, curriculum_config)
    file_descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f"{path.name}.tmp."
    )
    os.close(file_descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_torch_checkpoint(path, trainer, *, expected_projection=None):
    """Restore a strictly compatible Projected-PPO checkpoint."""
    data = torch.load(path, map_location=trainer.device, weights_only=False)
    if data.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported torch checkpoint schema")
    if data.get("algorithm_name") != "ppo_projected" or data.get("backend") != "torch":
        raise ValueError("incompatible torch projected PPO checkpoint")
    if (
        data.get("observation_dim") != trainer.model.obs_dim
        or data.get("action_dim") != trainer.model.action_dim
    ):
        raise ValueError("checkpoint model dimension mismatch")
    if data.get("continuous_action_dim") != 3:
        raise ValueError("checkpoint continuous action dimension mismatch")
    if data.get("action_catalog_version") != trainer.action_catalog_version:
        raise ValueError("checkpoint action catalog mismatch")
    if (
        expected_projection is not None
        and data.get("projection_config", {}) != expected_projection
    ):
        raise ValueError("checkpoint projection config mismatch")
    if data.get("curriculum_signature") != trainer.curriculum_signature:
        raise ValueError("checkpoint curriculum config mismatch")

    trainer.model.load_state_dict(data["model_state_dict"])
    trainer.optimizer.load_state_dict(data["optimizer_state_dict"])
    trainer.scaler.load_state_dict(data.get("grad_scaler_state_dict", {}))
    trainer.global_step = int(data.get("global_step", 0))
    trainer.update_index = int(data.get("update_index", 0))
    if trainer.scheduler and data.get("curriculum_state"):
        trainer.scheduler.load_state_dict(data["curriculum_state"])

    rng_state = data.get("rng_state", {})
    if rng_state.get("python") is not None:
        random.setstate(rng_state["python"])
    if rng_state.get("numpy") is not None:
        np.random.set_state(rng_state["numpy"])
    if rng_state.get("torch") is not None:
        torch.set_rng_state(rng_state["torch"].cpu())
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    return data


def _validate_run_arguments(
    total_steps, checkpoint_interval, num_envs, backend
) -> None:
    if total_steps < 0:
        raise ValueError("total_steps must be non-negative")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if backend not in {"serial", "subprocess"}:
        raise ValueError(f"unsupported vector environment backend: {backend}")
    if total_steps % num_envs != 0:
        raise ValueError("total_steps must be divisible by num_envs")


def run_torch_training(
    *,
    algorithm_config,
    actions,
    platform,
    seed,
    output_dir,
    total_steps,
    checkpoint_interval,
    device="auto",
    num_envs=None,
    env_backend=None,
    start_method="spawn",
    scenario=None,
    curriculum=None,
    max_policy_steps=None,
    resume=None,
    amp=None,
):
    device_config = algorithm_config.get("device", {})
    resolved_device = resolve_device(
        device if device != "auto" else device_config.get("type", "auto")
    )
    seed_everything(seed, bool(device_config.get("deterministic", False)))

    vector_config = algorithm_config.get("vector_env", {})
    num_envs = int(vector_config.get("num_envs", 8) if num_envs is None else num_envs)
    backend = env_backend or vector_config.get("backend", "subprocess")
    _validate_run_arguments(total_steps, checkpoint_interval, num_envs, backend)

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError(
            f"output directory is not empty: {output}; use --resume or another directory"
        )
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    if curriculum:
        curriculum_config = CurriculumConfig.from_yaml(curriculum)
        scheduler = CurriculumScheduler(
            curriculum_config,
            actions_path=actions,
            platform=platform,
            base_seed=seed,
            max_policy_steps=max_policy_steps,
            projection_config=algorithm_config.get("projection", {}),
        )
        specs = [scheduler.next_spec(index) for index in range(num_envs)]
    else:
        if not scenario:
            raise ValueError("either scenario or curriculum is required")
        curriculum_config = None
        scheduler = None
        specs = [
            EnvSpec(
                scenario,
                actions,
                platform,
                seed + index,
                max_policy_steps,
                True,
                "single",
                index,
                algorithm_config.get("projection", {}),
            )
            for index in range(num_envs)
        ]

    vector_env = (
        SerialVectorEnv(specs)
        if backend == "serial"
        else SubprocessVectorEnv(specs, start_method)
    )
    try:
        observations, infos = vector_env.reset()
        model_config = algorithm_config.get("model", {})
        model = TorchPPOActorCritic(
            observations.shape[1],
            3,
            tuple(model_config.get("hidden_sizes", [256, 256])),
        ).to(resolved_device)
        allowed = TorchPPOConfig.__dataclass_fields__
        ppo_config = TorchPPOConfig(
            **{
                key: value
                for key, value in algorithm_config.get("ppo", {}).items()
                if key in allowed
            }
        )
        trainer = TorchPPOTrainer(
            vector_env,
            model,
            resolved_device,
            ppo_config,
            scheduler,
            device_config.get("amp", True) if amp is None else amp,
        )
        trainer.action_catalog_version = ActionCatalog.from_yaml(actions).version
        trainer.curriculum_signature = (
            hashlib.sha256(Path(curriculum).read_bytes()).hexdigest()
            if curriculum
            else None
        )
        trainer.observations = observations
        trainer.infos = infos

        if resume:
            checkpoint_data = load_torch_checkpoint(
                resume,
                trainer,
                expected_projection=algorithm_config.get("projection", {}),
            )
            stored_specs = checkpoint_data.get("env_specs", [])
            if len(stored_specs) != num_envs:
                raise ValueError("checkpoint vector environment count mismatch")
            for stored_spec in stored_specs:
                if stored_spec.get("platform") != platform:
                    raise ValueError("checkpoint platform mismatch")
                if stored_spec.get("max_policy_steps") != max_policy_steps:
                    raise ValueError("checkpoint max_policy_steps mismatch")
                if curriculum is None and stored_spec.get("scenario_path") != scenario:
                    raise ValueError("checkpoint scenario mismatch")
            reset_rows = [
                vector_env.reset_at(index, EnvSpec(**stored_spec))
                for index, stored_spec in enumerate(stored_specs)
            ]
            trainer.observations = np.stack([row[0] for row in reset_rows]).astype(
                np.float32
            )
            trainer.infos = [row[1] for row in reset_rows]

        manifest = {
            "algorithm_name": "ppo_projected",
            "backend": "torch",
            "seed": seed,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "device": describe_device(resolved_device),
            "parallelism": {
                "num_envs": num_envs,
                "backend": backend,
                "start_method": start_method,
            },
            "scenario": scenario,
            "curriculum": curriculum,
            "action_catalog_version": trainer.action_catalog_version,
            "algorithm_config": algorithm_config,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
        )
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )

        next_checkpoint = (
            (trainer.global_step // checkpoint_interval) + 1
        ) * checkpoint_interval
        progress = ExperimentProgress(total_steps, "training", "step")
        try:
          with (
            (output / "train_metrics.jsonl").open(
                "a", encoding="utf-8"
            ) as metrics_file,
            (output / "episodes.jsonl").open("a", encoding="utf-8") as episode_file,
            (output / "curriculum.jsonl").open(
                "a", encoding="utf-8"
            ) as curriculum_file,
        ):
            while trainer.global_step < total_steps:
                remaining_vector_steps = (total_steps - trainer.global_step) // num_envs
                rollout_steps = min(
                    trainer.config.rollout_steps, remaining_vector_steps
                )
                buffer, timing = trainer.collect_rollout(rollout_steps)
                metrics = trainer.train_one_update(buffer) | timing
                elapsed = metrics["rollout_seconds"] + metrics["optimization_seconds"]
                metrics["samples_per_second"] = metrics["batch_size"] / max(
                    elapsed, 1e-9
                )
                metrics_file.write(json.dumps(metrics, sort_keys=True) + "\n")
                metrics_file.flush()

                for record in trainer.completed_episodes:
                    episode_file.write(json.dumps(record, sort_keys=True) + "\n")
                progress.record_outcomes(record.get("outcome") for record in trainer.completed_episodes)
                trainer.completed_episodes.clear()
                episode_file.flush()

                for event in trainer.curriculum_events:
                    curriculum_file.write(json.dumps(event, sort_keys=True) + "\n")
                    save_torch_checkpoint(
                        checkpoints / f"stage_{event['to_stage']}_start.pt",
                        trainer,
                        algorithm_config,
                        curriculum,
                    )
                trainer.curriculum_events.clear()
                curriculum_file.flush()
                progress.update(trainer.global_step, metrics)

                if trainer.global_step >= next_checkpoint:
                    save_torch_checkpoint(
                        checkpoints / f"step_{trainer.global_step}.pt",
                        trainer,
                        algorithm_config,
                        curriculum,
                    )
                    while next_checkpoint <= trainer.global_step:
                        next_checkpoint += checkpoint_interval
        finally:
            progress.close()

        save_torch_checkpoint(
            checkpoints / "latest.pt", trainer, algorithm_config, curriculum
        )
        return {
            "global_step": trainer.global_step,
            "output_dir": str(output),
            "device": str(resolved_device),
            "curriculum_stage": scheduler.stage.name if scheduler else None,
        }
    finally:
        vector_env.close()
