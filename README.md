# Air Combat RL

This repository is structured as a layered blue-escape research codebase.

> 中文使用说明与当前实现审计（场景、训练、GPU、结果展示、数学模型和参数）见
> [`docs/current_project_audit_zh.md`](docs/current_project_audit_zh.md)。该文档同时区分了
> “代码已经实现的能力”和“仍需物理校核/实验验证的假设”。

## Quick start

Install Python 3.10+ and all runtime/test dependencies:

```bash
python -m pip install -e '.[test]'
```

## Architecture

1. **Domain simulation kernel** (`src.air_combat_rl.core`, `src.air_combat_rl.domain`, `src.air_combat_rl.simulation`) owns coordinates, units, physical state, 3-DoF dynamics, guidance, propulsion, collision checks, scenarios, events, and snapshots.
2. **Task layer** (`src.air_combat_rl.tasks.blue_escape`) adapts the simulation kernel into a blue escape RL task with replaceable actions, observations, rewards, masks, and termination semantics.
3. **Algorithm layer** (`src.air_combat_rl.algorithms`) contains scene-agnostic learning code for the supported training schemes.
4. **Application layer** (`src.air_combat_rl.training`, `src.air_combat_rl.evaluation`, `scripts`) wires configs, runners, evaluators, logging, and checkpointing.

Dependency direction is intentionally top-down: dynamics do not import training, PPO, observations, or rewards; algorithms only depend on standard policy/environment interfaces; logging reads events and snapshots and never participates in state transition.

## Coordinate and environment contract

The canonical physical coordinate convention is XZY. State serialization is `[x, z, y, V, gamma, psi]`, where `y` is vertical altitude, `psi` increases for right turns from `+x` toward `+z`, and `gamma` increases for climb. Height, low-altitude, ground-collision, and safety-altitude logic must use `y`.

`BlueEscapeEnv` remains the canonical 29-discrete-action environment. Platform legality is represented by `action_mask`; illegal platform actions must be excluded by the policy or projection layer before action selection.

## Supported algorithms

The current primary and default algorithm is **`ppo_projected`**.

- **`ppo_projected`**: continuous PPO policy over maneuver intent. The actor samples a continuous action, bounds it, converts it to `[nx, nf, gamma_s]`, projects it onto the currently legal 29-action subset, and executes the selected `action_id` through `BlueEscapeEnv.step(action_id)`. PPO log probabilities and ratios belong to the continuous sample before projection, not to the final discrete action id.
- **`ppo_discrete`**: discrete PPO comparison baseline. The actor should produce a categorical distribution over the 29 actions after applying `action_mask`; log probabilities correspond to the actually executed discrete `action_id`.
- **`rainbow_dqn`**: discrete DQN comparison baseline. It uses the original 29-action `BlueEscapeEnv`, action masks, discrete replay semantics, and Rainbow-compatible checkpoints.

All algorithms are intended to share scenario configs, observation schema, reward functions, action catalog, termination semantics, and evaluation metrics so that comparisons measure algorithm differences rather than task differences.

## Single-scenario rollout CLI

The runtime automatically loads `configs/platform/<platform>.yaml` and selects
`configs/reward/escape_1v1.yaml` or `escape_1vn.yaml` from the scenario threat
count. Unknown parameters are rejected instead of being silently ignored.

Run one reproducible scenario without starting training or batch evaluation:

```bash
python scripts/run_scenario.py \
  --scenario configs/scenario/fixed_1v1.yaml \
  --actions configs/actions/blue_29.yaml \
  --platform zdj \
  --policy constant \
  --action-id 0 \
  --seed 0 \
  --output-dir runs/fixed_1v1_constant
```

`run_scenario.py` locates the repository root automatically when
run directly, including from an IDE or a working directory outside the
repository. Installing the project in editable mode as shown above remains the
recommended setup for development and for importing `src.air_combat_rl` elsewhere.

The command writes `manifest.json`, streaming `steps.jsonl`, and `episode_summary.json`. Use `--policy random_valid` to sample only currently legal masked actions with the supplied seed. A non-empty output directory is rejected unless `--overwrite` is passed.

Scenario YAML files can configure a regional, staggered red-missile launch with
`missile_spawn_distance_m`, `missile_spawn_bearing_deg`,
`missile_spawn_altitude_m`, `missile_first_launch_time_s`, and
`missile_launch_interval_s`. `blue_heading_deg` sets blue's initial heading and
`blue_detection_range_m` sets when blue starts executing policy maneuvers;
before detection it executes the level constant-speed action. See
`configs/scenario/regional_delayed_1vn.yaml` for a complete example.

## Training algorithms

The default training algorithm is `ppo_projected`.

Projected PPO uses this action chain: observation → continuous Gaussian actor → sampled pre-tanh action → tanh-bounded action → physical command `[nx, nf, gamma_s]` → weighted projection over the current `action_mask` legal 29-action catalog → `action_id` → `BlueEscapeEnv.step(action_id)`. PPO `old_log_prob`, `new_log_prob`, and ratio are computed from the continuous pre-tanh sample with the tanh Jacobian correction, not from the final projected discrete `action_id`.

Run the primary projected PPO path:

```bash
PYTHONPATH=src python scripts/train.py \
  --scenario configs/scenario/fixed_1v1.yaml \
  --actions configs/actions/blue_29.yaml \
  --algorithm configs/algorithm/ppo_projected.yaml \
  --platform zdj \
  --seed 0 \
  --output-dir runs/projected_ppo_fixed_1v1
```

Run the discrete PPO control path by replacing the algorithm file:

```bash
PYTHONPATH=src python scripts/train.py --scenario configs/scenario/fixed_1v1.yaml --actions configs/actions/blue_29.yaml --algorithm configs/algorithm/ppo_discrete.yaml --platform zdj --seed 0 --output-dir runs/discrete_ppo_fixed_1v1
```

Run the Rainbow DQN control path:

```bash
PYTHONPATH=src python scripts/train.py --scenario configs/scenario/fixed_1v1.yaml --actions configs/actions/blue_29.yaml --algorithm configs/algorithm/rainbow_dqn.yaml --platform zdj --seed 0 --output-dir runs/rainbow_dqn_fixed_1v1
```

Each run writes `manifest.json`, `train_metrics.jsonl`, `episodes.jsonl`, and atomic checkpoints under `checkpoints/latest.pt` plus interval `step_<N>.pt` files. Checkpoints include the algorithm name and are type-checked before algorithm-specific loading/resume code should accept them; do not load `ppo_projected`, `ppo_discrete`, and `rainbow_dqn` checkpoints across algorithm types.

Training and evaluation display live progress bars. Training shows rolling
success rate together with policy/value loss, entropy, KL divergence, gradient
norm, and throughput when those metrics are available. Evaluation shows its
rolling success rate as episodes complete.

Resume a compatible checkpoint into a new or existing output directory. The
`--total-steps` value is the desired final global step, not an additional count:

```bash
PYTHONPATH=src python scripts/train.py \
  --scenario configs/scenario/fixed_1v1.yaml \
  --actions configs/actions/blue_29.yaml \
  --algorithm configs/algorithm/ppo_projected.yaml \
  --platform zdj --seed 0 --total-steps 128 \
  --resume runs/projected_ppo_fixed_1v1/checkpoints/latest.pt \
  --output-dir runs/projected_ppo_resumed
```

## Batch evaluation and outcome metrics

Evaluate any learned algorithm by pairing its config with its own checkpoint:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --scenarios configs/scenario/fixed_1v1.yaml \
  --actions configs/actions/blue_29.yaml \
  --algorithm configs/algorithm/ppo_projected.yaml \
  --checkpoint runs/projected_ppo_fixed_1v1/checkpoints/latest.pt \
  --platform zdj --episodes 10 --seeds 0 1 --deterministic \
  --output-dir runs/eval_projected_ppo
```

Repeat with the matching Discrete PPO and Rainbow DQN config/checkpoint. Each
evaluation produces `metrics.json`, `episodes.csv`, `report.md`, a manifest,
`evaluation_steps.jsonl`, and per-episode files under `trajectories/`.
`success`/escape completion means the configured escape condition was met;
`hit` means a missile intercepted blue; `crash` means blue collided with the
ground; `exhausted` means threats ceased to be effective; and `timeout` means
the time/step limit was reached. Timeout remains a distinct outcome and is not
counted silently as success. Survival rate covers non-hit/non-crash episodes,
while escape-completion rate reports only the explicit success outcome.

## Result visualization

Visualization is an offline presentation layer: it reads recorded JSONL, CSV,
and JSON artifacts and never starts training or simulation. Matplotlib uses the
non-interactive `Agg` backend, so these commands work on headless workers.

Plot a single scenario run (XZY means the horizontal plane is **x-z** and
altitude is **y**):

```bash
PYTHONPATH=src python scripts/plot_run.py \
  --run-dir runs/fixed_1v1_constant
```

This writes horizontal and 3D trajectories, altitude, missile distance, reward,
and reward-component PNG files under `<run-dir>/plots/`. Missing optional reward
components produce an explanatory chart instead of failing the run artifact.

Plot an existing evaluation, including Projected PPO diagnostics when projection
fields are present:

```bash
PYTHONPATH=src python scripts/plot_evaluation.py \
  --evaluation-dir runs/eval_projected_ppo
```

Discrete PPO and Rainbow DQN evaluation plots omit Projected-PPO-only charts.
Compare algorithms only when manifests describe identical scenarios, platform,
episode/seed schedule, deterministic mode, and action catalog:

```bash
PYTHONPATH=src python scripts/compare_algorithms.py \
  --evaluations runs/eval_projected_ppo \
                runs/eval_discrete_ppo \
                runs/eval_rainbow_dqn \
  --output-dir runs/algorithm_comparison
```

The comparison includes outcome/survival rates, mean reward, duration, minimum
sampled distance, lowest y-altitude, and per-scenario survival. A condition
mismatch is rejected explicitly rather than producing an unfair chart.

## Tacview ACMI export

ACMI is an optional offline product; simulation, training, and evaluation do
not require it. Export a Phase-1 `steps.jsonl` trajectory with an explicit
geodetic origin (the tool never assumes 0°N, 0°E):

```bash
PYTHONPATH=src python scripts/export_acmi.py \
  --trajectory runs/fixed_1v1_constant/steps.jsonl \
  --origin-lat-deg 34.0 --origin-lon-deg 108.0 --origin-alt-m 0 \
  --reference-time 2026-01-01T00:00:00Z \
  --output runs/fixed_1v1_constant/trajectory.acmi
```

The exporter writes Tacview text ACMI 2.1. It maps simulation `x` to
north/latitude, `z` to east/longitude, and `y` to altitude above the supplied
origin. Each frame lists AIM-120 missiles as `b1`, `b2`, ... before the F16 as
`a1`, including `Name` and `Color` properties. It also writes heading from
`psi`, pitch from `gamma`, object removal, and supported hit and
ground-collision events. Because the model is three-DoF, roll is explicitly
written as zero rather than presenting a fabricated roll attitude.

## Output directory guide

- Scenario runs: `manifest.json`, `steps.jsonl`, `episode_summary.json`, and
  optional `trajectory.acmi`; offline charts are placed in `plots/`.
- Training runs: manifest and append-only training/episode JSONL plus
  algorithm-typed checkpoints under `checkpoints/`.
- Evaluation runs: aggregate JSON/CSV/Markdown, evaluation-step JSONL, episode
  trajectories, and optional charts under `plots/`.
- Comparison runs: fair-condition validation metadata and cross-algorithm PNGs.

Keep algorithm config and checkpoints paired: Projected PPO is the default
continuous-intent/projected primary method, Discrete PPO is the categorical
29-action PPO control, and Rainbow DQN is the discrete value-based control.

## GPU vector training and curriculum learning

The PyTorch Projected-PPO backend keeps physics, guidance, rewards, and action
projection in parallel CPU environment workers while batching policy inference
and PPO optimization on one GPU. Install a PyTorch build matching the worker's
CUDA runtime, then install the optional project dependencies:

```bash
python -m pip install -e '.[test,torch]'
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Run the staged curriculum with eight spawned simulation workers. `global_step`
counts transitions across all workers, so one 256-step rollout from eight
workers contributes 2,048 steps. `--total-steps` must be divisible by the
number of environments; the last rollout is shortened so training stops at the
requested transition count exactly:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src python scripts/train.py \
  --curriculum configs/curriculum/blue_escape.yaml \
  --actions configs/actions/blue_29.yaml \
  --algorithm configs/algorithm/ppo_projected_torch.yaml \
  --platform zdj --seed 0 --device cuda:0 \
  --num-envs 8 --env-backend subprocess \
  --worker-start-method spawn --total-steps 5000000 \
  --checkpoint-interval 100000 \
  --output-dir runs/projected_ppo_curriculum_gpu
```

The scheduler samples the weighted scenarios in the active stage, advances on
rolling success/survival criteria (or a configured maximum-step limit), retains
easier scenarios in later stages, and stores its stage, counters, recent
outcomes, RNG, and next episode id in every checkpoint. Resume with the same
algorithm and curriculum files plus `--resume`; the final `--total-steps` is the
desired aggregate transition count.

For deterministic batched evaluation, load the model once on the GPU and run a
fixed suite across parallel CPU workers:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src python scripts/evaluate.py \
  --evaluation-suite configs/evaluation/full_suite.yaml \
  --actions configs/actions/blue_29.yaml \
  --algorithm configs/algorithm/ppo_projected_torch.yaml \
  --checkpoint runs/projected_ppo_curriculum_gpu/checkpoints/latest.pt \
  --platform zdj --device cuda:0 --num-envs 8 \
  --env-backend subprocess --deterministic \
  --output-dir runs/eval_projected_ppo_curriculum_gpu
```

Use `--device cpu --env-backend serial` for deterministic debugging without a
GPU. CUDA is initialized only in the learner/evaluator process; spawned workers
never own model replicas. Training writes curriculum transitions to
`curriculum.jsonl`, hardware and parallelism metadata to `manifest.json`, and
atomic PyTorch checkpoints containing model, optimizer, AMP scaler, RNG, and
curriculum state. Checkpoint schema v3 also binds the action catalog,
projection configuration, vector-environment assignments, and curriculum file;
the earlier experimental v2 Torch checkpoints are rejected rather than resumed
with silently different training semantics.
