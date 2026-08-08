# Air Combat RL

This repository is structured as a layered blue-escape research codebase.

## Quick start

Install Python 3.10+ and all runtime/test dependencies:

```bash
python -m pip install -e '.[test]'
```

## Architecture

1. **Domain simulation kernel** (`air_combat_rl.core`, `air_combat_rl.domain`, `air_combat_rl.simulation`) owns coordinates, units, physical state, 3-DoF dynamics, guidance, propulsion, collision checks, scenarios, events, and snapshots.
2. **Task layer** (`air_combat_rl.tasks.blue_escape`) adapts the simulation kernel into a blue escape RL task with replaceable actions, observations, rewards, masks, and termination semantics.
3. **Algorithm layer** (`air_combat_rl.algorithms`) contains scene-agnostic learning code for the supported training schemes.
4. **Application layer** (`air_combat_rl.training`, `air_combat_rl.evaluation`, `scripts`) wires configs, runners, evaluators, logging, and checkpointing.

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

Run one reproducible scenario without starting training or batch evaluation:

```bash
PYTHONPATH=src python scripts/run_scenario.py \
  --scenario configs/scenario/fixed_1v1.yaml \
  --actions configs/actions/blue_29.yaml \
  --platform zdj \
  --policy constant \
  --action-id 0 \
  --seed 0 \
  --output-dir runs/fixed_1v1_constant
```

The command writes `manifest.json`, streaming `steps.jsonl`, and `episode_summary.json`. Use `--policy random_valid` to sample only currently legal masked actions with the supplied seed. A non-empty output directory is rejected unless `--overwrite` is passed.

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

The exporter maps simulation `x` to north/latitude, `z` to east/longitude, and
`y` to altitude above the supplied origin. It writes stable blue/missile IDs,
heading from `psi`, pitch from `gamma`, object removal, and supported hit and
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
