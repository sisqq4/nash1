# Air Combat RL

This repository is structured as a layered blue-escape research codebase.

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
