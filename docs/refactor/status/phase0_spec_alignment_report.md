# Phase 0 Specification Alignment and Implementation Baseline Report

## Scope

Phase 0 reviewed the current repository and updated only specification, ADR, README architecture text, and this status report. No training, simulation, environment, algorithm, or script source code was modified in this phase.

## Reviewed files and areas

- `docs/refactor/00_global_constraints.md`.
- All `docs/refactor/` specifications, ADRs, decisions, and status reports.
- `README.md`.
- Algorithm configs under `configs/algorithm/`.
- `src/air_combat_rl/training/runner.py` and `build_algorithm_runtime()`.
- `ProjectedContinuousActionWrapper` and continuous projection/mapping modules.
- Projected PPO modules under `src/air_combat_rl/algorithms/ppo/`.
- Rainbow DQN modules under `src/air_combat_rl/algorithms/rainbow/`.
- Application entrypoints `scripts/run_scenario.py`, `scripts/train.py`, and `scripts/evaluate.py`.
- I/O placeholders `trajectory_writer.py` and `acmi_writer.py`.
- Evaluation placeholders `evaluation/metrics.py` and `evaluation/report.py`.

## Specification updates made

- Updated global constraints to state that `ppo_projected` is the primary/default algorithm.
- Documented `ppo_discrete` as the discrete PPO comparison baseline and `rainbow_dqn` as the discrete DQN comparison baseline.
- Documented the projected PPO action chain from continuous policy sample through `[nx, nf, gamma_s]`, legal action-mask projection, final `action_id`, and `BlueEscapeEnv.step(action_id)`.
- Documented that projected PPO log probabilities and PPO ratios must be computed from the continuous sample before projection, not from the executed `action_id`.
- Updated README architecture and algorithm descriptions to match the latest algorithm decision.
- Added ADR-006 recording the rationale, responsibilities, non-differentiable projection boundary, log-probability requirement, projection-distance logging, retained baselines, fair-comparison contract, and risks.

## Baseline findings by category

### Fully implemented

- XZY coordinate convention and 3-DoF dynamics are documented by previous status reports and source modules.
- The 29-action catalog and platform action masks exist, with `configs/actions/blue_29.yaml` as the action source.
- `BlueEscapeEnv` exists as the canonical discrete 29-action task environment.
- Scenario config loading and named powered/terminal scenario YAML files exist.
- `SimulationWorld.step_held_policy_interval()` and `SimulationWorld.snapshot()` exist.
- Projected PPO core pieces exist: continuous wrapper, projector, nearest maneuver mapper, actor-critic, rollout buffer, loss, checkpoint helpers, and trainer.
- Rainbow DQN core pieces exist: Q network, target network handling, prioritized replay, policy adapter, trainer, and checkpoint compatibility helpers.

### Partially implemented

- `build_algorithm_runtime()` supports `rainbow_dqn` and `ppo_projected`, but its default is currently `rainbow_dqn`, conflicting with the updated primary/default `ppo_projected` specification.
- Projected PPO stores continuous actions/log probabilities and executed action ids, but end-to-end logging/checkpoint/report integration remains incomplete.
- Evaluation package structure exists, but actual metrics/report generation are placeholders.
- I/O package structure exists, but trajectory and ACMI writers are placeholders.
- Algorithm configs exist for projected PPO and Rainbow DQN, but their integration into runnable scripts is incomplete.

### Placeholder implementation

- `scripts/run_scenario.py` contains only a placeholder docstring.
- `scripts/train.py` contains only a placeholder docstring.
- `scripts/evaluate.py` contains only a placeholder docstring.
- `src/air_combat_rl/io/trajectory_writer.py` contains only a placeholder docstring.
- `src/air_combat_rl/io/acmi_writer.py` contains only a placeholder docstring.
- `src/air_combat_rl/evaluation/metrics.py` contains only a placeholder docstring.
- `src/air_combat_rl/evaluation/report.py` contains only a placeholder docstring.

### Interface exists but is not wired end-to-end

- `ProjectedContinuousActionWrapper` exists and calls the base environment with the mapped discrete action, but scripts and unified train/evaluate flows do not expose it end-to-end.
- `training.runner.build_algorithm_runtime()` exists, but runnable CLI entrypoints do not yet call it.
- Evaluation and I/O module names exist, but no implemented reporting pipeline consumes trainer/evaluator outputs.

### Documentation and code conflicts

- Previous global constraints did not state the now-confirmed primary/default `ppo_projected` decision; this phase corrected that documentation.
- README previously described the algorithm layer mainly as discrete PPO; this phase corrected it to include `ppo_projected`, `ppo_discrete`, and `rainbow_dqn` roles.
- Current code default in `build_algorithm_runtime()` is `rainbow_dqn`, which conflicts with the new specification default of `ppo_projected`; this must be fixed in a later code phase, not in phase 0.
- `configs/algorithm/ppo_discrete.yaml` names `ppo_discrete`, but no corresponding runtime branch was found in `build_algorithm_runtime()`.

## Required follow-up by phase

### Phase 1 file range

- `scripts/run_scenario.py`.
- `src/air_combat_rl/io/trajectory_writer.py`.
- `src/air_combat_rl/io/acmi_writer.py`.
- Supporting tests for scenario execution and file export.

### Phase 2 file range

- `scripts/train.py`.
- `src/air_combat_rl/training/runner.py`.
- Algorithm config loading under `configs/algorithm/`.
- Logging/checkpoint modules under `src/air_combat_rl/io/` and algorithm checkpoint modules as needed.
- Tests for default `ppo_projected`, explicit `ppo_discrete`, and explicit `rainbow_dqn` runtime selection.

### Phase 3--5 file range

- `scripts/evaluate.py`.
- `src/air_combat_rl/evaluation/metrics.py`.
- `src/air_combat_rl/evaluation/report.py`.
- `src/air_combat_rl/evaluation/evaluator.py` and `scenario_suite.py` if needed.
- Experiment result presentation, report serialization, checkpoint evaluation loading, fairness checks, and regression/acceptance tests.

## Phase 0 validation

- Verified no code changes were needed or made for training, simulation, environment, algorithms, or scripts.
- Ran documentation-focused repository checks listed in the final response.

## Known limitations

- No code behavior was changed in phase 0 by design.
- The runtime default still needs code alignment to `ppo_projected` in a later phase.
- Discrete PPO remains a specified comparison path but is not wired into `build_algorithm_runtime()`.
- End-to-end scenario run, training, evaluation, trajectory export, ACMI export, metrics, and reports remain future-phase work.
