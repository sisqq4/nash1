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
