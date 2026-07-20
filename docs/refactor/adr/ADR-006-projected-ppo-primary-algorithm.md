# ADR-006 Projected PPO as the Primary Algorithm

## Status

Accepted for phase 0 specification alignment.

## Context

The blue-escape task has a fixed 29-action maneuver catalog and platform-dependent legal-action masks, while the learning objective should let the primary PPO policy express smooth maneuver intent in physical command space. Earlier refactor notes treated discrete PPO as the main PPO path; the current confirmed decision makes `ppo_projected` the primary/default algorithm and keeps discrete PPO and Rainbow DQN as comparison baselines.

## Decision

Use projected continuous PPO (`ppo_projected`) as the primary/default algorithm. The policy samples a continuous maneuver intent, bounds it, converts it to `[nx, nf, gamma_s]`, projects that command onto the currently legal discrete action set, and executes the resulting `action_id` in `BlueEscapeEnv`.

Discrete PPO (`ppo_discrete`) remains a categorical 29-action comparison baseline. Rainbow DQN (`rainbow_dqn`) remains a discrete DQN comparison baseline over the same action catalog.

## Responsibilities

### Continuous action responsibilities

- Represent policy intent in physical maneuver-command coordinates.
- Produce the continuous sample used for PPO storage, `old_log_prob`, later `new_log_prob`, and PPO ratio computation.
- Preserve smooth exploration and optimization signals before any non-differentiable projection.

### Discrete action responsibilities

- Preserve the canonical `BlueEscapeEnv.step(action_id)` interface.
- Enforce stable 29-action catalog semantics and platform legal-action masks.
- Define the actual maneuver command executed by the simulation after projection or categorical selection.

## Why projection may be non-differentiable

The PPO update is based on the likelihood ratio of the stochastic policy distribution that generated the stored continuous action. The nearest-legal-action projection is part of the environment/action-adapter path used to choose the executed discrete maneuver. PPO does not require gradients through `BlueEscapeEnv.step()` or through the discrete projection operation; it only requires consistent log-probability evaluation for the sampled action under old and new policy parameters.

## Why the PPO ratio uses the continuous sample

For `ppo_projected`, the behavior policy samples from a continuous distribution. Therefore `old_log_prob`, `new_log_prob`, and `ratio = exp(new_log_prob - old_log_prob)` must all correspond to that continuous sample. The final `action_id` is a deterministic or fallback result of projection and masking, not a sample from the Gaussian policy. Using `action_id` as if it were a continuous Gaussian action would make the PPO objective mathematically inconsistent and would couple the ratio to an action chosen by a non-differentiable adapter rather than by the policy distribution.

## Why record projection distance

Projection distance is an observability and diagnostics signal. It shows how far the continuous intent is from the nearest legal discrete maneuver under current state/platform constraints, helps identify saturation or platform-mask problems, and supports fair interpretation of learned policies whose continuous intent frequently collapses to a small set of discrete actions. It should be logged alongside executed action frequencies, fallback counts, and action-switch rates.

## Why retain discrete PPO and Rainbow DQN

- `ppo_discrete` isolates the effect of continuous intent/projection by using the same PPO family with categorical 29-action sampling.
- `rainbow_dqn` preserves the existing discrete DQN runtime and checkpoint compatibility as an off-policy value-based comparison.
- Keeping both baselines helps distinguish improvements from algorithm family, action parameterization, projection behavior, and implementation maturity.

## Fair comparison contract

The three algorithms must use the same scenario suites, observation schema, rewards, 29-action catalog, platform action masks, termination/truncation semantics, rollout/evaluation seeds, and reported metrics wherever possible. Differences that are intrinsic to the algorithm, such as replay buffer versus on-policy rollout buffer or continuous versus categorical log-probabilities, must be explicitly reported rather than hidden in task configuration changes.

## Current implementation risks to verify

- Runtime default selection still needs code alignment because current implementation inspection found `build_algorithm_runtime()` defaulting to `rainbow_dqn`.
- `ppo_discrete` has configuration presence but no confirmed trainer/runtime implementation.
- `ppo_projected` uses a lightweight NumPy actor/trainer path and needs validation against longer training runs, checkpoint/resume, and metric logging requirements.
- Application entrypoints and evaluation/report writers are placeholders or only partially connected, so end-to-end experiment reproducibility remains unverified.
- Projection behavior must continue to be tested for legal-action-only selection, fallback handling, and consistency between logged continuous samples and executed discrete actions.
