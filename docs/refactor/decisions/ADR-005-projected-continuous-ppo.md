# ADR-005: Projected Continuous PPO alongside Rainbow DQN

## Status
Accepted for M5 after blocking-fix update.

## Decision
Keep `BlueEscapeEnv` as the canonical 29-action discrete environment for Rainbow DQN and add a separate `ProjectedContinuousActionWrapper` for PPO. PPO samples a continuous Gaussian raw action, applies `tanh`, scales to `[nx, nf, gamma_s]`, projects the physical command with state/platform constraints, maps it to the nearest legal maneuver by current-state 3-DoF effect distance, and finally calls the same discrete environment `step(action_id)`.

## Training separation
`algorithm.name` is selected once before the experiment starts:

- `rainbow_dqn` builds the base discrete environment, a Q network, target network, prioritized replay buffer, and `RainbowDQNTrainer`.
- `ppo_projected` builds the projected wrapper, 3-D Gaussian actor-critic, independent rollout buffer, and `PPOProjectedTrainer`.

There is no episode-internal algorithm switching, no alternating control, and no shared training buffer.

## PPO action and log-probability
The actor emits a 3-D diagonal Gaussian over raw pre-tanh variables. PPO stores the raw continuous action and corrected log-probability in the rollout buffer. Optimization reevaluates log-probability with the same raw continuous action via `evaluate_actions()`; it never computes PPO log-probability from the final discrete `executed_action_id`.

## Projection and mapping
Continuous constraints live in `ContinuousCommandProjector`, not in the actor network. The projector applies numeric bounds, platform g-limits (`zdj` about 9g, `yjj` about 3g), low-speed protection, high-speed acceleration limiting, altitude/safety protection, and then returns a projected `ManeuverCommand`.

`NearestManeuverMapper` computes the current-state effect vector:

```text
effect = [dV/dt, dgamma/dt, dpsi/dt]
```

and selects the legal action minimizing the normalized weighted distance:

```text
distance =
  w_v     * ((dV_i     - dV_c)     / scale_v)^2
+ w_gamma * ((dgamma_i - dgamma_c) / scale_gamma)^2
+ w_psi   * ((dpsi_i   - dpsi_c)   / scale_psi)^2
```

Ties choose the smallest `action_id`; empty legal sets use safe fallback action 0.

## Consequences
- Rainbow DQN remains discrete and can load untagged legacy Rainbow checkpoint payloads.
- PPO checkpoints validate `algorithm_name=ppo_projected` and action dimension 3.
- PPO logs continuous action statistics, bound-hit frequency, projection distance mean/max, executed action frequency, switch rate, and fallback count.
- Both algorithms still share scenarios, observation construction, rewards, action catalog, outcomes, and evaluation-facing metrics.
