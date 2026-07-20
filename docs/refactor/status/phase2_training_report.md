# Phase 2 Training Runtime Status

Implemented the phase 2 scope for the primary `ppo_projected` training path plus `ppo_discrete` and `rainbow_dqn` controls.

## Audit summary

1. Projected PPO outputs 3 dimensions.
2. It uses a diagonal Gaussian over pre-tanh continuous actions.
3. It uses `tanh`; log-prob includes the tanh Jacobian correction.
4. The bounded action maps to `[nx, nf, gamma_s]` through `ContinuousCommandProjector` bounds and platform/safety projection.
5. Projection now uses weighted normalized command distance over `nx`, `nf`, and wrapped `gamma_s`.
6. `action_mask` filters projection candidates.
7. `old_log_prob` is stored from the sampled continuous pre-tanh action.
8. The PPO ratio is not computed from final `action_id`.
9. Discrete PPO now has masked 29-logit sampling, rollout, GAE, clipped update, and deterministic masked argmax.
10. The runtime builder returns a unified `(name, env, policy, trainer)` runtime; only Projected PPO wraps the environment with `ProjectedContinuousActionWrapper`.

## Scope exclusions

Batch evaluation, plotting, and ACMI were intentionally not implemented in this phase.
