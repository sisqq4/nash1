# 00 Global Constraints

## Layering and ownership

- Dynamics must not import PPO, reward, training, replay buffer, episode, or neural-network tensor code.
- Algorithms must interact only through standard environment/policy interfaces, observations, action masks, rewards, termination/truncation flags, and info dictionaries.
- Rewards and logging are read-only with respect to physical state.
- Scenarios define initial conditions and distributions; dynamics define time evolution.

## Coordinate and time conventions

- The canonical coordinate serialization is XZY: `[x, z, y]`, with `y` as altitude.
- Full flight-state serialization is `[x, z, y, V, gamma, psi]`.
- `psi = atan2(vz, vx)`; increasing `psi` turns from `+x` toward `+z`.
- `gamma = atan2(vy, sqrt(vx^2 + vz^2))`; increasing `gamma` climbs.
- Height, low-altitude protection, ground collision, and safety-altitude calculations must use `y`, not `z`.
- Default time scales are configurable `physics_dt = 0.005 s` and `policy_dt = 0.1 s`, so one policy action is normally held for 20 physics substeps.

## Algorithm roles

The project supports three algorithms over the same blue-escape task. The current primary and default algorithm is `ppo_projected`.

- `ppo_projected`: primary/default projected continuous PPO.
- `ppo_discrete`: discrete PPO comparison baseline.
- `rainbow_dqn`: discrete DQN comparison baseline using the original 29-action environment.

All three algorithms must share the same scenario definitions, observations, rewards, 29-action catalog, platform action masks, termination/truncation semantics, and evaluation metrics unless a later ADR explicitly records a justified exception.

## Projected PPO action semantics

`ppo_projected` follows this action chain:

```text
observation
→ continuous policy distribution
→ sampled continuous action
→ bounded continuous action
→ physical maneuver command [nx, nf, gamma_s]
→ projection onto the currently legal action_mask set
→ action_id
→ BlueEscapeEnv.step(action_id)
```

- Projected PPO outputs continuous maneuver intent, not a categorical action id.
- The bounded continuous action is converted to the physical maneuver command `[nx, nf, gamma_s]`.
- Projection may select only actions that are legal under the current `action_mask`; platform-illegal maneuvers must be masked out before selection rather than silently relied on as dynamics clipping.
- The final `action_id` is the discrete action actually executed by `BlueEscapeEnv`.
- PPO `old_log_prob`, `new_log_prob`, and ratio must be computed from the continuous policy's actual sampled action before discrete projection.
- The executed `action_id` must never replace the continuous sample when computing Gaussian or other continuous-distribution log probabilities.
