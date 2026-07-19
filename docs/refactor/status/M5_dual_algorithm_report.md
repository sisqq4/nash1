# M5 Dual Algorithm Status Report

## 1. 新增和修改文件

- `src/air_combat_rl/interfaces/policy.py`: added `PolicyDecision` for shared DQN/PPO decision metadata.
- `src/air_combat_rl/tasks/blue_escape/continuous/projector.py`: added continuous physical scaling/projection, including `zdj`/`yjj`, low-speed, high-speed, and low-altitude constraints.
- `src/air_combat_rl/tasks/blue_escape/continuous/mapper.py`: added nearest legal maneuver mapping with normalized weighted 3-DoF effect distance.
- `src/air_combat_rl/tasks/blue_escape/continuous/wrapper.py`: added PPO-facing continuous wrapper that calls the base discrete environment and enriches `info`.
- `src/air_combat_rl/algorithms/ppo/actor_critic.py`: added 3-D continuous Gaussian actor, corrected tanh log-probability, `evaluate_actions()`, and scalar critic.
- `src/air_combat_rl/algorithms/ppo/rollout_buffer.py`: stores raw continuous actions, corrected log-probs, values, terminated/truncated flags, episode starts, projected actions, executed discrete ids, and projection distances.
- `src/air_combat_rl/algorithms/ppo/trainer.py`: added independent `PPOProjectedTrainer`, rollout collection, GAE/returns, continuous-action log-prob reevaluation, clipped objective update, gradient clipping, KL stats, and action/projection logging.
- `src/air_combat_rl/algorithms/ppo/loss.py`: added PPO clipped objective stats.
- `src/air_combat_rl/algorithms/ppo/checkpoint.py`: saves and validates PPO metadata, optimizer state, observation normalization, global step, curriculum stage, random state, and config.
- `src/air_combat_rl/algorithms/rainbow/*`: added discrete Q network, target network, prioritized replay buffer, trainer, adapter, and legacy-compatible checkpoint loader.
- `src/air_combat_rl/training/runner.py`: config-time construction of independent `rainbow_dqn` or `ppo_projected` runtime paths.
- `configs/algorithm/rainbow_dqn.yaml` and `configs/algorithm/ppo_projected.yaml`: explicit algorithm configs.
- `tests/unit/test_m5_projected_ppo.py`: expanded M5 coverage for continuous PPO, projection/mapping, wrapper, trainers, buffers, checkpoints, and config switching.

## 2. PPO连续动作定义

PPO uses a feed-forward MLP actor-critic. The actor outputs a diagonal Gaussian over raw pre-tanh variables with dimension 3:

```text
raw_action = [raw_nx, raw_nf, raw_gamma_s]
squashed_action = tanh(raw_action)
physical_command = [nx, nf, gamma_s]
```

The raw sample is stored in the rollout buffer. The corrected log-probability is computed as Gaussian log-probability minus the tanh Jacobian correction. PPO optimization reevaluates log-probability from the stored raw continuous action, not from the mapped discrete action id.

## 3. 连续约束顺序

Continuous commands are processed in this order:

1. Numeric range clipping to configured physical bounds.
2. Platform capability clipping:
   - `zdj`: about 9g.
   - `yjj`: about 3g.
3. Low-speed protection: prevent strong deceleration and limit high normal load.
4. High-speed protection: limit positive acceleration above high-speed thresholds.
5. Current altitude and safety protection: suppress unsafe dive commands near low altitude.
6. Other projector-configured state constraints.

## 4. 最近机动距离公式

For the projected continuous command and every currently legal discrete action, the mapper computes the current-state 3-DoF effect vector:

```text
effect = [dV/dt, dgamma/dt, dpsi/dt]
```

with speed and `cos(gamma)` protection. The selected action minimizes:

```text
distance =
  w_v     * ((dV_i     - dV_c)     / scale_v)^2
+ w_gamma * ((dgamma_i - dgamma_c) / scale_gamma)^2
+ w_psi   * ((dpsi_i   - dpsi_c)   / scale_psi)^2
```

All scales and weights are configurable. Ties select the smallest `action_id`. Empty legal sets use safe fallback action 0.

## 5. PPO与DQN训练路径

- `algorithm.name: rainbow_dqn`: base `BlueEscapeEnv`, `RainbowQNetwork`, target network, `PrioritizedReplayBuffer`, `RainbowDQNPolicyAdapter`, and `RainbowDQNTrainer`.
- `algorithm.name: ppo_projected`: base `BlueEscapeEnv` wrapped by `ProjectedContinuousActionWrapper`, `PPOActorCritic`, independent `RolloutBuffer`, and `PPOProjectedTrainer`.

Algorithms are selected before experiment start. There is no same-episode dynamic switching, no alternating control, and no shared update buffer.

## 6. 旧DQN兼容情况

The Rainbow checkpoint loader accepts untagged legacy pickle payloads and checkpoints marked as `rainbow_dqn`/`RainbowDQN`; it rejects checkpoints explicitly marked as a different algorithm. The Rainbow path remains discrete and stores DQN transitions in prioritized replay.

## 7. 日志和比较指标

PPO trainer now summarizes:

- `continuous_action_mean`;
- `continuous_action_std`;
- `bounded_action_frequency`;
- `projection_distance_mean`;
- `projection_distance_max`;
- `executed_action_frequency`;
- `action_switch_rate`;
- `fallback_count`.

Both algorithms continue to use the same environment outcome, reward components, scenario construction, and evaluation-facing episode metrics.

## 8. 测试命令和结果

Executed locally:

```bash
python -m compileall -q src tests
pytest -q
git diff --check
```

Result:

```text
50 passed
```

Expanded M5 tests cover continuous Gaussian actor output, tanh range and log-prob correction, `zdj`/`yjj` projection, low-speed/low-altitude/high-speed constraints, effect vector mapping, normalized distance, legal-action filtering, deterministic tie-break, fallback, wrapper-to-discrete-env execution, PPO rollout fields, PPO trainer update, Rainbow prioritized replay trainer update, old Rainbow checkpoint loading, PPO checkpoint metadata validation, config switching, and independent buffers.

## 9. 尚未解决的风险

1. The NumPy PPO trainer is intentionally lightweight for M5 smoke training and interface validation; future work may replace finite-difference updates with an autograd backend for performance.
2. Rainbow implementation is minimal but preserves the discrete network/target/replay/checkpoint/trainer interfaces required for this refactor stage.
3. Projection thresholds should still be calibrated against higher-fidelity aircraft envelopes.

## 10. git diff摘要

Blocking fixes added true independent PPO/Rainbow trainers and buffers, continuous-action log-prob reevaluation in PPO optimization, high-speed constraints, action/projection logging, checkpoint coverage, expanded tests, and updated M5 documentation.
