from __future__ import annotations

import numpy as np

from config import EnvConfig
from env.escape_env import EscapeEnv


def _build_env(strategy: str) -> EscapeEnv:
    cfg = EnvConfig(
        num_missiles=3,
        missile_coordination_strategy=strategy,
        log_trajectories=False,
    )
    env = EscapeEnv(cfg, seed=7)
    env.reset()
    env.blue_pos = np.array([20.0, 0.0, 10.0], dtype=float)
    env.blue_vel = np.array([0.28, 0.02, 0.0], dtype=float)
    env.missile_launched[:] = True
    env.missile_alive[:] = True
    env.missile_pos = np.array(
        [
            [0.0, -3.0, 10.0],
            [0.0, 0.0, 10.0],
            [0.0, 3.0, 10.0],
        ],
        dtype=float,
    )
    env.missile_vel = np.array(
        [
            [0.40, 0.12, 0.0],
            [0.42, 0.00, 0.0],
            [0.40, -0.12, 0.0],
        ],
        dtype=float,
    )
    env.missile_launch_times = np.array([0.0, 0.6, 1.2], dtype=float)
    return env


def test_coordination_strategy_switchable() -> None:
    guidance_active = np.ones(3, dtype=bool)
    env_none = _build_env("none")
    env_s1 = _build_env("strategy1")

    bias_none = env_none._compute_coordination_bias(guidance_active, missile_dt=0.01)
    bias_s1 = env_s1._compute_coordination_bias(guidance_active, missile_dt=0.01)

    assert np.allclose(bias_none, 0.0)
    assert np.linalg.norm(bias_s1) > 0.0


def test_dynamic_tgo_estimation_state_updates() -> None:
    env = _build_env("strategy1")
    guidance_active = np.ones(3, dtype=bool)

    _ = env._compute_coordination_bias(guidance_active, missile_dt=0.05)
    first_tgo = env.missile_tgo_hat.copy()

    env.blue_pos = env.blue_pos + np.array([0.2, 0.0, 0.0], dtype=float)
    _ = env._compute_coordination_bias(guidance_active, missile_dt=0.05)
    second_tgo = env.missile_tgo_hat.copy()

    finite_mask = np.isfinite(second_tgo)
    assert np.any(finite_mask)
    assert not np.allclose(first_tgo[finite_mask], second_tgo[finite_mask])


def test_window_feedback_enters_closed_loop() -> None:
    env = _build_env("strategy1")
    env.cfg.missile_strategy1_w_star = 0.01
    guidance_active = np.ones(3, dtype=bool)

    _ = env._compute_coordination_bias(guidance_active, missile_dt=0.05)
    assert np.isfinite(env.coordination_window_est)
    assert env.coordination_window_error > 0.0
    assert env.coordination_gain_scale > 0.0

    env.cfg.missile_strategy1_w_star = 1000.0
    _ = env._compute_coordination_bias(guidance_active, missile_dt=0.05)
    assert env.coordination_gain_scale == 0.0


def test_bias_constraints_and_smoothing() -> None:
    env = _build_env("strategy1")
    env.cfg.missile_coordination_max_bias_accel = 0.05
    env.cfg.missile_coordination_bias_rate_limit = 0.1
    env.cfg.missile_coordination_activation_tau = 0.2
    guidance_active = np.ones(3, dtype=bool)

    bias_1 = env._compute_coordination_bias(guidance_active, missile_dt=0.05)
    # 法向约束：偏置应近似垂直于速度方向
    for i in range(3):
        v = env.missile_vel[i]
        if np.linalg.norm(v) > 1e-8:
            assert abs(float(np.dot(bias_1[i], v))) < 1e-4

    # 触发收缩为零后，偏置应平滑衰减，单步变化受速率限制。
    env.cfg.missile_strategy1_w_star = 1e6
    bias_2 = env._compute_coordination_bias(guidance_active, missile_dt=0.05)
    max_delta = env.cfg.missile_coordination_bias_rate_limit * 0.05 + 1e-8
    for i in range(3):
        delta = float(np.linalg.norm(bias_2[i] - bias_1[i]))
        assert delta <= max_delta


def test_coordination_diagnostics_are_updated() -> None:
    env = _build_env("strategy1")
    guidance_active = np.ones(3, dtype=bool)
    _ = env._compute_coordination_bias(guidance_active, missile_dt=0.05)

    assert np.isfinite(env.coordination_window_est)
    assert np.isfinite(env.coordination_window_error)
    assert np.isfinite(env.coordination_window_trend)
    assert np.isfinite(env.coordination_intra_wave_error)
    assert np.isfinite(env.coordination_inter_wave_gap_error)


def test_multi_wave_cumulative_offsets_follow_launch_sequence() -> None:
    env = _build_env("strategy1")
    launch_times = np.array([0.0, 0.1, 1.0, 1.15, 2.7], dtype=float)
    wave_idx, wave_offsets = env._compute_wave_index_and_cumulative_offsets(launch_times)

    assert np.array_equal(wave_idx, np.array([0, 0, 1, 1, 2], dtype=int))
    # 严格累计: 第3波偏移应为 (1.0-0.0) + (2.7-1.0) = 2.7
    assert np.isclose(wave_offsets[4], 2.7, atol=1e-6)


def test_tgo_estimation_requires_closing_condition() -> None:
    env = _build_env("strategy1")
    guidance_active = np.ones(3, dtype=bool)
    # 令第一枚导弹径向发散，触发 r_dot >= 0 条件
    env.missile_vel[0] = np.array([-0.3, 0.0, 0.0], dtype=float)

    _ = env._compute_coordination_bias(guidance_active, missile_dt=0.05)
    assert np.isclose(env.missile_tgo_hat[0], env.cfg.missile_max_flight_time)


def test_explicit_wave_gap_sequence_overrides_observed_delta() -> None:
    env = _build_env("strategy1")
    env.cfg.missile_strategy1_wave_gap_sequence = (0.5, 1.2)
    launch_times = np.array([0.0, 0.1, 1.0, 1.15, 2.7], dtype=float)
    wave_idx, wave_offsets = env._compute_wave_index_and_cumulative_offsets(launch_times)

    assert np.array_equal(wave_idx, np.array([0, 0, 1, 1, 2], dtype=int))
    # 累计偏移应来自显式序列: [0.5, 1.2] => 第3波累计 1.7
    assert np.isclose(wave_offsets[4], 1.7, atol=1e-6)


def test_individual_coordination_gain_increases_with_error() -> None:
    env = _build_env("strategy1")
    env.cfg.missile_coordination_gain_error_weight = 0.8
    e_sync = np.array([0.1, 0.5, 1.0], dtype=float)
    gain_i = env._compute_individual_coordination_gain(e_sync=e_sync, gain_t=2.0)

    assert np.all(gain_i > 0.0)
    assert gain_i[2] > gain_i[1] > gain_i[0]


def test_target_accel_compensation_is_lateral_and_weighted() -> None:
    env = _build_env("strategy1")
    env.cfg.missile_coordination_target_accel_comp_weight = 0.5
    missile_vel = np.array([0.4, 0.0, 0.0], dtype=float)
    target_acc = np.array([0.2, 0.3, 0.1], dtype=float)
    comp = env._compute_target_accel_compensation(missile_vel, target_acc)

    # 与速度方向正交（仅法向补偿）
    assert np.isclose(float(np.dot(comp, missile_vel)), 0.0, atol=1e-8)
    # x 分量被剔除，y/z 分量按权重缩放
    assert np.isclose(comp[0], 0.0, atol=1e-8)
    assert np.isclose(comp[1], 0.15, atol=1e-8)
    assert np.isclose(comp[2], 0.05, atol=1e-8)
