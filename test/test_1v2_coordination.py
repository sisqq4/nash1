"""1v2 协同规避对比测试：同向进攻 + 可控时间差 + 严格同 seed 对照。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scenario_test_utils import resolve_checkpoint_path, run_scenario_sweep_multi_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1v2 协同规避对比测试")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run-id", type=str, default="20260326_103719_1v2")
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_ep1000.pt")
    parser.add_argument("--episodes-per-scenario", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-mode", type=str, default="multi_coop")
    parser.add_argument("--blue-eval-policy", type=str, choices=["dqn", "bt"], default="bt")
    return parser.parse_args()


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_episode_initializer(
    first_dist_range: tuple[float, float],
    first_lateral_range: tuple[float, float],
    second_dist_range: tuple[float, float],
    second_lateral_range: tuple[float, float],
    launch_gap_range: tuple[float, float],
) -> Any:
    def _initializer(env: Any, episode_seed: int, _: int) -> None:
        rng = np.random.default_rng(int(episode_seed) + 7919)

        blue_pos = env.blue_pos.copy()
        blue_vel = env.blue_vel.copy()
        heading_xy = blue_vel[:2]
        h_norm = float(np.linalg.norm(heading_xy))
        if h_norm < 1e-8:
            heading_xy = np.array([1.0, 0.0], dtype=float)
            h_norm = 1.0
        h_hat_xy = heading_xy / h_norm
        lateral_xy = np.array([-h_hat_xy[1], h_hat_xy[0]], dtype=float)

        d1 = float(rng.uniform(first_dist_range[0], first_dist_range[1]))
        d2 = float(rng.uniform(second_dist_range[0], second_dist_range[1]))
        o1_mag = float(rng.uniform(first_lateral_range[0], first_lateral_range[1]))
        o2_mag = float(rng.uniform(second_lateral_range[0], second_lateral_range[1]))

        sign1 = 1.0 if rng.random() < 0.5 else -1.0
        sign2 = -sign1

        p1_xy = blue_pos[:2] - d1 * h_hat_xy + sign1 * o1_mag * lateral_xy
        p2_xy = blue_pos[:2] - d2 * h_hat_xy + sign2 * o2_mag * lateral_xy

        env.missile_pos[0, 0:2] = p1_xy
        env.missile_pos[1, 0:2] = p2_xy
        env.missile_pos[:, 2] = blue_pos[2]

        # 第一发为“诱导型”几何，第二发更“正”地压向目标。
        for i in range(2):
            rel = blue_pos - env.missile_pos[i]
            rel_norm = float(np.linalg.norm(rel))
            if rel_norm < 1e-8:
                rel = np.array([1.0, 0.0, 0.0], dtype=float)
                rel_norm = 1.0
            dir_vec = rel / rel_norm
            env.missile_vel[i] = dir_vec * env.missile_speed[i]

        gap = float(rng.uniform(launch_gap_range[0], launch_gap_range[1]))
        env.missile_launch_times = np.array([0.0, gap], dtype=float)

    return _initializer


def _build_scenarios() -> list[dict[str, object]]:
    # 三个关键变量扫描：发射时延、第一发几何劣化、蓝机可机动强度
    launch_delay_windows = [
        ("delay_06_08", (0.6, 0.8)),
        ("delay_09_11", (0.9, 1.1)),
        ("delay_13_15", (1.3, 1.5)),
    ]
    first_shot_geometry_levels = [
        ("first_mild_deg", (16.0, 18.0), (1.2, 2.2)),
        ("first_heavy_deg", (18.0, 21.0), (2.0, 3.4)),
    ]
    blue_accel_levels = [
        ("blue_mid_g", 0.085),
        ("blue_high_g", 0.100),
    ]

    blue_x = 18.0
    base = {
        "num_missiles": 2,
        "missile_update_dt": 0.01,
        "blue_x_min": blue_x,
        "blue_x_max": blue_x,
        "blue_y_min": 0.0,
        "blue_y_max": 0.0,
        "blue_z_min": 10.0,
        "blue_z_max": 10.0,
        "red_launch_z_min": 10.0,
        "red_launch_z_max": 10.0,
        "red_launch_x_min": 0.0,
        "red_launch_x_max": 0.0,
        "red_launch_y_min": 0.0,
        "red_launch_y_max": 0.0,
    }

    scenarios: list[dict[str, object]] = []
    for delay_tag, delay_window in launch_delay_windows:
        for geom_tag, first_dist_range, first_lateral_range in first_shot_geometry_levels:
            for accel_tag, blue_accel in blue_accel_levels:
                seed_group = f"{delay_tag}_{geom_tag}_{accel_tag}"
                initializer = _build_episode_initializer(
                    first_dist_range=first_dist_range,
                    first_lateral_range=first_lateral_range,
                    second_dist_range=(12.0, 14.0),
                    second_lateral_range=(0.2, 0.8),
                    launch_gap_range=delay_window,
                )
                for coordination in ["none", "strategy1"]:
                    scenario_name = f"{seed_group}_{coordination}"
                    scenarios.append(
                        {
                            "scenario_name": scenario_name,
                            "family": "same_direction_time_lag",
                            "sub_type": f"{delay_tag}|{geom_tag}|{accel_tag}",
                            "coordination": coordination,
                            "seed_group": seed_group,
                            "episode_initializer": initializer,
                            "env_overrides": {
                                **base,
                                "blue_accel": blue_accel,
                                "missile_coordination_strategy": coordination,
                            },
                        }
                    )
    return scenarios


def _plot_diagnostics(step_rows: list[dict[str, float]], out_dir: Path) -> None:
    metrics = ["corridor_width", "tgo_std", "active_missiles"]
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in step_rows:
        key = str(row["scenario_name"])
        if int(row["episode"]) == 1:
            grouped[key].append(row)

    for scenario_name, rows in grouped.items():
        rows = sorted(rows, key=lambda x: int(x["step"]))
        t = [float(r["time"]) for r in rows]
        for metric in metrics:
            y = [float(r[metric]) for r in rows]
            plt.figure(figsize=(8.5, 4.8))
            plt.plot(t, y, linewidth=1.5)
            plt.xlabel("Time (s)")
            plt.ylabel(metric)
            plt.title(f"{scenario_name}: {metric} over time")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(out_dir / f"{scenario_name}_{metric}.png", dpi=150)
            plt.close()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        run_id=args.run_id,
        episode=args.episode,
        checkpoint_name=args.checkpoint_name,
    )

    scenarios = _build_scenarios()
    all_rows, step_rows = run_scenario_sweep_multi_diagnostics(
        checkpoint_path=checkpoint_path,
        output_root=str(Path("outputs") / "tests_1v2_coordination"),
        scenarios=scenarios,
        episodes_per_scenario=args.episodes_per_scenario,
        seed=args.seed,
        checkpoint_interval=10,
        report_interval=10,
        reward_mode=args.reward_mode,
        blue_eval_policy=args.blue_eval_policy,
        enable_step_diagnostics=True,
    )

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        scan_group = str(row["param_sub_type"])
        coord_mode = str(row.get("param_coordination", "none"))
        key = (scan_group, coord_mode)
        for k in [
            "win", "reward", "steps", "min_dist", "final_dist", "avg_speed", "hit", "timeout", "crashed", "missiles_exhausted",
            "first_missile_hit", "second_missile_hit",
            "threat_switch_count", "threat_id_jitter_rate", "corridor_width_mean", "corridor_width_min", "corridor_width_trend",
            "tgo_std_mean", "tgo_std_max", "degrade_to_1_time", "degrade_to_0_time",
            "coord_window_mean", "coord_window_min", "coord_window_error_mean", "coord_window_trend_mean",
            "coord_gain_scale_mean", "coord_activation_mean", "coord_intra_wave_error_mean",
            "coord_inter_wave_gap_error_mean", "coord_target_met_rate", "coord_target_met_time",
        ]:
            grouped[key][k].append(float(row[k]))

    result_rows: list[dict[str, float]] = []
    for (scan_group, coord_mode), vals in sorted(grouped.items()):
        result_rows.append(
            {
                "scan_group": scan_group,
                "coordination": coord_mode,
                "episodes": float(len(vals["win"])),
                "win_rate": _mean(vals["win"]),
                "hit_rate": _mean(vals["hit"]),
                "first_missile_hit_rate": _mean(vals["first_missile_hit"]),
                "second_missile_hit_rate": _mean(vals["second_missile_hit"]),
                "crash_rate": _mean(vals["crashed"]),
                "timeout_rate": _mean(vals["timeout"]),
                "missiles_exhausted_rate": _mean(vals["missiles_exhausted"]),
                "avg_reward": _mean(vals["reward"]),
                "avg_steps": _mean(vals["steps"]),
                "avg_min_dist": _mean(vals["min_dist"]),
                "avg_final_dist": _mean(vals["final_dist"]),
                "avg_speed": _mean(vals["avg_speed"]),
                "avg_threat_switch_count": _mean(vals["threat_switch_count"]),
                "avg_threat_id_jitter_rate": _mean(vals["threat_id_jitter_rate"]),
                "avg_corridor_width": _mean(vals["corridor_width_mean"]),
                "avg_min_corridor_width": _mean(vals["corridor_width_min"]),
                "avg_corridor_width_trend": _mean(vals["corridor_width_trend"]),
                "avg_tgo_std": _mean(vals["tgo_std_mean"]),
                "avg_tgo_std_max": _mean(vals["tgo_std_max"]),
                "avg_degrade_to_1_time": _mean(vals["degrade_to_1_time"]),
                "avg_degrade_to_0_time": _mean(vals["degrade_to_0_time"]),
                "avg_coord_window_mean": _mean(vals["coord_window_mean"]),
                "avg_coord_window_min": _mean(vals["coord_window_min"]),
                "avg_coord_window_error_mean": _mean(vals["coord_window_error_mean"]),
                "avg_coord_window_trend_mean": _mean(vals["coord_window_trend_mean"]),
                "avg_coord_gain_scale_mean": _mean(vals["coord_gain_scale_mean"]),
                "avg_coord_activation_mean": _mean(vals["coord_activation_mean"]),
                "avg_coord_intra_wave_error_mean": _mean(vals["coord_intra_wave_error_mean"]),
                "avg_coord_inter_wave_gap_error_mean": _mean(vals["coord_inter_wave_gap_error_mean"]),
                "avg_coord_target_met_rate": _mean(vals["coord_target_met_rate"]),
                "avg_coord_target_met_time": _mean(vals["coord_target_met_time"]),
            }
        )

    results_dir = Path("outputs") / "tests_1v2_coordination" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(results_dir / "result_1v2_coordination_compare.csv", result_rows)
    _plot_diagnostics(step_rows, results_dir)


if __name__ == "__main__":
    main()
