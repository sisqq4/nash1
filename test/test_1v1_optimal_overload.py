"""1v1 最优过载测试：扫描不同蓝机过载(加速度)常数，比较逃逸表现。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scenario_test_utils import resolve_checkpoint_path, run_scenario_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1v1 最优过载选择测试")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run-id", type=str, default="20260316_105748")
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_ep1000.pt")
    parser.add_argument("--episodes-per-scenario", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-mode", type=str, default=None)
    parser.add_argument("--min-distance-km", type=int, default=6)
    parser.add_argument("--max-distance-km", type=int, default=30)
    parser.add_argument("--distance-step-km", type=int, default=1)
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

def _plot_overload(rows: list[dict[str, float]], out_dir: Path) -> None:
    if not rows:
        return
    metrics = ["win_rate", "avg_reward", "avg_steps", "avg_min_dist"]
    for metric in metrics:
        overload_to_points: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for r in rows:
            overload = int(r["blue_overload_g"])
            overload_to_points[overload].append((float(r["distance_km"]), float(r[metric])))
        plt.figure(figsize=(9, 5))
        for overload in sorted(overload_to_points):
            pts = sorted(overload_to_points[overload], key=lambda x: x[0])
            plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"{overload}g")
        plt.xlabel("Distance (km)")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Distance (grouped by overload)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_vs_distance_by_overload.png", dpi=150)
        plt.close()


def g_to_kmps2(overload_g: float) -> float:
    return overload_g * 9.8 / 1000.0


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        run_id=args.run_id,
        episode=args.episode,
        checkpoint_name=args.checkpoint_name,
    )

    overload_set = [6.0, 9.0, 12.0]
    distances = list(range(args.min_distance_km, args.max_distance_km + 1, args.distance_step_km))
    scenarios = []
    for distance in distances:
        for overload_g in overload_set:
            scenarios.append(
                {
                    "scenario_name": f"d{distance:02d}_blue_overload_{overload_g:.0f}g",
                    "distance_km": float(distance),
                    "blue_overload_g": overload_g,
                    "blue_accel_kmps2": g_to_kmps2(overload_g),
                    "env_overrides": {
                        "blue_x_min": float(distance),
                        "blue_x_max": float(distance),
                        "blue_y_min": 0.0,
                        "blue_y_max": 0.0,
                        "blue_z_min": 10.0,
                        "blue_z_max": 10.0,
                        "red_launch_x_min": 0.0,
                        "red_launch_x_max": 0.0,
                        "red_launch_y_min": 0.0,
                        "red_launch_y_max": 0.0,
                        "red_launch_z_min": 10.0,
                        "red_launch_z_max": 10.0,
                        "blue_accel": g_to_kmps2(overload_g),
                    },
                }
            )

    all_rows = run_scenario_sweep(
        checkpoint_path=checkpoint_path,
        output_root=str(Path("outputs") / "tests_1v1_optimal_overload"),
        scenarios=scenarios,
        episodes_per_scenario=args.episodes_per_scenario,
        seed=args.seed,
        checkpoint_interval=10,
        report_interval=10,
        reward_mode=args.reward_mode,
    )

    grouped: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        distance = int(float(row["param_distance_km"]))
        overload = int(float(row["param_blue_overload_g"]))
        key = (distance, overload)
        grouped[key]["win"].append(float(row["win"]))
        grouped[key]["reward"].append(float(row["reward"]))
        grouped[key]["steps"].append(float(row["steps"]))
        grouped[key]["min_dist"].append(float(row["min_dist"]))

    result_rows: list[dict[str, float]] = []
    for (distance, overload), vals in sorted(grouped.items()):
        result_rows.append(
            {
                "distance_km": float(distance),
                "blue_overload_g": float(overload),
                "episodes": float(len(vals["win"])),
                "win_rate": _mean(vals["win"]),
                "avg_reward": _mean(vals["reward"]),
                "avg_steps": _mean(vals["steps"]),
                "avg_min_dist": _mean(vals["min_dist"]),
            }
        )

    results_dir = Path("outputs") / "tests_1v1_optimal_overload" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(results_dir / "result_overload_distance_scan.csv", result_rows)
    _plot_overload(result_rows, results_dir)


if __name__ == "__main__":
    main()
