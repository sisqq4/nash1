"""Plots built exclusively from persisted Phase 1/3 artifacts.

Matplotlib is imported lazily so simulation and evaluation remain usable if the
optional presentation layer fails to initialize.
"""
from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


class PlotDataError(ValueError):
    """A requested plot cannot be made from the supplied artifacts."""


BLUE = "#1565c0"
MISSILES = ("#d32f2f", "#ef6c00", "#7b1fa2", "#00838f", "#6d4c41")
OUTCOME_COLORS = {"hit": "#d32f2f", "crash": "#6d4c41", "success": "#2e7d32", "exhausted": "#00838f", "timeout": "#757575", "running": "#f9a825"}
COMPARISON_ALGORITHMS = {"ppo_projected", "ppo_discrete", "rainbow_dqn"}


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise PlotDataError(f"required JSONL file does not exist: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise PlotDataError(f"cannot parse {path}: {exc}") from exc
    if not rows:
        raise PlotDataError(f"trajectory is empty: {path}")
    return rows


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    _plt().close(fig)
    if not path.is_file() or path.stat().st_size == 0:
        raise PlotDataError(f"plot was not written: {path}")
    return path


def _position(entity: dict) -> tuple[float, float, float]:
    position = entity.get("position_xzy_m")
    if not isinstance(position, list) or len(position) != 3:
        raise PlotDataError("world snapshot entity lacks position_xzy_m=[x,z,y]")
    return tuple(float(v) for v in position)


def _event_kinds(row: dict) -> set[str]:
    kinds = set()
    for event in row.get("events") or []:
        kinds.add(str(event.get("kind", event) if isinstance(event, dict) else event).lower())
    outcome = str(row.get("outcome", "")).lower()
    if outcome and outcome != "running":
        kinds.add(outcome)
    return kinds


def _mark_events(ax, rows: list[dict], x_values: list[float], y_values: list[float]) -> None:
    labels = set()
    aliases = (("hit", "hit"), ("closest", "closest approach"), ("ground", "ground collision"), ("crash", "ground collision"), ("timeout", "timeout"))
    for i, row in enumerate(rows):
        kinds = _event_kinds(row)
        for needle, label in aliases:
            if any(needle in kind for kind in kinds) and label not in labels:
                ax.scatter([x_values[i]], [y_values[i]], marker="X", s=55, label=label, zorder=5)
                labels.add(label)


def plot_run(run_dir: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    """Generate single-episode trajectory/reward plots from ``steps.jsonl``."""
    run_dir = Path(run_dir)
    rows = _jsonl(run_dir / "steps.jsonl")
    snapshots = [row.get("world_snapshot") for row in rows]
    if any(not isinstance(snapshot, dict) for snapshot in snapshots):
        raise PlotDataError("steps.jsonl does not contain world_snapshot records")
    out = Path(output_dir) if output_dir else run_dir / "plots"
    times = [float(row.get("time_s", i)) for i, row in enumerate(rows)]
    blue = [_position(snapshot["blue"]) for snapshot in snapshots]
    missile_ids = []
    for snapshot in snapshots:
        for i, missile in enumerate(snapshot.get("missiles") or []):
            mid = str(missile.get("id", f"missile_{i}"))
            if mid not in missile_ids:
                missile_ids.append(mid)
    missile_tracks = {mid: [] for mid in missile_ids}
    for snapshot in snapshots:
        current = {str(m.get("id", f"missile_{i}")): m for i, m in enumerate(snapshot.get("missiles") or [])}
        for mid in missile_ids:
            missile_tracks[mid].append(_position(current[mid]) if mid in current else None)

    plt = _plt(); made = []
    fig, ax = plt.subplots()
    ax.plot([p[0] for p in blue], [p[1] for p in blue], color=BLUE, label="blue aircraft")
    for i, (mid, track) in enumerate(missile_tracks.items()):
        valid = [p for p in track if p is not None]
        ax.plot([p[0] for p in valid], [p[1] for p in valid], color=MISSILES[i % len(MISSILES)], label=mid)
    _mark_events(ax, rows, [p[0] for p in blue], [p[1] for p in blue])
    ax.set(xlabel="x / north (m)", ylabel="z / east (m)", title="Horizontal trajectory (x-z)"); ax.axis("equal"); ax.legend()
    made.append(_save(fig, out / "trajectory_horizontal.png"))

    fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
    ax.plot([p[0] for p in blue], [p[1] for p in blue], [p[2] for p in blue], color=BLUE, label="blue aircraft")
    for i, (mid, track) in enumerate(missile_tracks.items()):
        valid = [p for p in track if p is not None]
        ax.plot([p[0] for p in valid], [p[1] for p in valid], [p[2] for p in valid], color=MISSILES[i % len(MISSILES)], label=mid)
    marked = set()
    for i, row in enumerate(rows):
        for needle, label in (("hit", "hit"), ("ground", "ground collision"), ("crash", "ground collision"), ("timeout", "timeout")):
            if label not in marked and any(needle in kind for kind in _event_kinds(row)):
                ax.scatter(*blue[i], marker="X", s=55, label=label)
                marked.add(label)
    ax.set(xlabel="x / north (m)", ylabel="z / east (m)", zlabel="y / altitude (m)", title="3D trajectory (x-z-y)"); ax.legend()
    made.append(_save(fig, out / "trajectory_3d.png"))

    fig, ax = plt.subplots(); altitude = [p[2] for p in blue]; ax.plot(times, altitude, color=BLUE)
    _mark_events(ax, rows, times, altitude); ax.set(xlabel="time (s)", ylabel="y / altitude (m)", title="Blue-aircraft altitude"); ax.legend() if ax.get_legend_handles_labels()[0] else None
    made.append(_save(fig, out / "altitude.png"))

    fig, ax = plt.subplots()
    for i, (mid, track) in enumerate(missile_tracks.items()):
        distance = [math.dist(blue[j], p) if p is not None else math.nan for j, p in enumerate(track)]
        ax.plot(times, distance, color=MISSILES[i % len(MISSILES)], label=mid)
        finite = [(j, d) for j, d in enumerate(distance) if math.isfinite(d)]
        if finite:
            j, d = min(finite, key=lambda item: item[1]); ax.scatter(times[j], d, marker="X", color=MISSILES[i % len(MISSILES)], label=f"{mid} closest approach")
    ax.set(xlabel="time (s)", ylabel="missile-target distance (m)", title="Missile-target distance"); ax.legend()
    made.append(_save(fig, out / "distance.png"))

    rewards = [float(row.get("reward", 0.0)) for row in rows]
    cumulative = []; total = 0.0
    for i, row in enumerate(rows):
        total = float(row.get("cumulative_reward", total + rewards[i])); cumulative.append(total)
    fig, (a, b) = plt.subplots(2, 1, sharex=True); a.plot(times, rewards); b.plot(times, cumulative)
    a.set(ylabel="step reward", title="Reward"); b.set(xlabel="time (s)", ylabel="cumulative reward")
    made.append(_save(fig, out / "reward.png"))

    components = sorted({key for row in rows for key in (row.get("reward_components") or {})})
    fig, ax = plt.subplots()
    if components:
        for key in components:
            ax.plot(times, [float((row.get("reward_components") or {}).get(key, 0.0)) for row in rows], label=key)
        ax.legend()
    else:
        ax.text(.5, .5, "No reward components recorded", ha="center", va="center", transform=ax.transAxes)
    ax.set(xlabel="time (s)", ylabel="component reward", title="Reward components")
    made.append(_save(fig, out / "reward_components.png"))
    return made


def _decode(value: Any) -> Any:
    if not isinstance(value, str): return value
    value = value.strip()
    if not value: return value
    try: return json.loads(value)
    except json.JSONDecodeError:
        try: return ast.literal_eval(value)
        except (ValueError, SyntaxError): return value


def _csv_rows(path: Path) -> list[dict]:
    if not path.is_file(): raise PlotDataError(f"required CSV file does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as f: rows = [{k: _decode(v) for k, v in row.items()} for row in csv.DictReader(f)]
    if not rows: raise PlotDataError(f"episode table is empty: {path}")
    return rows


def plot_evaluation(evaluation_dir: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    """Plot outcome summaries and Projected PPO diagnostics when present."""
    root = Path(evaluation_dir); out = Path(output_dir) if output_dir else root / "plots"
    episodes = _csv_rows(root / "episodes.csv"); steps = _jsonl(root / "evaluation_steps.jsonl")
    plt = _plt(); made = []
    outcomes = ["hit", "crash", "success", "exhausted", "timeout"]
    counts = [sum(str(row.get("outcome")) == outcome for row in episodes) for outcome in outcomes]
    fig, ax = plt.subplots(); ax.bar(outcomes, counts, color=[OUTCOME_COLORS[o] for o in outcomes]); ax.set(ylabel="episodes", title="Outcome distribution")
    made.append(_save(fig, out / "outcome_distribution.png"))

    diagnostics = [row.get("projected_ppo") or {} for row in episodes]
    distances = [float(v) for diag in diagnostics for v in (diag.get("projection_distances") or [])]
    continuous = [v for diag in diagnostics for v in (diag.get("continuous_actions") or [])]
    continuous_commands = [v for diag in diagnostics for v in (diag.get("continuous_commands") or [])]
    projected_commands = [v for diag in diagnostics for v in (diag.get("projected_commands") or [])]
    distributions = {}
    for diag in diagnostics:
        for key, value in (diag.get("projected_action_distribution") or {}).items(): distributions[str(key)] = distributions.get(str(key), 0) + int(value)
    if not distances and not continuous and not distributions:
        return made  # Projection metrics are inapplicable to discrete algorithms.
    fig, (a, b) = plt.subplots(2, 1)
    offset = 0
    for episode_index, diag in enumerate(diagnostics):
        episode_distances = [float(value) for value in (diag.get("projection_distances") or [])]
        episode_steps = [row for row in steps if int(row.get("episode_index", 0)) == episode_index]
        episode_times = [float(row.get("time_s", i)) for i, row in enumerate(episode_steps)]
        x = episode_times[:len(episode_distances)] if len(episode_times) >= len(episode_distances) else list(range(offset, offset + len(episode_distances)))
        if episode_distances:
            a.plot(x, episode_distances, alpha=.75, label=f"episode {episode_index}")
        offset += len(episode_distances)
    a.set(ylabel="projection distance", title="Projection distance over time")
    if sum(bool(diag.get("projection_distances")) for diag in diagnostics) > 1:
        a.legend()
    b.hist(distances, bins=min(20, max(1, len(distances)))); b.set(xlabel="projection distance", ylabel="frequency")
    made.append(_save(fig, out / "projection_distance.png"))
    fig, ax = plt.subplots(); keys = sorted(distributions, key=lambda k: int(k)); ax.bar(keys, [distributions[k] for k in keys]); ax.set(xlabel="projected action_id", ylabel="count", title="Projected action histogram")
    made.append(_save(fig, out / "projected_action_histogram.png"))
    if continuous:
        values = list(zip(*continuous)); labels = ("nx", "nf", "gamma_s")
        distribution_values = list(zip(*continuous_commands)) if continuous_commands else values
        distribution_units = ("g", "g", "rad") if continuous_commands else ("normalized",) * 3
        fig, axes = plt.subplots(2, 2); axes = axes.ravel()
        for i, label in enumerate(labels):
            axes[i].hist(distribution_values[i], bins=min(20, max(1, len(distribution_values[i])))); axes[i].set(title=f"continuous {label} distribution", xlabel=distribution_units[i])
        saturation = [max(abs(float(x)) for x in row) >= .99 for row in continuous]
        axes[3].bar(["not saturated", "saturated"], [len(saturation)-sum(saturation), sum(saturation)]); axes[3].set_title("Continuous action saturation")
        made.append(_save(fig, out / "continuous_action_diagnostics.png"))
        # New artifacts contain physical command pairs; retain an explicit
        # action-id fallback for older Phase 3 outputs without inventing values.
        command_rows = min(len(continuous_commands), len(projected_commands))
        fig, axes = plt.subplots(3, 1, sharex=True)
        if command_rows:
            command_values = list(zip(*continuous_commands[:command_rows])); projected_values = list(zip(*projected_commands[:command_rows]))
            physical_labels = ("nx (g)", "nf (g)", "gamma_s (rad)")
            for i, label in enumerate(physical_labels):
                axes[i].plot(command_values[i], label="continuous command"); axes[i].plot(projected_values[i], "--", label="projected command"); axes[i].set_ylabel(label); axes[i].legend()
        else:
            action_ids = [int(row.get("action_id", 0)) for row in steps[:len(continuous)]]
            for i, label in enumerate(labels):
                axes[i].plot(values[i], label="bounded continuous output"); axes[i].scatter(range(len(action_ids)), action_ids, s=8, label="projected action_id (legacy artifact)"); axes[i].set_ylabel(label); axes[i].legend()
        axes[-1].set_xlabel("policy step"); fig.suptitle("Continuous command and projected action")
        made.append(_save(fig, out / "continuous_projected_comparison.png"))
    return made


def _condition_signature(manifest: dict) -> dict:
    return {key: manifest.get(key) for key in ("scenarios", "platform", "episodes", "seeds", "deterministic", "actions", "action_catalog_version")}


def compare_evaluations(evaluations: Iterable[str | Path], output_dir: str | Path) -> list[Path]:
    """Validate fair conditions and create overall/scenario algorithm charts."""
    roots = [Path(path) for path in evaluations]
    if len(roots) != 3:
        raise PlotDataError("exactly three evaluation directories are required")
    manifests = [json.loads((root / "manifest.json").read_text(encoding="utf-8")) for root in roots]
    expected = _condition_signature(manifests[0])
    for root, manifest in zip(roots[1:], manifests[1:]):
        if _condition_signature(manifest) != expected:
            raise PlotDataError(f"evaluation conditions differ for {root}; scenarios, platform, episodes, seeds, deterministic mode, and actions must match")
    metrics = [json.loads((root / "metrics.json").read_text(encoding="utf-8")) for root in roots]
    algorithms = [str(m.get("algorithm", root.name)) for m, root in zip(manifests, roots)]
    if set(algorithms) != COMPARISON_ALGORITHMS or len(set(algorithms)) != len(algorithms):
        raise PlotDataError("evaluations must contain ppo_projected, ppo_discrete, and rainbow_dqn exactly once")
    names = ["survival_rate", "escape_completion_rate", "hit_rate", "crash_rate", "timeout_rate", "exhausted_rate"]
    numeric = [("mean reward", "reward"), ("episode duration (s)", "duration_s"), ("min sampled distance (m)", "min_sampled_distance_m"), ("lowest altitude y (m)", "lowest_altitude_y_m")]
    plt = _plt(); fig, axes = plt.subplots(2, 3, figsize=(15, 8)); axes = axes.ravel()
    width = .8 / len(algorithms); x = list(range(len(names)))
    for i, (algorithm, metric) in enumerate(zip(algorithms, metrics)):
        overall = metric["overall"]; axes[0].bar([v + i*width for v in x], [float(overall.get(k, 0)) for k in names], width, label=algorithm)
    axes[0].set_xticks([v + width*(len(algorithms)-1)/2 for v in x], names, rotation=35, ha="right"); axes[0].set_ylabel("rate"); axes[0].legend()
    for ax, (title, key) in zip(axes[1:], numeric):
        vals=[]
        for metric in metrics:
            value=metric["overall"].get(key); vals.append(value.get("mean") if isinstance(value,dict) else value)
        ax.bar(algorithms, [math.nan if v is None else float(v) for v in vals]); ax.set_title(title); ax.tick_params(axis="x", rotation=20)
    axes[-1].axis("off")
    made=[_save(fig, Path(output_dir) / "algorithm_comparison.png")]
    scenarios = list(expected.get("scenarios") or [])
    if scenarios:
        scenario_metrics = (
            ("survival rate", "survival_rate"), ("escape completion rate", "escape_completion_rate"),
            ("mean reward", "reward"), ("duration (s)", "duration_s"),
            ("min sampled distance (m)", "min_sampled_distance_m"), ("lowest altitude y (m)", "lowest_altitude_y_m"),
        )
        fig, axes = plt.subplots(len(scenarios) * 2, 3, figsize=(15, max(7, 6*len(scenarios))), squeeze=False)
        for scenario_index, scenario in enumerate(scenarios):
            for metric_index, (title, key) in enumerate(scenario_metrics):
                ax = axes[scenario_index * 2 + metric_index // 3, metric_index % 3]
                values = []
                for metric in metrics:
                    groups = [g["metrics"] for g in metric.get("groups", []) if g.get("keys", {}).get("scenario") == scenario]
                    group_values = []
                    for group in groups:
                        value = group.get(key)
                        value = value.get("mean") if isinstance(value, dict) else value
                        if value is not None:
                            group_values.append(float(value))
                    values.append(sum(group_values) / len(group_values) if group_values else math.nan)
                ax.bar(algorithms, values)
                ax.set_title(f"{scenario}\n{title}")
                ax.tick_params(axis="x", rotation=20)
                if key.endswith("_rate"):
                    ax.set_ylim(0, 1)
        made.append(_save(fig, Path(output_dir) / "scenario_comparison.png"))
    return made
