"""Run one BlueEscapeEnv scenario and write a JSONL trajectory plus summary."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess
import sys

from air_combat_rl.io.trajectory_writer import SCHEMA_VERSION, normalize_json, TrajectoryWriter
from air_combat_rl.runtime import build_blue_escape_env


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _snapshot(world):
    snap = world.snapshot()
    def kin(k):
        return {"position_xzy_m": [k.position.x, k.position.z, k.position.y], "speed_mps": k.speed, "gamma_rad": k.angles.gamma, "psi_rad": k.angles.psi}
    return {
        "time_s": snap.time_s,
        "blue": {**kin(snap.blue.kinematics), "alive": snap.blue.alive, "platform": snap.blue.platform},
        "missiles": [
            {"id": f"missile_{i}", **kin(m.kinematics), "alive": m.alive, "locked": m.locked, "powered": m.powered, "age_s": m.age_s}
            for i, m in enumerate(snap.missiles)
        ],
    }


def _distance(blue, missile):
    bp = blue.kinematics.position; mp = missile.kinematics.position
    return math.sqrt((bp.x-mp.x)**2 + (bp.z-mp.z)**2 + (bp.y-mp.y)**2)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(normalize_json(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run one blue-escape scenario and write JSON artifacts.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--actions", required=True)
    parser.add_argument("--platform", required=True, choices=["zdj", "yjj"])
    parser.add_argument("--policy", required=True, choices=["constant", "random_valid"])
    parser.add_argument("--action-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-policy-steps", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty: {out}; pass --overwrite to replace run artifacts")
    out.mkdir(parents=True, exist_ok=True)

    env, cfg = build_blue_escape_env(args.scenario, args.actions, args.platform, args.seed, args.max_policy_steps)
    obs, reset_info = env.reset(seed=args.seed)
    rng = random.Random(args.seed)
    cumulative = 0.0
    event_counts = Counter()
    min_alt = max_alt = env.world.blue.kinematics.position.y
    min_sampled_distance = math.inf
    total_substeps = 0
    final_outcome = "running"; terminated = truncated = False

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scenario_path": cfg.scenario_path,
        "actions_path": cfg.actions_path,
        "platform": args.platform,
        "policy": args.policy,
        "action_id": args.action_id,
        "seed": args.seed,
        "physics_dt": cfg.scenario.physics_dt,
        "policy_dt": cfg.scenario.policy_dt,
        "max_episode_time_s": cfg.scenario.max_episode_time_s,
        "max_policy_steps": args.max_policy_steps,
        "scenario": cfg.scenario,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    _write_json(out / "manifest.json", manifest)

    with TrajectoryWriter(out / "steps.jsonl") as writer:
        step = 0
        while True:
            mask = list(bool(x) for x in reset_info["action_mask"])
            if args.policy == "constant":
                if args.action_id is None:
                    raise SystemExit("--action-id is required for --policy constant")
                action_id = args.action_id
                if action_id < 0 or action_id >= len(mask) or not mask[action_id]:
                    raise SystemExit(f"action_id {action_id} is illegal for platform {args.platform}")
            else:
                legal = [i for i, ok in enumerate(mask) if ok]
                action_id = rng.choice(legal)
            action = env.actions.action(action_id)
            result = env.step(action_id)
            cumulative += result.reward
            final_outcome = str(result.info["outcome"]); terminated = result.terminated; truncated = result.truncated
            total_substeps += int(result.info["substeps"])
            for event in result.info["events"]:
                event_counts[event.kind] += 1
            y = env.world.blue.kinematics.position.y
            min_alt = min(min_alt, y); max_alt = max(max_alt, y)
            for missile in env.world.missiles:
                min_sampled_distance = min(min_sampled_distance, _distance(env.world.blue, missile))
            writer.write_step({
                "episode": 0, "step": step, "time_s": result.info["time_s"], "action_id": action_id, "action_name": action.name,
                "reward": result.reward, "cumulative_reward": cumulative, "terminated": result.terminated, "truncated": result.truncated,
                "outcome": final_outcome, "substeps": result.info["substeps"], "events": result.info["events"],
                "reward_components": result.info["reward_components"], "missile_mask": result.info["missile_mask"], "action_mask": result.info["action_mask"],
                "world_snapshot": _snapshot(env.world),
            })
            reset_info = result.info
            step += 1
            if terminated or truncated:
                break
    _write_json(out / "episode_summary.json", {
        "schema_version": SCHEMA_VERSION, "final_outcome": final_outcome, "terminated": terminated, "truncated": truncated,
        "policy_steps": step, "physics_substeps": total_substeps, "duration_s": env.world.time_s, "total_reward": cumulative,
        "min_altitude_y_m": min_alt, "max_altitude_y_m": max_alt,
        "min_sampled_distance_m": None if min_sampled_distance == math.inf else min_sampled_distance,
        "event_counts": dict(event_counts),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
