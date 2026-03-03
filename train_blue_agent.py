"""Simulation script using behavior-tree policy for blue aircraft."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from agent.blue_bt_agent import BlueBTAgent, MissileSnapshot, PlaneSnapshot
from config import EnvConfig, TrainConfig
from env.acmi_io import write_acmi
from env.escape_env import EscapeEnv


def rolling_mean(values: List[float], window: int) -> float:
    if not values:
        return 0.0
    w = min(window, len(values))
    return float(np.mean(values[-w:]))


def apply_fixed_scenario(env: EscapeEnv, distance_km: float) -> np.ndarray:
    """Apply fixed blue/missile initialization for the current episode and return updated observation."""
    env.blue_pos = np.array([distance_km, 0.0, 10.0], dtype=float)
    env.blue_vel = np.array([-env.cfg.blue_max_speed, 0.0, 0.0], dtype=float)

    env.missile_pos[0] = np.array([0.0, 0.0, 10.0], dtype=float)
    env.missile_launch_times[0] = 0.0

    env.initial_missile_distances = np.linalg.norm(
        env.missile_pos - env.blue_pos[None, :],
        axis=1,
    )
    env.prev_min_dist = float(np.min(env.initial_missile_distances))
    env.prev_blue_vel = env.blue_vel.copy()

    if env.log_enabled:
        env._init_logging()
        env._log_current_state()

    return env._get_obs()


def _build_plane_snapshot(env: EscapeEnv) -> PlaneSnapshot:
    roll = env.blue_model.roll_rad
    return PlaneSnapshot(
        pos=env.blue_pos.copy(),
        vel=env.blue_vel.copy(),
        roll_rad=0.0 if roll is None else float(roll),
    )


def _build_missile_snapshots(env: EscapeEnv) -> List[MissileSnapshot]:
    snapshots: List[MissileSnapshot] = []
    for i in range(env.cfg.num_missiles):
        snapshots.append(
            MissileSnapshot(
                pos=env.missile_pos[i].copy(),
                team=1,
                is_active=bool(env.missile_alive[i] and env.missile_launched[i]),
            )
        )
    return snapshots


def train_for_distance(distance_km: int, root_run_dir: str, run_id: str) -> None:
    env_cfg = EnvConfig()
    train_cfg = TrainConfig()
    env_cfg.reward_mode = train_cfg.reward_mode

    env_cfg.num_missiles = 1
    env_cfg.blue_x_min = float(distance_km)
    env_cfg.blue_x_max = float(distance_km)
    env_cfg.blue_y_min = 0.0
    env_cfg.blue_y_max = 0.0
    env_cfg.blue_z_min = 10.0
    env_cfg.blue_z_max = 10.0
    env_cfg.blue_heading_min = 180.0
    env_cfg.blue_heading_max = 180.0

    round_name = f"distance_{distance_km:02d}km"
    run_dir = os.path.join(root_run_dir, round_name)
    env_cfg.save_dir = run_dir
    train_cfg.results_dir = os.path.join(run_dir, "results")

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "round_name": round_name,
                "distance_km": distance_km,
                "env": asdict(env_cfg),
                "train": asdict(train_cfg),
                "policy": "BlueBTAgent",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    os.makedirs(train_cfg.results_dir, exist_ok=True)
    env = EscapeEnv(env_cfg, seed=0)
    bt_agent = BlueBTAgent(uid="blue_bt", team=0)

    success_count = 0
    rewards: List[float] = []
    win_flags: List[int] = []
    rows: List[Dict[str, Any]] = []

    for ep in range(1, train_cfg.episodes + 1):
        start = time.time()
        env.reset()
        obs = apply_fixed_scenario(env, float(distance_km))
        bt_agent.reset(obs)
        done = False
        ep_reward = 0.0
        episode_info: Dict[str, Any] | None = None

        while not done:
            plane = _build_plane_snapshot(env)
            missiles = _build_missile_snapshots(env)
            action = bt_agent.get_action(plane, missiles, enemies=[])
            _, reward, done, info = env.step(action)
            episode_info = info
            ep_reward += reward

        rewards.append(ep_reward)

        is_timeout = bool((episode_info or {}).get("timeout", False))
        is_hit = bool((episode_info or {}).get("hit", False))
        crashed = bool((episode_info or {}).get("crashed", False))
        missiles_exhausted = bool((episode_info or {}).get("missiles_exhausted", False))
        episode_success = (is_timeout or missiles_exhausted) and (not is_hit) and (not crashed)
        if episode_success:
            success_count += 1

        win_flags.append(int(episode_success))
        cumulative_success_rate = success_count / ep
        reward_ma100 = rolling_mean(rewards, window=100)
        win_rate100 = rolling_mean(win_flags, window=100)

        rows.append(
            {
                "episode": ep,
                "steps": env.step_count,
                "win": int(episode_success),
                "episode_reward": ep_reward,
                "cumulative_success_rate": cumulative_success_rate,
                "reward_ma100": reward_ma100,
                "win_rate100": win_rate100,
            }
        )

        if ep % train_cfg.print_interval == 0:
            avg_reward = float(np.mean(rewards[-train_cfg.print_interval :]))
            print(
                f"[distance={distance_km:02d}km] Episode {ep:4d} | "
                f"avg_reward(last {train_cfg.print_interval}) = {avg_reward:6.3f} | "
                f"success = {success_count}/{ep} ({cumulative_success_rate * 100:5.1f}%) | "
                f"R_ma100 = {reward_ma100:7.3f} | WinRate100 = {win_rate100 * 100:5.1f}% | "
                f"elapsed = {time.time() - start:6.1f}s"
            )

        if env_cfg.log_trajectories:
            csv_dir = os.path.join(env_cfg.save_dir, "csv", str(ep))
            if os.path.isdir(csv_dir):
                add_plane_explosion = not episode_success
                target_name = f"session_ep{ep:04d}"
                write_acmi(
                    target_name=target_name,
                    source_dir=csv_dir,
                    time_unit=env_cfg.dt,
                    explode_time=10,
                    add_plane_explosion=add_plane_explosion,
                )

    print(f"Simulation finished for distance={distance_km:02d}km.")

    if rows:
        result_df = pd.DataFrame(rows)
        result_df.to_csv(os.path.join(train_cfg.results_dir, "episode_summary.csv"), index=False)


def train() -> None:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    root_run_dir = os.path.join(EnvConfig().save_dir, run_id)
    os.makedirs(root_run_dir, exist_ok=True)

    for distance_km in range(5, 51):
        train_for_distance(distance_km=distance_km, root_run_dir=root_run_dir, run_id=run_id)


if __name__ == "__main__":
    train()
