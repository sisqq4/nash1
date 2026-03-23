"""Shared utilities for 1v1 evaluation scenario sweeps."""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import sys

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import EnvConfig, TrainConfig
from env.acmi_io import write_acmi
from train_blue_agent import load_checkpoint, make_env_and_agent, save_checkpoint


def is_success(info: Optional[Dict[str, Any]]) -> bool:
    if not info:
        return False
    is_hit = bool(info.get("hit", False))
    is_crashed = bool(info.get("crashed", False))
    is_timeout = bool(info.get("timeout", False))
    missiles_exhausted = bool(info.get("missiles_exhausted", False))
    return (is_timeout or missiles_exhausted) and (not is_hit) and (not is_crashed)


def apply_config(obj: Any, cfg: Dict[str, Any]) -> None:
    for key, value in cfg.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def load_run_config(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    run_dir = Path(checkpoint_path).resolve().parent.parent
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_checkpoint_path(
    checkpoint: Optional[str],
    run_id: Optional[str],
    episode: Optional[int],
    checkpoint_name: Optional[str],
) -> str:
    outputs_dir = Path(EnvConfig().save_dir)
    if checkpoint:
        checkpoint_path = Path(checkpoint)
    elif checkpoint_name:
        checkpoint_path = outputs_dir / run_id / "checkpoints" / checkpoint_name if run_id else outputs_dir / "checkpoints" / checkpoint_name
    else:
        if not run_id or episode is None:
            raise ValueError("需要 --checkpoint，或同时提供 --run-id 与 --episode。")
        checkpoint_path = outputs_dir / run_id / "checkpoints" / f"checkpoint_ep{episode:04d}.pt"

    checkpoint_path = checkpoint_path.resolve()
    if checkpoint_path.suffix != ".pt":
        raise ValueError(f"Checkpoint 必须是 .pt 文件: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")
    return str(checkpoint_path)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_scenario_sweep(
    checkpoint_path: str,
    output_root: str,
    scenarios: Iterable[Dict[str, Any]],
    episodes_per_scenario: int,
    seed: int,
    checkpoint_interval: int = 10,
    report_interval: int = 10,
    reward_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    env_cfg = EnvConfig()
    train_cfg = TrainConfig()

    loaded_cfg = load_run_config(checkpoint_path)
    if loaded_cfg:
        apply_config(env_cfg, loaded_cfg.get("env", {}))
        apply_config(train_cfg, loaded_cfg.get("train", {}))

    if reward_mode is not None:
        env_cfg.reward_mode = reward_mode
        train_cfg.reward_mode = reward_mode

    env_cfg.log_trajectories = True
    os.makedirs(output_root, exist_ok=True)

    with open(os.path.join(output_root, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"env": asdict(env_cfg), "train": asdict(train_cfg)}, f, ensure_ascii=False, indent=2)

    env, agent = make_env_and_agent(env_cfg, train_cfg, seed=seed)
    # NOTE:
    #   For evaluation we intentionally do NOT load red state from checkpoint.
    #   Checkpoints are saved at episode end; red nav_gains in that snapshot may be
    #   terminal values (e.g., zero after missiles expire), which would disable PN
    #   in later tests. We keep red parameters from env_cfg/config.json so red uses
    #   the same PN/drag/speed model settings as training-time configuration.
    load_checkpoint(checkpoint_path, agent, env, load_blue=True, load_red=False)
    agent.q_net.eval()

    all_rows: List[Dict[str, Any]] = []
    overall_episode = 0

    for scenario_idx, scenario in enumerate(scenarios, start=1):
        scenario_name = str(scenario["scenario_name"])
        scenario_dir = os.path.join(output_root, f"scenario_{scenario_idx:02d}_{scenario_name}")
        os.makedirs(scenario_dir, exist_ok=True)
        seed_rng = random.Random((seed + 1) * 1_000_003 + scenario_idx)

        sc_env_cfg = EnvConfig()
        apply_config(sc_env_cfg, asdict(env_cfg))
        for k, v in scenario.get("env_overrides", {}).items():
            if hasattr(sc_env_cfg, k):
                setattr(sc_env_cfg, k, v)
        sc_env_cfg.save_dir = scenario_dir
        sc_env_cfg.log_trajectories = True

        env, _ = make_env_and_agent(sc_env_cfg, train_cfg, seed=seed + scenario_idx)
        load_checkpoint(checkpoint_path, agent, env, load_blue=True, load_red=False)

        wins = 0
        for ep in range(1, episodes_per_scenario + 1):
            overall_episode += 1
            episode_seed = int(seed_rng.randint(0, 2**32 - 1))
            env.rng = np.random.default_rng(episode_seed)
            env.launcher.rng = env.rng
            obs = env.reset()
            done = False
            info: Optional[Dict[str, Any]] = None
            ep_reward = 0.0

            while not done:
                action = agent.select_action(obs, eval_mode=True)
                obs, reward, done, info = env.step(action)
                ep_reward += reward

            win = is_success(info)
            if win:
                wins += 1

            row = {
                "scenario_index": scenario_idx,
                "scenario_name": scenario_name,
                "episode": ep,
                "global_episode": overall_episode,
                "episode_seed": episode_seed,
                "reward": float(ep_reward),
                "win": int(win),
                "steps": int(info.get("step", 0)) if info else 0,
                "min_dist": float(info.get("min_dist", 0.0)) if info else 0.0,
                "timeout": int(bool(info.get("timeout", False))) if info else 0,
                "hit": int(bool(info.get("hit", False))) if info else 0,
                "crashed": int(bool(info.get("crashed", False))) if info else 0,
                "missiles_exhausted": int(bool(info.get("missiles_exhausted", False))) if info else 0,
            }
            row.update({f"param_{k}": v for k, v in scenario.items() if k != "env_overrides"})
            all_rows.append(row)

            if ep % checkpoint_interval == 0:
                ckpt_name = f"test_checkpoint_ep{ep:04d}.pt"
                ckpt_dir = os.path.join(scenario_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                save_checkpoint(
                    path=os.path.join(ckpt_dir, ckpt_name),
                    episode=ep,
                    agent=agent,
                    env=env,
                )

                csv_dir = os.path.join(scenario_dir, "csv", str(ep))
                if os.path.isdir(csv_dir):
                    write_acmi(
                        target_name=f"test_ep{ep:04d}",
                        source_dir=csv_dir,
                        time_unit=sc_env_cfg.dt,
                        explode_time=10,
                        add_plane_explosion=not win,
                    )

            if report_interval > 0 and ep % report_interval == 0:
                print(f"[{scenario_name}] Episode {ep}/{episodes_per_scenario} | win_rate={wins/ep:.3f}")

    results_dir = os.path.join(output_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    _write_csv(os.path.join(results_dir, "episode_summary.csv"), all_rows)

    grouped: Dict[tuple[int, str], Dict[str, Any]] = {}
    for row in all_rows:
        key = (int(row["scenario_index"]), str(row["scenario_name"]))
        if key not in grouped:
            grouped[key] = {
                "scenario_index": key[0],
                "scenario_name": key[1],
                "episodes": 0,
                "win_sum": 0,
                "reward_sum": 0.0,
                "steps_sum": 0,
                "min_dist_sum": 0.0,
            }
        g = grouped[key]
        g["episodes"] += 1
        g["win_sum"] += int(row["win"])
        g["reward_sum"] += float(row["reward"])
        g["steps_sum"] += int(row["steps"])
        g["min_dist_sum"] += float(row["min_dist"])

    result_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        g = grouped[key]
        episodes = max(int(g["episodes"]), 1)
        result_rows.append(
            {
                "scenario_index": g["scenario_index"],
                "scenario_name": g["scenario_name"],
                "episodes": g["episodes"],
                "win_rate": g["win_sum"] / episodes,
                "avg_reward": g["reward_sum"] / episodes,
                "avg_steps": g["steps_sum"] / episodes,
                "avg_min_dist": g["min_dist_sum"] / episodes,
            }
        )
    _write_csv(os.path.join(results_dir, "result.csv"), result_rows)

    return all_rows