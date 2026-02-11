
"""Training script for the blue escape agent."""

from __future__ import annotations

import os
import time
import json
from dataclasses import asdict
from typing import Tuple, Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import EnvConfig, TrainConfig
from env.escape_env import EscapeEnv
from env.acmi_io import write_acmi
from agent.dqn_agent import DQNAgent, DQNConfig


def rolling_mean(values: List[float], window: int) -> float:
    if not values:
        return 0.0
    w = min(window, len(values))
    return float(np.mean(values[-w:]))


def rolling_std(values: List[float], window: int) -> float:
    if not values:
        return 0.0
    w = min(window, len(values))
    return float(np.std(values[-w:]))

def make_env_and_agent(
    env_cfg: EnvConfig,
    train_cfg: TrainConfig,
    seed: int = 0,
) -> Tuple[EscapeEnv, DQNAgent]:
    env = EscapeEnv(env_cfg, seed=seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dqn_cfg = DQNConfig(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        lr=train_cfg.lr,
        gamma=train_cfg.gamma,
        batch_size=train_cfg.batch_size,
        replay_size=train_cfg.replay_size,
        start_learning=train_cfg.start_learning,
        epsilon_start=train_cfg.epsilon_start,
        epsilon_end=train_cfg.epsilon_end,
        epsilon_decay=train_cfg.epsilon_decay,
        target_update_interval=train_cfg.target_update_interval,
        device=device,
    )
    agent = DQNAgent(dqn_cfg)
    return env, agent


def save_checkpoint(
    path: str,
    episode: int,
    agent: DQNAgent,
    env: EscapeEnv,
) -> None:
    payload: Dict[str, Any] = {
        "episode": episode,
        "blue": agent.get_state(),
        "red": env.get_red_params(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    agent: DQNAgent,
    env: EscapeEnv,
    load_blue: bool = True,
    load_red: bool = True,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=agent.device)
    if load_blue and "blue" in payload:
        agent.load_state(payload["blue"])
    if load_red and "red" in payload:
        env.set_red_params(payload["red"])
    return payload


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
    train_cfg.checkpoint_dir = os.path.join(run_dir, "checkpoints")
    train_cfg.results_dir = os.path.join(run_dir, "results")

    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "round_name": round_name,
                "distance_km": distance_km,
                "env": asdict(env_cfg),
                "train": asdict(train_cfg),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if env_cfg.log_trajectories:
        os.makedirs(env_cfg.save_dir, exist_ok=True)

    env, agent = make_env_and_agent(env_cfg, train_cfg, seed=0)

    if train_cfg.load_checkpoint_path:
        load_checkpoint(
            train_cfg.load_checkpoint_path,
            agent,
            env,
            load_blue=train_cfg.load_blue,
            load_red=train_cfg.load_red,
        )

    if train_cfg.checkpoint_interval > 0:
        os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    episode_rewards = []
    global_step = 0
    success_count = 0
    episode_losses: List[float] = []
    win_flags: List[int] = []
    per_episode_rows: List[Dict[str, Any]] = []
    convergence_points: List[Dict[str, float]] = []

    os.makedirs(train_cfg.results_dir, exist_ok=True)

    for ep in range(1, train_cfg.episodes + 1):
        start_time = time.time()
        step = 0
        env.reset()
        obs = apply_fixed_scenario(env, float(distance_km))
        done = False
        ep_reward = 0.0
        episode_info = None
        loss_values = []

        while not done:
            action = agent.select_action(obs, eval_mode=False)
            next_obs, reward, done, info = env.step(action)
            episode_info = info

            agent.store_transition(obs, action, reward, next_obs, done)
            loss = agent.update()
            if loss is not None:
                loss_values.append(loss)

            obs = next_obs
            ep_reward += reward
            global_step += 1
            step += 1

        episode_rewards.append(ep_reward)
        # Determine whether this episode is a successful escape (blue survives until timeout).
        episode_steps = env.step_count
        episode_success = False
        if episode_info is not None:
            is_timeout = bool(episode_info.get("timeout", False))
            is_hit = bool(episode_info.get("hit", False))
            crashed = bool(episode_info.get("crashed", False))
            missiles_exhausted = bool(episode_info.get("missiles_exhausted", False))
            if is_timeout or missiles_exhausted and (not is_hit) and (not crashed):
                success_count += 1
                episode_success = True

        cumulative_success_rate = success_count / ep if ep > 0 else 0.0
        mean_loss = float(np.mean(loss_values)) if loss_values else np.nan
        episode_losses.append(mean_loss)
        win_flags.append(int(episode_success))

        reward_ma100 = rolling_mean(episode_rewards, window=100)
        reward_std100 = rolling_std(episode_rewards, window=100)
        reward_cv100 = reward_std100 / (abs(reward_ma100) + 1e-6)
        loss_ma100 = rolling_mean(
            [x for x in episode_losses if np.isfinite(x)],
            window=100,
        )
        win_rate100 = rolling_mean(win_flags, window=100)
        per_episode_rows.append(
            {
                "episode": ep,
                "steps": episode_steps,
                "win": int(episode_success),
                "episode_reward": ep_reward,
                "episode_mean_loss": mean_loss,
                "cumulative_success_rate": cumulative_success_rate,
                "reward_ma100": reward_ma100,
                "reward_std100": reward_std100,
                "reward_cv100": reward_cv100,
                "loss_ma100": loss_ma100,
                "win_rate100": win_rate100,
            }
        )

        if ep % train_cfg.print_interval == 0:
            avg_reward = sum(episode_rewards[-train_cfg.print_interval :]) / train_cfg.print_interval
            elapsed = time.time() - start_time
            success_rate = cumulative_success_rate
            print(
                f"[distance={distance_km:02d}km] "
                f"Episode {ep:4d} | avg_reward(last {train_cfg.print_interval}) = {avg_reward:6.3f} | "
                f"success = {success_count}/{ep} ({success_rate * 100:5.1f}%) | "
                f"R_ma100 = {reward_ma100:7.3f} | R_cv100 = {reward_cv100:6.3f} | "
                f"L_ma100 = {loss_ma100:8.5f} | WinRate100 = {win_rate100 * 100:5.1f}% | "
                f"steps = {step:6d} | elapsed = {elapsed:6.1f}s"
            )

        if ep % 10 == 0:
            convergence_points.append(
                {
                    "episode": ep,
                    "cumulative_success_rate": cumulative_success_rate,
                    "reward_ma100": reward_ma100,
                    "loss_ma100": loss_ma100,
                    "win_rate100": win_rate100,
                }
            )

        if train_cfg.checkpoint_interval > 0 and ep % train_cfg.checkpoint_interval == 0:
            ckpt_name = f"checkpoint_ep{ep:04d}.pt"
            ckpt_path = os.path.join(train_cfg.checkpoint_dir, ckpt_name)
            save_checkpoint(ckpt_path, ep, agent, env)

        # Every 10 episodes, convert this episode to a Tacview ACMI
        if env_cfg.log_trajectories and ep % 10 == 0:
            csv_dir = os.path.join(env_cfg.save_dir, "csv", str(ep))
            if os.path.isdir(csv_dir):
                add_plane_explosion = True
                if episode_info is not None:
                    is_timeout = bool(episode_info.get("timeout", False))
                    is_hit = bool(episode_info.get("hit", False))
                    crashed = bool(episode_info.get("crashed", False))
                    missiles_exhausted = bool(episode_info.get("missiles_exhausted", False))
                    if (is_timeout or missiles_exhausted) and (not is_hit) and (not crashed):
                        add_plane_explosion = False
                target_name = f"session_ep{ep:04d}"
                write_acmi(
                    target_name=target_name,
                    source_dir=csv_dir,
                    time_unit=env_cfg.dt,
                    explode_time=10,
                    add_plane_explosion=add_plane_explosion,
                )
                print(f"[ACMI][distance={distance_km:02d}km] Episode {ep}: wrote {target_name}.acmi from {csv_dir}")
            else:
                print(f"[ACMI][distance={distance_km:02d}km] Episode {ep}: csv dir {csv_dir} not found, skip.")

    print(f"Training finished for distance={distance_km:02d}km.")

    if convergence_points:
        curve_df = pd.DataFrame(convergence_points)
        x_vals = curve_df["episode"]

        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

        axes[0].plot(x_vals, curve_df["cumulative_success_rate"], marker="o", label="Cumulative Success")
        axes[0].plot(x_vals, curve_df["win_rate100"], marker="s", label="WinRate100")
        axes[0].set_ylabel("Success Rate")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend(loc="lower right")

        axes[1].plot(x_vals, curve_df["reward_ma100"], color="tab:green", label="Reward MA100")
        axes[1].set_ylabel("Reward")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend(loc="best")

        axes[2].plot(x_vals, curve_df["loss_ma100"], color="tab:red", label="Loss MA100")
        axes[2].set_ylabel("Loss")
        axes[2].set_xlabel("Episode")
        axes[2].grid(True, linestyle="--", alpha=0.5)
        axes[2].legend(loc="best")

        fig.suptitle("Blue Agent Convergence Indicators")
        plot_path = os.path.join(train_cfg.results_dir, "convergence_indicators.png")
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        indicator_csv_path = os.path.join(train_cfg.results_dir, "convergence_indicators.csv")
        curve_df.to_csv(indicator_csv_path, index=False)

    if per_episode_rows:
        df = pd.DataFrame(per_episode_rows)
        excel_path = os.path.join(train_cfg.results_dir, "episode_summary.csv")
        df.to_csv(excel_path, index=False)

def train() -> None:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    root_run_dir = os.path.join(EnvConfig().save_dir, run_id)
    os.makedirs(root_run_dir, exist_ok=True)

    for distance_km in range(5, 51):
        train_for_distance(distance_km=distance_km, root_run_dir=root_run_dir, run_id=run_id)


if __name__ == "__main__":
    train()
