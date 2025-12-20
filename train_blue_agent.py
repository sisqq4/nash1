
"""Training script for the blue escape agent.

This script wires together:
- the EscapeEnv environment (red missiles + PN guidance + Nash launcher +
  differential-game controller for nav gains + trajectory logging),
- the DQNAgent (blue reinforcement learning agent),
- and a simple training loop.

Usage (from project root):

    python train_blue_agent.py

Requirements:
    - Python 3.9+
    - numpy
    - torch
"""

from __future__ import annotations

import os
import time
from typing import Tuple

import numpy as np
import torch

from config import EnvConfig, TrainConfig
from env.escape_env import EscapeEnv
from env.acmi_io import write_acmi
from agent.dqn_agent import DQNAgent, DQNConfig


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


def train():
    env_cfg = EnvConfig()
    train_cfg = TrainConfig()

    if env_cfg.log_trajectories:
        os.makedirs(env_cfg.save_dir, exist_ok=True)

    env, agent = make_env_and_agent(env_cfg, train_cfg, seed=0)

    episode_rewards = []
    global_step = 0
    start_time = time.time()

    for ep in range(1, train_cfg.episodes + 1):
        obs = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(obs, eval_mode=False)
            next_obs, reward, done, info = env.step(action)

            agent.store_transition(obs, action, reward, next_obs, done)
            _ = agent.update()

            obs = next_obs
            ep_reward += reward
            global_step += 1

        episode_rewards.append(ep_reward)

        if ep % train_cfg.print_interval == 0:
            avg_reward = sum(episode_rewards[-train_cfg.print_interval:]) / train_cfg.print_interval
            elapsed = time.time() - start_time
            print(
                f"Episode {ep:4d} | avg_reward (last {train_cfg.print_interval}) = {avg_reward:6.3f} | "
                f"steps = {global_step:6d} | elapsed = {elapsed:6.1f}s"
            )

        # Every 10 episodes, convert that episode's CSV into a Tacview ACMI
        if env_cfg.log_trajectories and ep % 10 == 0:
            csv_dir = os.path.join(env_cfg.save_dir, "csv", str(ep))
            if os.path.isdir(csv_dir):
                target_name = f"session_ep{ep:04d}"
                write_acmi(
                    target_name=target_name,
                    source_dir=csv_dir,
                    time_unit=env_cfg.dt,
                    explode_time=10,
                )
                print(f"[ACMI] Episode {ep}: wrote {target_name}.acmi from {csv_dir}")
            else:
                print(f"[ACMI] Episode {ep}: csv dir {csv_dir} not found, skip.")

    print("Training finished.")

    # After training, convert all csv trajectories to a single ACMI file.
    # if env_cfg.log_trajectories:
    #     print("Converting CSV logs to Tacview ACMI...")
    #     write_acmi(target_name="session", save_dir=env_cfg.save_dir, time_unit=env_cfg.dt, explode_time=10)
    #     print(f"ACMI file written to {os.path.join(env_cfg.save_dir, 'acmi', 'session.acmi')}")


if __name__ == "__main__":
    train()
