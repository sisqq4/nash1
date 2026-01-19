"""Evaluate a stored blue-agent checkpoint without updating parameters."""

from __future__ import annotations

import argparse
import os
from typing import Dict, Any

from config import EnvConfig, TrainConfig
from train_blue_agent import make_env_and_agent, load_checkpoint


def _is_success(info: Dict[str, Any] | None) -> bool:
    if not info:
        return False
    is_hit = bool(info.get("hit", False))
    is_crashed = bool(info.get("crashed", False))
    is_timeout = bool(info.get("timeout", False))
    missiles_exhausted = bool(info.get("missiles_exhausted", False))
    return (is_timeout or missiles_exhausted) and (not is_hit) and (not is_crashed)


def evaluate(
    checkpoint_path: str,
    episodes: int = 100,
    seed: int = 0,
    load_blue: bool = True,
    load_red: bool = True,
    report_interval: int = 10,
) -> float:
    env_cfg = EnvConfig()
    env_cfg.log_trajectories = False
    train_cfg = TrainConfig()

    env, agent = make_env_and_agent(env_cfg, train_cfg, seed=seed)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    load_checkpoint(
        checkpoint_path,
        agent,
        env,
        load_blue=load_blue,
        load_red=load_red,
    )

    agent.q_net.eval()

    win_count = 0
    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        info = None
        step = 0

        while not done:
            action = agent.select_action(obs, eval_mode=True)
            obs, _reward, done, info = env.step(action)
            step += 1

        episode_win = _is_success(info)
        if episode_win:
            win_count += 1

        if report_interval > 0 and ep % report_interval == 0:
            win_rate = win_count / ep
            print(
                f"Episode {ep:4d} | win = {int(win_count)} | "
                f"win_rate = {win_rate * 100:5.1f}% | steps = {step:6d}"
            )

    win_rate = win_count / episodes if episodes > 0 else 0.0
    print(f"Evaluation finished. Win rate: {win_rate * 100:.2f}% ({win_count}/{episodes})")
    return win_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained blue agent.")
    parser.add_argument("--checkpoint", type=str, default="outputs/20260119_112414/checkpoints/checkpoint_ep2000.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--no-load-blue", action="store_true", help="Skip loading blue agent params")
    parser.add_argument("--load-red", action="store_true", help="Load red launcher params")
    parser.add_argument("--report-interval", type=int, default=10, help="Episodes between progress logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint or TrainConfig().load_checkpoint_path
    if not checkpoint_path:
        raise ValueError("Checkpoint path required (use --checkpoint or set TrainConfig.load_checkpoint_path).")
    evaluate(
        checkpoint_path=checkpoint_path,
        episodes=args.episodes,
        seed=args.seed,
        load_blue=not args.no_load_blue,
        load_red=args.load_red,
        report_interval=args.report_interval,
    )


if __name__ == "__main__":
    main()