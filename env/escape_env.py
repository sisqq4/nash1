
"""Escape environment: blue aircraft vs. three red missiles.

The red missiles' *launch positions* are selected by a game-theoretic style
launcher, while the blue aircraft is controlled by a reinforcement learning
agent that learns an escape policy.

This environment intentionally follows a Gym-like API but does not depend
on the gym package:

    env = EscapeEnv(env_config)
    obs = env.reset()
    obs, reward, done, info = env.step(action)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any
import numpy as np

from .game_theory_launcher import GameTheoreticLauncher, LaunchRegion
from .missile_dynamics import update_blue_state, update_missiles_towards_blue
from config import EnvConfig


class EscapeEnv:
    """3D escape environment for training the blue RL agent."""

    def __init__(self, cfg: EnvConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        region = LaunchRegion(
            region_min=cfg.region_min,
            region_max=cfg.region_max,
            num_missiles=cfg.num_missiles,
            candidate_launch_count=cfg.candidate_launch_count,
        )
        self.launcher = GameTheoreticLauncher(region, rng=self.rng)

        # Internal simulation state
        self.blue_pos = np.zeros(3, dtype=float)
        self.blue_vel = np.zeros(3, dtype=float)
        self.missile_pos = np.zeros((cfg.num_missiles, 3), dtype=float)
        self.t = 0
        self.done = False

        # Pre-compute state dimension and action space size
        # State: [blue_pos(3), blue_vel(3), missile_rel_pos(3*M)]
        self.observation_dim = 3 + 3 + 3 * cfg.num_missiles
        self.action_dim = 7  # 0: no acc, 1..6: +/-x,y,z

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        self.t = 0
        self.done = False

        # Sample blue position on a sphere of radius blue_init_radius,
        # outside the red launch cube.
        radius = self.cfg.blue_init_radius
        # Sample random direction on unit sphere
        v = self.rng.normal(size=3)
        v /= np.linalg.norm(v)
        self.blue_pos = v * radius
        self.blue_vel = np.zeros(3, dtype=float)

        # Compute red missile launch positions via game-theoretic launcher.
        launch_positions = self.launcher.compute_launch_positions(self.blue_pos)
        # Ensure correct shape (M,3)
        self.missile_pos = launch_positions.reshape(self.cfg.num_missiles, 3)

        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Advance environment by one step.

        Args:
            action: integer in [0, action_dim-1]

        Returns:
            obs: next observation
            reward: scalar reward
            done: episode termination flag
            info: extra diagnostics
        """
        if self.done:
            raise RuntimeError("Call reset() before stepping a finished episode.")

        # Clip invalid actions
        if not (0 <= int(action) < self.action_dim):
            raise ValueError(f"Invalid action {action}, expected in [0, {self.action_dim - 1}].")

        # Update blue state.
        self.blue_pos, self.blue_vel = update_blue_state(
            self.blue_pos,
            self.blue_vel,
            int(action),
            dt=self.cfg.dt,
            accel_mag=self.cfg.blue_accel,
            v_max=self.cfg.blue_max_speed,
        )

        # Update missile positions (pure pursuit).
        self.missile_pos = update_missiles_towards_blue(
            self.missile_pos, self.blue_pos, self.cfg.missile_speed, self.cfg.dt
        )

        self.t += 1

        # Compute distances and termination.
        dists = np.linalg.norm(self.missile_pos - self.blue_pos[None, :], axis=1)
        min_dist = float(np.min(dists))

        hit = min_dist <= self.cfg.hit_radius
        timeout = self.t >= self.cfg.max_steps

        reward = 0.0
        if hit:
            reward = -1.0
            self.done = True
        elif timeout:
            reward = 1.0
            self.done = True
        else:
            # Small shaping: encourage staying away from missiles.
            # Normalize by some characteristic distance.
            reward = 0.01 * (min_dist / (self.cfg.region_max - self.cfg.region_min))

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "t": self.t,
            "min_dist": min_dist,
            "hit": hit,
            "timeout": timeout,
        }
        return obs, float(reward), bool(self.done), info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Observation is blue state + relative missile positions."""
        rel_missiles = self.missile_pos - self.blue_pos[None, :]  # (M,3)
        obs = np.concatenate(
            [self.blue_pos, self.blue_vel, rel_missiles.reshape(-1)],
            axis=0,
        )
        return obs.astype(np.float32)
