
"""Escape environment: blue aircraft vs. red missiles using PN guidance,
a differential-game controller for navigation gains, and trajectory logging.

场景设置（近似）：
- 三枚雷达弹起始坐标均在：
    x ∈ [0, 20] km, y ∈ [-10, 10] km, z ∈ [-10, 10] km；
- 蓝方飞机起始坐标约为 [70, 0, 0] km，初始速度方向为任意，
  速度模长约为 2000 km/h（≈0.556 km/s）；
- 导弹最大速度约 4900 km/h（≈1.361 km/s），使用 PN 导引追踪蓝机；
- 蓝机最大过载约 9g（通过内置加速度上限近似实现）；
- 每枚导弹的 PN 导引增益由微分博弈控制器联合、实时调节。
"""

from __future__ import annotations

from typing import Dict, Tuple, Any, List
import os
import math

import numpy as np

from .game_theory_launcher import GameTheoreticLauncher, LaunchRegion
from .missile_dynamics import update_blue_state, update_missiles_pn
from .diff_game_controller import DifferentialGameController
from .acmi_io import write_csv
from config import EnvConfig


class EscapeEnv:
    """3D escape environment for training the blue RL agent."""

    def __init__(self, cfg: EnvConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        region = LaunchRegion(
            num_missiles=cfg.num_missiles,
            candidate_launch_count=cfg.candidate_launch_count,
            num_blue_strategies=cfg.num_blue_strategies,
            fictitious_iters=cfg.fictitious_iters,
            blue_escape_distance=cfg.blue_escape_distance,
        )
        self.launcher = GameTheoreticLauncher(region, rng=self.rng)
        self.diff_ctrl = DifferentialGameController(cfg) if cfg.use_diff_game else None

        # Internal simulation state
        self.blue_pos = np.zeros(3, dtype=float)
        self.blue_vel = np.zeros(3, dtype=float)
        self.missile_pos = np.zeros((cfg.num_missiles, 3), dtype=float)
        self.missile_vel = np.zeros((cfg.num_missiles, 3), dtype=float)
        # Navigation gains for each missile (jointly updated by diff-game controller).
        self.nav_gains = np.full(cfg.num_missiles, cfg.nav_gain, dtype=float)

        self.t = 0
        self.done = False

        # Pre-compute state dimension and action space size
        # State: [blue_pos(3), blue_vel(3), missile_rel_pos(3*M)]
        self.observation_dim = 3 + 3 + 3 * cfg.num_missiles
        self.action_dim = 7  # 0: no acc, 1..6: +/-x,y,z

        # Trajectory logging (for Tacview export)
        self.log_enabled = bool(cfg.log_trajectories)
        self.save_dir = cfg.save_dir
        self.episode_index = 0
        self.plane_global_id = 0
        self.missile_global_id = 0

        # Will be initialized in reset()
        self._plane_track: List[List[float]] | None = None
        self._plane_name: str | None = None
        self._missile_tracks: List[List[List[float]]] | None = None
        self._missile_names: List[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        self.t = 0
        self.done = False

        # --- Blue initial state: fixed position [70, 0, 0] km,
        #     random velocity direction with fixed speed norm. ---
        self.blue_pos = np.array(
            [self.cfg.blue_init_x, self.cfg.blue_init_y, self.cfg.blue_init_z],
            dtype=float,
        )
        # Random unit direction
        v_dir = self.rng.normal(size=3)
        v_norm = np.linalg.norm(v_dir)
        if v_norm < 1e-6:
            v_dir = np.array([1.0, 0.0, 0.0])
            v_norm = 1.0
        v_dir /= v_norm
        self.blue_vel = v_dir * self.cfg.blue_max_speed

        # --- Red missiles: Nash game to select launch positions inside
        #     x:[0,20], y:[-10,10], z:[-10,10] km. ---
        launch_positions = self.launcher.compute_launch_positions(self.blue_pos)
        self.missile_pos = launch_positions.reshape(self.cfg.num_missiles, 3)

        # Initialize missile velocities to point towards the blue aircraft.
        self.missile_vel = np.zeros_like(self.missile_pos)
        for i in range(self.cfg.num_missiles):
            direction = self.blue_pos - self.missile_pos[i]
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                direction = np.array([1.0, 0.0, 0.0])
                norm = 1.0
            direction /= norm
            self.missile_vel[i] = direction * self.cfg.missile_speed

        # Reset nav gains to base value.
        self.nav_gains = np.full(self.cfg.num_missiles, self.cfg.nav_gain, dtype=float)

        # Initialize logging for this episode.
        if self.log_enabled:
            self._init_logging()

            # Log initial state at t=0
            self._log_current_state()

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

        # 1) Update blue state based on RL action.
        self.blue_pos, self.blue_vel = update_blue_state(
            self.blue_pos,
            self.blue_vel,
            int(action),
            dt=self.cfg.dt,
            accel_mag=self.cfg.blue_accel,
            v_max=self.cfg.blue_max_speed,
        )

        # 2) Jointly update nav_gains via differential-game controller (if enabled).
        if self.diff_ctrl is not None:
            self.nav_gains = self.diff_ctrl.update_nav_gains(
                self.blue_pos,
                self.blue_vel,
                self.missile_pos,
                self.missile_vel,
                self.nav_gains,
                self.cfg.dt,
            )

        # 3) Update missile positions and velocities using PN-like guidance
        #    with (possibly) missile-dependent navigation gains.
        self.missile_pos, self.missile_vel = update_missiles_pn(
            self.missile_pos,
            self.missile_vel,
            self.blue_pos,
            self.blue_vel,
            self.cfg.missile_speed,
            self.cfg.dt,
            self.nav_gains,
        )

        self.t += 1

        # Log state after this step
        if self.log_enabled:
            self._log_current_state()

        # 4) Compute distances and termination.
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
            # Normalize by characteristic distance (region_span).
            reward = 0.01 * (min_dist / self.cfg.region_span)

        # If episode just finished, flush logs to CSV.
        if self.done and self.log_enabled:
            self._flush_logs_to_csv()

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "t": self.t,
            "min_dist": min_dist,
            "hit": hit,
            "timeout": timeout,
            "nav_gains": self.nav_gains.copy(),
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

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _init_logging(self) -> None:
        """Prepare trajectory containers and assign object IDs for this episode."""
        self.episode_index += 1

        # Plane: single blue aircraft
        self.plane_global_id += 1
        plane_id = self.plane_global_id
        self._plane_name = f"plane_blue.{plane_id}.0"  # start_time=0

        # Missiles: three red missiles
        self._missile_tracks = []
        self._missile_names = []
        for _ in range(self.cfg.num_missiles):
            self.missile_global_id += 1
            mid = self.missile_global_id
            name = f"missile_red.{mid}.0"  # start_time=0
            self._missile_names.append(name)
            self._missile_tracks.append([])

        self._plane_track = []

    def _compute_orientation(self, vel: np.ndarray) -> Tuple[float, float, float]:
        """Compute (roll, pitch, yaw) in degrees from velocity vector.

        This is a simple kinematic approximation:
            yaw   = atan2(v_y, v_x)
            pitch = atan2(v_z, sqrt(v_x^2 + v_y^2))
            roll  = 0 (not modeled)
        """
        vx, vy, vz = vel
        # avoid division by zero
        horiz = math.sqrt(vx * vx + vy * vy)
        yaw = math.degrees(math.atan2(vy, vx))
        pitch = math.degrees(math.atan2(vz, horiz))
        roll = 0.0
        return roll, pitch, yaw

    def _log_current_state(self) -> None:
        if self._plane_track is None or self._missile_tracks is None:
            return

        # Plane
        roll, pitch, yaw = self._compute_orientation(self.blue_vel)
        self._plane_track.append([
            float(self.blue_pos[0]),
            float(self.blue_pos[1]),
            float(self.blue_pos[2]),
            roll,
            pitch,
            yaw,
        ])

        # Missiles
        for i in range(self.cfg.num_missiles):
            m_pos = self.missile_pos[i]
            m_vel = self.missile_vel[i]
            roll_m, pitch_m, yaw_m = self._compute_orientation(m_vel)
            self._missile_tracks[i].append([
                float(m_pos[0]),
                float(m_pos[1]),
                float(m_pos[2]),
                roll_m,
                pitch_m,
                yaw_m,
            ])

    def _flush_logs_to_csv(self) -> None:
        if self._plane_track is None or self._missile_tracks is None:
            return
        if self._plane_name is None or self._missile_names is None:
            return

        # 这里的 episode_index 从 1 开始，每次 reset() 自增
        ep_idx = self.episode_index

        # Write plane track
        if self._plane_track:
            write_csv(self.cfg.save_dir, self._plane_name, self._plane_track, episode_index=ep_idx,)

        # Write missiles
        for track, name in zip(self._missile_tracks, self._missile_names):
            if track:
                write_csv(self.cfg.save_dir, name, track, episode_index=ep_idx)

        # Clear references
        self._plane_track = None
        self._missile_tracks = None
        self._plane_name = None
        self._missile_names = None
