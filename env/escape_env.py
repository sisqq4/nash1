
"""3D escape environment: blue aircraft vs. three red missiles."""

from __future__ import annotations

from typing import Dict, Tuple, Any, List
import math
import numpy as np

from .game_theory_launcher import GameTheoreticLauncher, LaunchRegion
from .missile_dynamics import update_blue_state, update_missiles_pn
from .aircraft_missiles import Aircraft, Missiles
from .diff_game_controller import DifferentialGameController
from .acmi_io import write_csv
from config import EnvConfig



class EscapeEnv:
    def __init__(self, cfg: EnvConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        region = LaunchRegion(
            num_missiles=cfg.num_missiles,
            candidate_launch_count=cfg.candidate_launch_count,
            num_blue_strategies=cfg.num_blue_strategies,
            fictitious_iters=cfg.fictitious_iters,
            blue_escape_distance=cfg.blue_escape_distance,
            max_launch_time=cfg.max_launch_time,
            min_launch_interval=cfg.min_launch_interval,
        )
        self.launcher = GameTheoreticLauncher(region, rng=self.rng)
        self.diff_ctrl = DifferentialGameController(cfg) if cfg.use_diff_game else None
        # Dynamic models for blue aircraft and red missiles
        self.blue_model = Aircraft(
            dt=cfg.dt,
            accel_mag=cfg.blue_accel,
            v_max=cfg.blue_max_speed,
        )
        self.missile_model = Missiles(
            dt=cfg.dt,
            speed=cfg.missile_speed,
        )

        self.blue_pos = np.zeros(3, dtype=float)
        self.blue_vel = np.zeros(3, dtype=float)

        M = cfg.num_missiles
        self.missile_pos = np.zeros((M, 3), dtype=float)
        self.missile_vel = np.zeros((M, 3), dtype=float)
        self.missile_speed = np.zeros(M, dtype=float)
        self.missile_initial_speed = np.zeros(M, dtype=float)
        self.nav_gains = np.full(M, cfg.nav_gain, dtype=float)

        # Launch & lifetime
        self.missile_launch_times = np.zeros(M, dtype=float)
        self.missile_launched = np.zeros(M, dtype=bool)
        self.missile_alive = np.ones(M, dtype=bool)
        self.missile_time_alive = np.zeros(M, dtype=float)

        self.step_count = 0
        self.time = 0.0
        self.done = False

        # Observation: blue pos (3) + blue vel (3) + rel missile pos (3*M)
        self.observation_dim = 3 + 3 + 3 * M
        self.action_dim = 7

        # Logging / episode indexing
        self.log_enabled = bool(cfg.log_trajectories)
        self.save_dir = cfg.save_dir
        self.episode_index = 0
        self.plane_global_id = 0
        self.missile_global_id = 0

        self._plane_track: List[List[float]] | None = None
        self._plane_name: str | None = None
        self._missile_tracks: List[List[List[float]]] | None = None
        self._missile_names: List[str] | None = None

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.time = 0.0
        self.done = False

        # Blue initial position: random in box (above ground)
        self.blue_pos = np.array(
            [
                self.rng.uniform(self.cfg.blue_x_min, self.cfg.blue_x_max),
                self.rng.uniform(self.cfg.blue_y_min, self.cfg.blue_y_max),
                self.rng.uniform(self.cfg.blue_z_min, self.cfg.blue_z_max),
            ],
            dtype=float,
        )

        # Random initial velocity direction
        v_dir = self.rng.normal(size=3)
        v_norm = np.linalg.norm(v_dir)
        if v_norm < 1e-6:
            v_dir = np.array([1.0, 0.0, 0.0])
            v_norm = 1.0
        v_dir /= v_norm
        self.blue_vel = v_dir * self.cfg.blue_max_speed

        # Red: matrix game for launch positions + times
        launch_pos, launch_times = self.launcher.compute_launch_plan(
            blue_initial_pos=self.blue_pos,
            blue_speed=self.cfg.blue_max_speed,
            missile_speed=self.cfg.missile_speed,
        )
        self.missile_pos = launch_pos.reshape(self.cfg.num_missiles, 3)
        self.missile_launch_times = launch_times.astype(float)
        self.missile_launched[:] = False

        # Velocities start at zero (not yet launched)
        self.missile_vel.fill(0.0)
        self.missile_speed.fill(0.0)
        self.missile_initial_speed.fill(0.0)
        self.nav_gains.fill(self.cfg.nav_gain)

        # Lifetime
        self.missile_alive[:] = True
        self.missile_time_alive[:] = 0.0

        if self.log_enabled:
            self._init_logging()
            self._log_current_state()

        return self._get_obs()

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("reset() must be called before stepping a finished episode")

        action = int(action)
        if not (0 <= action < self.action_dim):
            raise ValueError(f"Invalid action {action}")

        dt = self.cfg.dt

        prev_blue_pos = self.blue_pos.copy()
        prev_missile_pos = self.missile_pos.copy()

        # 1) Update blue aircraft
        self.blue_pos, self.blue_vel = self.blue_model.step(
            self.blue_pos,
            self.blue_vel,
            action,
        )

        # Enforce ground (terrain) for blue
        crashed = False
        if self.blue_pos[2] <= 0.0:
            self.blue_pos[2] = 0.0
            crashed = True

        # 2) Update time and possibly launch new missiles
        self.step_count += 1
        self.time += dt

        for i in range(self.cfg.num_missiles):
            if (
                (not self.missile_launched[i])
                and self.missile_alive[i]
                and self.time >= self.missile_launch_times[i]
            ):
                # Launch missile i: set initial velocity toward current blue position
                direction = self.blue_pos - self.missile_pos[i]
                n = np.linalg.norm(direction)
                if n < 1e-6:
                    direction = np.array([1.0, 0.0, 0.0])
                    n = 1.0
                direction /= n
                initial_speed = float(np.linalg.norm(self.blue_vel))
                self.missile_initial_speed[i] = initial_speed
                self.missile_speed[i] = initial_speed
                self.missile_vel[i] = direction * initial_speed
                self.missile_launched[i] = True
                self.missile_time_alive[i] = 0.0

        # 3) Update missile speed profiles for launched & alive missiles
        idx_active = np.where(self.missile_launched & self.missile_alive)[0]
        if idx_active.size > 0:
            for i in idx_active:
                next_time = self.missile_time_alive[i] + dt
                if next_time <= self.cfg.missile_boost_duration:
                    progress = next_time / self.cfg.missile_boost_duration
                    speed = self.missile_initial_speed[i] + (
                            self.cfg.missile_target_speed - self.missile_initial_speed[i]
                    ) * progress
                else:
                    decay_steps = int(
                        math.floor(
                            (next_time - self.cfg.missile_boost_duration)
                            / self.cfg.missile_speed_decay_interval
                        )
                    )
                    speed = self.cfg.missile_target_speed * (
                            self.cfg.missile_speed_decay_factor ** decay_steps
                    )

                if speed < self.cfg.missile_min_speed:
                    self.missile_alive[i] = False
                    self.nav_gains[i] = 0.0
                    self.missile_vel[i] = 0.0
                    self.missile_speed[i] = 0.0
                else:
                    self.missile_speed[i] = speed
        # 4) Differential-game update of nav_gains for launched & alive missiles
        if self.diff_ctrl is not None:
            idx = np.where(self.missile_launched & self.missile_alive)[0]
            if idx.size > 0:
                nav_sub = self.nav_gains[idx].copy()
                new_nav = self.diff_ctrl.update_nav_gains(
                    self.blue_pos,
                    self.blue_vel,
                    self.missile_pos[idx],
                    self.missile_vel[idx],
                    nav_sub,
                    dt,
                )
                self.nav_gains[idx] = new_nav
                self.nav_gains[~self.missile_alive] = 0.0

        # 5) PN update for launched missiles
        idx_launched = np.where(self.missile_launched)[0]
        if idx_launched.size > 0:
            sub_pos, sub_vel = self.missile_model.step(
                self.missile_pos[idx_launched],
                self.missile_vel[idx_launched],
                self.missile_speed[idx_launched],
                self.blue_pos,
                self.blue_vel,
                self.nav_gains[idx_launched],
            )
            self.missile_pos[idx_launched] = sub_pos
            self.missile_vel[idx_launched] = sub_vel

        # 6) Update missile lifetime / energy
        self.missile_time_alive[self.missile_launched & self.missile_alive] += dt
        expired = self.missile_time_alive >= self.cfg.missile_max_flight_time
        self.missile_alive[expired] = False
        self.nav_gains[expired] = 0.0
        self.missile_speed[expired] = 0.0

        # 7) Enforce ground for missiles: z <= 0 destroys the missile
        for i in range(self.cfg.num_missiles):
            if self.missile_launched[i] and self.missile_alive[i] and self.missile_pos[i, 2] <= 0.0:
                self.missile_pos[i, 2] = 0.0
                self.missile_alive[i] = False
                self.nav_gains[i] = 0.0
                self.missile_vel[i] = 0.0
                self.missile_speed[i] = 0.0

        if self.log_enabled:
            self._log_current_state()

        # 8) Hit detection (line-segment / sphere)
        hit, min_dist = self._check_hits(prev_blue_pos, prev_missile_pos)

        timeout = self.step_count >= self.cfg.max_steps
        missiles_exhausted = not np.any(self.missile_alive)

        # 9) Terminal conditions and reward
        if crashed:
            reward = self.cfg.ground_crash_penalty
            self.done = True
        elif hit:
            reward = -1.0
            self.done = True
        elif missiles_exhausted:
            reward = 1.0
            self.done = True
        elif timeout:
            reward = 1.0
            self.done = True
        else:
            reward = 0.01 * (min_dist / self.cfg.region_span)

        if self.done and self.log_enabled:
            self._flush_logs_to_csv()

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "time": float(self.time),
            "step": int(self.step_count),
            "min_dist": float(min_dist),
            "hit": bool(hit),
            "timeout": bool(timeout),
            "crashed": bool(crashed),
            "missiles_exhausted": bool(missiles_exhausted),
            "nav_gains": self.nav_gains.copy(),
            "missile_alive": self.missile_alive.copy(),
            "missile_launched": self.missile_launched.copy(),
            "missile_time_alive": self.missile_time_alive.copy(),
            "launch_times": self.missile_launch_times.copy(),
        }
        return obs, float(reward), bool(self.done), info

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        rel_missiles = self.missile_pos - self.blue_pos[None, :]
        obs = np.concatenate(
            [self.blue_pos, self.blue_vel, rel_missiles.reshape(-1)],
            axis=0,
        )
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Hit detection helpers
    # ------------------------------------------------------------------
    def _segment_sphere_hit(
        self,
        r0: np.ndarray,
        r1: np.ndarray,
        radius: float,
    ) -> Tuple[bool, float]:
        d = r1 - r0
        a = float(np.dot(d, d))
        if a < 1e-12:
            dist0 = float(np.linalg.norm(r0))
            return dist0 <= radius, dist0
        t = -float(np.dot(r0, d)) / a
        t_clamped = max(0.0, min(1.0, t))
        closest = r0 + t_clamped * d
        dist = float(np.linalg.norm(closest))
        hit = dist <= radius
        return hit, dist

    def _check_hits(
        self,
        prev_blue_pos: np.ndarray,
        prev_missile_pos: np.ndarray,
    ) -> Tuple[bool, float]:
        hit_any = False
        min_dist = float("inf")

        for i in range(self.cfg.num_missiles):
            if not self.missile_launched[i]:
                continue
            r0 = prev_missile_pos[i] - prev_blue_pos
            r1 = self.missile_pos[i] - self.blue_pos
            hit, dist = self._segment_sphere_hit(r0, r1, self.cfg.hit_radius)
            if dist < min_dist:
                min_dist = dist
            if hit:
                hit_any = True
                break

        if not np.isfinite(min_dist):
            dists = np.linalg.norm(
                self.missile_pos[self.missile_launched] - self.blue_pos[None, :],
                axis=1,
            )
            if dists.size > 0:
                min_dist = float(np.min(dists))
            else:
                min_dist = self.cfg.region_span
        return hit_any, min_dist

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _init_logging(self) -> None:
        self.episode_index += 1

        self.plane_global_id += 1
        plane_id = self.plane_global_id
        self._plane_name = f"plane_blue.{plane_id}.0"

        self._missile_tracks = []
        self._missile_names = []
        for i in range(self.cfg.num_missiles):
            self.missile_global_id += 1
            mid = self.missile_global_id
            # Encode launch step index (integer) into filename so ACMI can
            # start each missile at its launch time.
            launch_step = int(round(self.missile_launch_times[i] / self.cfg.dt))
            start_token = str(max(0, launch_step))
            name = f"missile_red.{mid}.{start_token}"
            self._missile_names.append(name)
            self._missile_tracks.append([])

        self._plane_track = []

    def _compute_orientation(self, vel: np.ndarray) -> Tuple[float, float, float]:
        vx, vy, vz = vel
        horiz = math.sqrt(vx * vx + vy * vy)
        yaw = math.degrees(math.atan2(vy, vx))
        pitch = math.degrees(math.atan2(vz, horiz))
        roll = 0.0
        return roll, pitch, yaw

    def _log_current_state(self) -> None:
        if self._plane_track is None or self._missile_tracks is None:
            return

        # Plane is always visible from t=0
        roll, pitch, yaw = self._compute_orientation(self.blue_vel)
        self._plane_track.append(
            [
                float(self.blue_pos[0]),
                float(self.blue_pos[1]),
                float(self.blue_pos[2]),
                roll,
                pitch,
                yaw,
            ]
        )

        # Missiles are only logged *after* they have been launched,
        # so they are invisible in Tacview before launch.
        for i in range(self.cfg.num_missiles):
            if not self.missile_launched[i]:
                continue
            m_pos = self.missile_pos[i]
            m_vel = self.missile_vel[i]
            roll_m, pitch_m, yaw_m = self._compute_orientation(m_vel)
            self._missile_tracks[i].append(
                [
                    float(m_pos[0]),
                    float(m_pos[1]),
                    float(m_pos[2]),
                    roll_m,
                    pitch_m,
                    yaw_m,
                ]
            )

    def _flush_logs_to_csv(self) -> None:
        if (
            self._plane_track is None
            or self._missile_tracks is None
            or self._plane_name is None
            or self._missile_names is None
        ):
            return

        ep_idx = self.episode_index

        if self._plane_track:
            write_csv(self.cfg.save_dir, self._plane_name, self._plane_track, episode_index=ep_idx)

        for track, name in zip(self._missile_tracks, self._missile_names):
            if track:
                write_csv(self.cfg.save_dir, name, track, episode_index=ep_idx)

        self._plane_track = None
        self._missile_tracks = None
        self._plane_name = None
        self._missile_names = None
