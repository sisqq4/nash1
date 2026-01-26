
"""3D escape environment: blue aircraft vs. three red missiles."""

from __future__ import annotations

from typing import Dict, Tuple, Any, List
import math
import numpy as np

from .game_theory_launcher import GameTheoreticLauncher, LaunchRegion
from .missile_dynamics import update_blue_state, update_missiles_pn
from .aircraft_missiles import Aircraft, Missiles
from .diff_game_controller import DifferentialGameController
from .acmi_io import write_csv, write_action_csv
from . import action_space
from .threat_eval import ThreatEvaluator, ThreatParams
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
            v_min=cfg.blue_min_speed,
        )
        self.missile_model = Missiles(
            dt=cfg.dt,
            speed=cfg.missile_speed,
            max_overload_g=cfg.missile_max_overload_g,
        )

        self.blue_pos = np.zeros(3, dtype=float)
        self.blue_vel = np.zeros(3, dtype=float)

        M = cfg.num_missiles
        self.missile_pos = np.zeros((M, 3), dtype=float)
        self.missile_vel = np.zeros((M, 3), dtype=float)
        self.missile_speed = np.zeros(M, dtype=float)
        self.missile_initial_speed = np.zeros(M, dtype=float)
        self.nav_gains = np.full(M, cfg.nav_gain, dtype=float)
        self.initial_nav_gains = np.full(M, cfg.nav_gain, dtype=float)
        self.nav_gains = self.initial_nav_gains.copy()

        # Launch & lifetime
        self.missile_launch_times = np.zeros(M, dtype=float)
        self.missile_launched = np.zeros(M, dtype=bool)
        self.missile_alive = np.ones(M, dtype=bool)
        self.missile_time_alive = np.zeros(M, dtype=float)

        self.step_count = 0
        self.time = 0.0
        self.done = False
        self.prev_threat = 1.0
        self.prev_min_dist = cfg.region_span
        self.prev_blue_vel = np.zeros(3, dtype=float)
        self.initial_missile_distances = np.zeros(M, dtype=float)
        self.forced_maneuver_steps = 0
        self.threat_mode = False

        criteria = np.array(
            [
                [1, 1 / 2, 1 / 8],
                [2, 1, 1 / 6],
                [8, 6, 1],
            ],
            dtype=float,
        )
        self.threat_evaluator = ThreatEvaluator(
            ThreatParams(
                heading_max=cfg.threat_heading_max,
                pitch_max=cfg.threat_pitch_max,
                omega=cfg.threat_omega,
                dist_max=cfg.threat_dist_max,
                kd=cfg.threat_kd,
                sigma=cfg.threat_sigma,
                criteria=criteria,
            )
        )

        # Observation: blue pos (3) + blue vel (3) + rel missile pos (3*M)
        self.observation_dim = 3 + 3 + 3 * M
        self.action_dim = self.blue_model.num_strategies

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
        self._blue_action_log: List[List[float]] | None = None
        self._blue_action_name: str | None = None

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.time = 0.0
        self.done = False
        self.blue_model.reset()

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
        # Random initial velocity direction in xy-plane
        heading_min = math.radians(float(self.cfg.blue_heading_min))
        heading_max = math.radians(float(self.cfg.blue_heading_max))
        if heading_max < heading_min:
            heading_min, heading_max = heading_max, heading_min
        heading = self.rng.uniform(heading_min, heading_max)
        v_dir = np.array([math.cos(heading), math.sin(heading), 0.0], dtype=float)
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
        if self.initial_nav_gains.shape[0] != self.cfg.num_missiles:
            self.initial_nav_gains = np.full(self.cfg.num_missiles, self.cfg.nav_gain, dtype=float)
        self.nav_gains = self.initial_nav_gains.copy()

        # Lifetime
        self.missile_alive[:] = True
        self.missile_time_alive[:] = 0.0
        self.prev_threat = 1.0
        self.forced_maneuver_steps = 0
        self.threat_mode = False
        self.initial_missile_distances = np.linalg.norm(
            self.missile_pos - self.blue_pos[None, :],
            axis=1,
        )
        self.prev_min_dist = float(np.min(self.initial_missile_distances))
        self.prev_blue_vel = self.blue_vel.copy()

        if self.log_enabled:
            self._init_logging()
            self._log_current_state()

        return self._get_obs()

    def get_red_params(self) -> Dict[str, Any]:
        return {
            "launcher_state": self.launcher.get_state(),
            "nav_gains": self.nav_gains.copy(),
        }

    def set_red_params(self, params: Dict[str, Any]) -> None:
        launcher_state = params.get("launcher_state")
        if launcher_state is not None:
            self.launcher.set_state(launcher_state)

        nav_gains = params.get("nav_gains")
        if nav_gains is not None:
            gains = np.asarray(nav_gains, dtype=float)
            if gains.shape[0] != self.cfg.num_missiles:
                gains = np.full(self.cfg.num_missiles, self.cfg.nav_gain, dtype=float)
            self.initial_nav_gains = gains.copy()

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("reset() must be called before stepping a finished episode")

        action = int(action)
        if not (0 <= action < self.action_dim):
            raise ValueError(f"Invalid action {action}")

        dt = self.cfg.dt

        prev_blue_pos = self.blue_pos.copy()
        prev_blue_vel = self.blue_vel.copy()
        prev_missile_pos = self.missile_pos.copy()

        # 1) Update blue aircraft
        threat_pre = self._compute_threat()
        if threat_pre >= self.cfg.threat_maneuver_start:
            self.threat_mode = False
        elif threat_pre <= self.cfg.threat_maneuver_stop:
            self.threat_mode = False

        if self.threat_mode and not self.blue_model.has_forced_actions():
            self._schedule_evasive_maneuver(threat_pre)
        self.blue_pos, self.blue_vel = self.blue_model.step(
            self.blue_pos,
            self.blue_vel,
            action,
        )
        if self.forced_maneuver_steps > 0:
            self.forced_maneuver_steps -= 1

        # Enforce ground (terrain) for blue
        crashed = False
        if self.blue_pos[2] <= 0.0:
            self.blue_pos[2] = 0.0
            crashed = True

        # 2) Update time and possibly launch new missiles
        self.step_count += 1
        self.time += dt
        if self.log_enabled and self._blue_action_log is not None:
            self._blue_action_log.append([float(self.time), int(action)])

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

                speed = min(speed, self.cfg.missile_max_speed)
                drag = self._missile_drag_decel(self.missile_pos[i, 2], speed)
                speed = max(speed - drag * dt, 0.0)

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

        # 6) Destroy missiles that lose target (blue outside +/- 60 deg FOV)
        fov_cos = math.cos(math.radians(60.0))
        for i in range(self.cfg.num_missiles):
            if not (self.missile_launched[i] and self.missile_alive[i]):
                continue
            rel = self.blue_pos - self.missile_pos[i]
            rel_norm = float(np.linalg.norm(rel))
            vel_norm = float(np.linalg.norm(self.missile_vel[i]))
            if rel_norm < 1e-6 or vel_norm < 1e-6:
                continue
            cos_angle = float(np.dot(self.missile_vel[i], rel) / (vel_norm * rel_norm))
            if cos_angle < fov_cos:
                self.missile_alive[i] = False
                self.nav_gains[i] = 0.0
                self.missile_vel[i] = 0.0
                self.missile_speed[i] = 0.0

        # 7) Update missile lifetime / energy
        self.missile_time_alive[self.missile_launched & self.missile_alive] += dt
        expired = self.missile_time_alive >= self.cfg.missile_max_flight_time
        self.missile_alive[expired] = False
        self.nav_gains[expired] = 0.0
        self.missile_speed[expired] = 0.0

        # 8) Enforce ground for missiles: z <= 0 destroys the missile
        for i in range(self.cfg.num_missiles):
            if self.missile_launched[i] and self.missile_alive[i] and self.missile_pos[i, 2] <= 0.0:
                self.missile_pos[i, 2] = 0.0
                self.missile_alive[i] = False
                self.nav_gains[i] = 0.0
                self.missile_vel[i] = 0.0
                self.missile_speed[i] = 0.0

        if self.log_enabled:
            self._log_current_state()

        # 9) Hit detection (line-segment / sphere)
        hit, min_dist = self._check_hits(prev_blue_pos, prev_missile_pos)

        timeout = self.step_count >= self.cfg.max_steps
        missiles_exhausted = not np.any(self.missile_alive)

        # 10) Terminal conditions and reward
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
            reward = self._compute_reward(min_dist, prev_blue_vel)

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
            "threat": float(self.prev_threat),
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

    def _height_reward(self, altitude: float) -> float:
        safe_min = self.cfg.safe_altitude_min
        safe_max = self.cfg.safe_altitude_max
        tolerance = self.cfg.safe_altitude_tolerance
        hard_min = safe_min - tolerance
        hard_max = safe_max + tolerance

        if altitude < hard_min or altitude > hard_max:
            return -1.5
        if altitude < safe_min:
            ratio = (altitude - hard_min) / (safe_min - hard_min)
            return -1.0 + 2.0 * ratio
        if altitude > safe_max:
            ratio = (hard_max - altitude) / (hard_max - safe_max)
            return -1.0 + 2.0 * ratio

        center = (safe_min + safe_max) / 2.0
        span = (safe_max - safe_min) / 2.0
        offset = (altitude - center) / span
        return 1.0 - offset ** 2

    def _distance_reward(self) -> float:
        rd = 0.0
        rd_min = float("inf")
        danger_distance = max(self.cfg.danger_distance, 1e-6)
        engagement_range = self.cfg.engagement_range
        scale = engagement_range / danger_distance

        for i in range(self.cfg.num_missiles):
            if not (self.missile_launched[i] and self.missile_alive[i]):
                continue
            dist = float(np.linalg.norm(self.missile_pos[i] - self.blue_pos))
            dist = max(dist, 1e-6)
            if scale <= 1.0:
                rd = -1.5
            else:
                rd = math.log(dist / danger_distance) / math.log(scale)
            rd = max(min(rd, 1.0), -1.5)
            if rd < rd_min:
                rd_min = rd

        return 0.0 if rd_min == float("inf") else rd_min

    def _compute_threat(self) -> float:
        threat = 0.0
        for i in range(self.cfg.num_missiles):
            if not (self.missile_launched[i] and self.missile_alive[i]):
                continue
            treat = self.threat_evaluator.evaluate(
                self.blue_pos,
                self.blue_vel,
                self.missile_pos[i],
                self.missile_vel[i],
            )
            if treat > threat:
                threat = treat
        return float(threat)

    def _compute_reward(self, min_dist: float, prev_blue_vel: np.ndarray) -> float:
        primary_idx = self._select_primary_missile()
        azimuth_deg = (
            self._compute_missile_azimuth_deg(primary_idx) if primary_idx is not None else 0.0
        )

        mode = self.cfg.reward_mode
        if mode == "short_range":
            reward = self._reward_short_range(min_dist, prev_blue_vel)
        elif mode == "mid_small_azimuth":
            reward = self._reward_mid_small_azimuth(min_dist, azimuth_deg, primary_idx)
        elif mode == "mid_large_azimuth":
            reward = self._reward_mid_large_azimuth(min_dist, azimuth_deg, primary_idx)
        else:
            if min_dist <= self.cfg.short_range_distance:
                reward = self._reward_short_range(min_dist, prev_blue_vel)
            elif azimuth_deg <= self.cfg.small_azimuth_deg:
                reward = self._reward_mid_small_azimuth(min_dist, azimuth_deg, primary_idx)
            else:
                reward = self._reward_mid_large_azimuth(min_dist, azimuth_deg, primary_idx)

        threat = self._compute_threat()
        if threat <= self.prev_threat:
            reward += self.cfg.threat_reward_relief * threat
        else:
            reward -= self.cfg.threat_reward_increase * threat
        if threat > self.cfg.threat_aggressive_threshold:
            reward -= self.cfg.threat_aggressive_scale * (threat - self.cfg.threat_aggressive_threshold)

        self.prev_threat = threat
        self.prev_min_dist = float(min_dist)
        self.prev_blue_vel = self.blue_vel.copy()
        return float(reward)

    def _select_primary_missile(self) -> int | None:
        active = np.where(self.missile_launched & self.missile_alive)[0]
        if active.size == 0:
            return None
        dists = np.linalg.norm(self.missile_pos[active] - self.blue_pos[None, :], axis=1)
        return int(active[int(np.argmin(dists))])

    def _compute_missile_azimuth_deg(self, idx: int) -> float:
        rel = self.blue_pos - self.missile_pos[idx]
        rel_xy = np.array([rel[0], rel[1]], dtype=float)
        vel_xy = np.array([self.missile_vel[idx][0], self.missile_vel[idx][1]], dtype=float)
        rel_norm = float(np.linalg.norm(rel_xy))
        vel_norm = float(np.linalg.norm(vel_xy))
        if rel_norm < 1e-6 or vel_norm < 1e-6:
            return 0.0
        cos_angle = float(np.dot(rel_xy, vel_xy) / (rel_norm * vel_norm))
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        return float(math.degrees(math.acos(cos_angle)))

    def _compute_turn_rate_deg(self, prev_vel: np.ndarray) -> float:
        prev_xy = np.array([prev_vel[0], prev_vel[1]], dtype=float)
        curr_xy = np.array([self.blue_vel[0], self.blue_vel[1]], dtype=float)
        prev_norm = float(np.linalg.norm(prev_xy))
        curr_norm = float(np.linalg.norm(curr_xy))
        if prev_norm < 1e-6 or curr_norm < 1e-6:
            return 0.0
        cos_angle = float(np.dot(prev_xy, curr_xy) / (prev_norm * curr_norm))
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = float(math.degrees(math.acos(cos_angle)))
        return angle_deg / max(self.cfg.dt, 1e-6)

    def _reward_short_range(self, min_dist: float, prev_blue_vel: np.ndarray) -> float:
        reward = 0.0
        distance_gain = (min_dist - self.prev_min_dist) / max(self.cfg.short_range_distance, 1e-6)
        reward += self.cfg.short_range_distance_weight * distance_gain

        roll_rad = self.blue_model.roll_rad or 0.0
        roll_target = math.radians(self.cfg.short_turn_roll_target_deg)
        roll_score = min(abs(roll_rad) / max(roll_target, 1e-6), 1.0)
        reward += self.cfg.short_range_roll_weight * roll_score

        turn_rate = self._compute_turn_rate_deg(prev_blue_vel)
        turn_score = min(turn_rate / 180.0, 1.0)
        reward += self.cfg.short_range_turn_weight * turn_score

        speed = float(np.linalg.norm(self.blue_vel))
        speed_norm = (speed - self.cfg.blue_min_speed) / max(
            self.cfg.blue_max_speed - self.cfg.blue_min_speed, 1e-6
        )
        reward += self.cfg.short_range_speed_weight * speed_norm

        reward += self.cfg.short_range_height_weight * self._height_reward(self.blue_pos[2])
        return reward

    def _reward_mid_small_azimuth(
        self,
        min_dist: float,
        azimuth_deg: float,
        primary_idx: int | None,
    ) -> float:
        reward = 0.0
        azimuth_score = 1.0 - min(azimuth_deg / max(self.cfg.small_azimuth_deg, 1e-6), 1.0)
        reward += self.cfg.mid_small_azimuth_weight * azimuth_score

        reward += self.cfg.mid_small_height_weight * self._height_reward(self.blue_pos[2])

        if primary_idx is not None:
            rel = self.blue_pos - self.missile_pos[primary_idx]
            rel_norm = float(np.linalg.norm(rel))
            if rel_norm > 1e-6:
                opposite = float(np.dot(self.blue_vel, rel) / (rel_norm * max(np.linalg.norm(self.blue_vel), 1e-6)))
                reward += self.cfg.mid_small_opposite_weight * max(opposite, 0.0)

        speed = float(np.linalg.norm(self.blue_vel))
        speed_norm = (speed - self.cfg.blue_min_speed) / max(
            self.cfg.blue_max_speed - self.cfg.blue_min_speed, 1e-6
        )
        reward += self.cfg.mid_small_speed_weight * speed_norm

        roll_rad = self.blue_model.roll_rad or 0.0
        roll_penalty = abs(roll_rad) / math.pi
        climb_angle = math.degrees(math.atan2(self.blue_vel[2], max(np.linalg.norm(self.blue_vel[:2]), 1e-6)))
        level_penalty = (abs(climb_angle) / 90.0) + roll_penalty
        reward -= self.cfg.mid_small_level_weight * min(level_penalty, 1.0)
        return reward

    def _reward_mid_large_azimuth(
        self,
        min_dist: float,
        azimuth_deg: float,
        primary_idx: int | None,
    ) -> float:
        reward = 0.0
        distance_gain = (min_dist - self.prev_min_dist) / max(self.cfg.short_range_distance, 1e-6)
        reward += self.cfg.mid_large_distance_weight * distance_gain

        azimuth_score = 1.0 - min(azimuth_deg / 180.0, 1.0)
        reward += self.cfg.mid_large_azimuth_weight * azimuth_score

        reward += self.cfg.mid_large_height_weight * self._height_reward(self.blue_pos[2])

        speed = float(np.linalg.norm(self.blue_vel))
        speed_norm = (speed - self.cfg.blue_min_speed) / max(
            self.cfg.blue_max_speed - self.cfg.blue_min_speed, 1e-6
        )
        reward += self.cfg.mid_large_speed_weight * speed_norm

        roll_rad = self.blue_model.roll_rad or 0.0
        roll_penalty = abs(roll_rad) / math.pi
        climb_angle = math.degrees(math.atan2(self.blue_vel[2], max(np.linalg.norm(self.blue_vel[:2]), 1e-6)))
        level_penalty = (abs(climb_angle) / 90.0) + roll_penalty
        reward -= self.cfg.mid_large_level_weight * min(level_penalty, 1.0)

        if min_dist <= self.cfg.short_range_distance + self.cfg.short_range_buffer:
            roll_zero_score = 1.0 - min(abs(roll_rad) / math.radians(30.0), 1.0)
            reward += self.cfg.mid_large_roll_zero_weight * roll_zero_score
        return reward

    def _air_density(self, altitude_km: float) -> float:
        altitude_m = max(0.0, altitude_km * 1000.0)
        rho0 = 1.225
        t0 = 288.15
        if altitude_m <= 11000.0:
            t = t0 - 0.0065 * altitude_m
            return rho0 * (t / t0) ** 4.25588
        if altitude_m <= 20000.0:
            return 0.36392 * math.exp((-altitude_m + 11000.0) / 6341.62)
        t = 216.65 + 0.001 * (altitude_m - 20000.0)
        return 0.088035 * (t / 216.65) ** -35.1632

    def _missile_drag_decel(self, altitude_km: float, speed_km_s: float) -> float:
        rho = self._air_density(altitude_km)
        speed_m_s = max(0.0, speed_km_s * 1000.0)
        drag_n = 0.5 * rho * speed_m_s ** 2 * self.cfg.missile_cd * self.cfg.missile_ref_area_m2
        accel_m_s2 = drag_n / max(self.cfg.missile_mass_kg, 1e-6)
        return accel_m_s2 / 1000.0

    def _schedule_evasive_maneuver(self, threat: float) -> None:
        if self.forced_maneuver_steps > 0:
            return
        if threat <= 0.0:
            return

        primitives = action_space.get_simple_list()
        if len(primitives) < 11:
            return

        idx_active = np.where(self.missile_launched & self.missile_alive)[0]
        roll_sign = 1.0
        climb = True
        if idx_active.size > 0:
            i = idx_active[0]
            rel = self.missile_pos[i] - self.blue_pos
            roll_sign = 1.0 if rel[1] >= 0 else -1.0
            climb = rel[2] >= 0

        base_turn = primitives[9] if roll_sign >= 0 else primitives[10]
        base_vert = primitives[5] if climb else primitives[7]

        steps = max(1, int(self.cfg.threat_maneuver_steps))
        half = steps // 2
        sequence = [base_turn] * half + [base_vert] * (steps - half)
        self.blue_model.force_actions([np.asarray(a, dtype=float) for a in sequence])
        self.forced_maneuver_steps = steps

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
        self._blue_action_log = []
        self._blue_action_name = f"plane_blue_actions.{plane_id}"

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

        if self._blue_action_log and self._blue_action_name is not None:
            write_action_csv(
                self.cfg.save_dir,
                self._blue_action_name,
                self._blue_action_log,
                episode_index=ep_idx,
            )

        self._plane_track = None
        self._missile_tracks = None
        self._plane_name = None
        self._missile_names = None
        self._blue_action_log = None
        self._blue_action_name = None
