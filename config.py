
from dataclasses import dataclass
from typing import Optional

@dataclass
class EnvConfig:
    """Environment configuration for the missile–escape game.

    Units:
        - Position: kilometers (km)
        - Time: seconds (s)
        - Speed: kilometers per second (km/s)
    """

    # Overall scale (for reward shaping)
    region_span: float = 160.0  # km

    # Blue aircraft initial region (random inside this box)
    blue_x_min: float = 60.0
    blue_x_max: float = 80.0
    blue_y_min: float = -10.0
    blue_y_max: float = 10.0
    blue_z_min: float = 4.0
    blue_z_max: float = 10.0

    # Simulation
    dt: float = 0.1          # [s] physics & hit-judgement step
    max_steps: int = 1200    # episode length in steps (~120 s)

    # Blue aircraft dynamics
    blue_max_speed: float = 2000.0 / 3600.0  # km/s
    blue_accel: float = 0.09                 # km/s^2 (~9 g)
    # Blue aircraft initial heading (degrees in xy-plane, 0 along +x)
    blue_heading_min: float = 135.0
    blue_heading_max: float = 225.0

    # Missile dynamics
    missile_speed: float = 4900.0 / 3600.0   # km/s
    missile_target_speed: float = 4800.0 / 3600.0  # km/s
    missile_boost_duration: float = 5.0      # [s]
    missile_speed_decay_interval: float = 1.0  # [s]
    missile_speed_decay_factor: float = 0.99
    missile_min_speed: float = 980.0 / 3600.0  # km/s
    num_missiles: int = 3
    nav_gain: float = 3.0
    missile_max_overload_g: float = 30.0  # max lateral load factor [g]

    # Missile lifetime / energy
    missile_max_flight_time: float = 120.0   # [s]

    # Hit radius (warhead lethal radius, km)
    hit_radius: float = 0.015  # ~15 m

    # Game-theoretic launcher (position + launch time)
    candidate_launch_count: int = 32
    num_blue_strategies: int = 8
    fictitious_iters: int = 200
    blue_escape_distance: float = 10.0      # km (only for rough payoff shaping)
    max_launch_time: float = 8.0            # latest first-launch time [s]
    min_launch_interval: float = 1.0        # between launches [s]

    # Differential-game controller for PN gains
    use_diff_game: bool = False
    diff_step_size: float = 0.2
    diff_delta_gain: float = 0.2
    diff_gain_min: float = 0.5
    diff_gain_max: float = 8.0
    diff_w_dist: float = 1.0
    diff_w_gain: float = 0.01

    # Ground / terrain
    ground_crash_penalty: float = -5.0  # penalty when blue hits the ground

    # Reward shaping (units: km, km/s)
    safe_altitude_min: float = 8.0
    safe_altitude_max: float = 12.0
    safe_altitude_tolerance: float = 1.0
    height_reward_weight: float = 1.2
    distance_ratio_weight: float = 1.0
    danger_distance: float = 5.0
    engagement_range: float = 25.0
    danger_scale: float = 3.0

    # Threat evaluation parameters
    threat_heading_max: float = 0.5 * 3.141592653589793
    threat_pitch_max: float = 0.5 * 3.141592653589793
    threat_omega: float = 0.2
    threat_dist_max: float = 160.0
    threat_kd: float = 1.0
    threat_sigma: float = 1e-8
    threat_reward_relief: float = 0.25
    threat_reward_increase: float = 1.0
    threat_aggressive_threshold: float = 0.6
    threat_aggressive_scale: float = 1.5

    # Threat-driven maneuver overrides
    threat_maneuver_start: float = 0.6
    threat_maneuver_stop: float = 0.4
    threat_maneuver_steps: int = 20

    # Logging / Tacview export
    save_dir: str = "outputs"
    log_trajectories: bool = True


@dataclass
class TrainConfig:
    """Training hyperparameters for the blue RL agent."""

    episodes: int = 1000
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 50_000
    start_learning: int = 1_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 20_000

    target_update_interval: int = 1_000

    print_interval: int = 10
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_interval: int = 50
    load_checkpoint_path: Optional[str] = None
    load_blue: bool = True
    load_red: bool = False
    results_dir: str = "outputs/results"