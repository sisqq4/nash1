
from dataclasses import dataclass

@dataclass
class EnvConfig:
    """Environment configuration for the missile–escape game.

    Coordinates are interpreted as *kilometers*, time step as *seconds*,
    and speeds as *kilometers per second* (converted from km/h).
    """
    # Red missile launch region is handled inside GameTheoreticLauncher;
    # here we keep a characteristic span for reward shaping only.
    region_span: float = 160.0  # km, characteristic scale for distance normalization

    # Blue aircraft initial position (km)
    blue_init_x: float = 70.0
    blue_init_y: float = 0.0
    blue_init_z: float = 5.0

    # Simulation parameters
    dt: float = 1.0          # [s] time step
    max_steps: int = 200     # max steps per episode

    # Blue aircraft dynamics
    # 2000 km/h -> km/s
    blue_max_speed: float = 2000.0 / 3600.0  # ≈0.556 km/s
    # 9g ≈ 88.3 m/s^2 ≈ 0.0883 km/s^2
    blue_accel: float = 0.09  # km/s^2 (approx 9g)

    # Missile dynamics (proportional navigation style)
    # 4900 km/h -> km/s
    missile_speed: float = 4900.0 / 3600.0  # ≈1.361 km/s
    num_missiles: int = 3
    nav_gain: float = 3.0  # base navigation gain for PN-like heading update

    # Interaction distances
    hit_radius: float = 0.015  # km (~15 m lethal radius)

    # Game-theoretic launcher parameters (Nash for initial launch positions)
    candidate_launch_count: int = 40  # how many candidate launch points are evaluated each reset
    num_blue_strategies: int = 8      # number of blue candidate escape headings in the static game
    fictitious_iters: int = 200       # iterations of fictitious play to approximate Nash
    blue_escape_distance: float = 10.0  # assumed distance blue may move in the static game payoff (km)

    # Differential-game controller for PN gains
    use_diff_game: bool = True
    diff_step_size: float = 0.2       # gradient descent step size for nav gain update
    diff_delta_gain: float = 0.2      # finite-difference perturbation for gradient
    diff_gain_min: float = 0.5        # lower bound on nav gains
    diff_gain_max: float = 8.0        # upper bound on nav gains
    diff_w_dist: float = 1.0          # weight on distance term in running cost
    diff_w_gain: float = 0.01         # weight on nav gain regularization in running cost

    # Logging / Tacview export
    save_dir: str = "outputs"         # base directory for csv/acmi export
    log_trajectories: bool = True     # whether to log trajectories to csv


@dataclass
class TrainConfig:
    """Training hyperparameters for the blue RL agent."""
    episodes: int = 500
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 50_000
    start_learning: int = 1_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 20_000  # in environment steps

    target_update_interval: int = 1_000

    # Logging
    print_interval: int = 10
