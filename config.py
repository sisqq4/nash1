
from dataclasses import dataclass

@dataclass
class EnvConfig:
    """Environment configuration for the missile–escape game."""
    # Red missile launch region (axis-aligned cube)
    region_min: float = -50.0
    region_max: float = 50.0

    # Blue aircraft initial distance from origin (outside the red launch cube)
    blue_init_radius: float = 120.0

    # Simulation parameters
    dt: float = 0.1
    max_steps: int = 200

    # Blue aircraft dynamics
    blue_max_speed: float = 30.0
    blue_accel: float = 10.0

    # Missile dynamics (proportional navigation style)
    missile_speed: float = 50.0
    num_missiles: int = 3
    nav_gain: float = 3.0  # navigation gain for PN-like heading update

    # Interaction distances
    hit_radius: float = 5.0

    # Game-theoretic launcher parameters
    candidate_launch_count: int = 40  # how many candidate launch points are evaluated each reset
    num_blue_strategies: int = 8      # number of blue candidate escape headings in the static game
    fictitious_iters: int = 200       # iterations of fictitious play to approximate Nash
    blue_escape_distance: float = 60.0  # assumed distance blue may move in the static game payoff


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
