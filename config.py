
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

    # Missile dynamics
    missile_speed: float = 50.0
    num_missiles: int = 3

    # Interaction distances
    hit_radius: float = 5.0
    # If the episode runs to max_steps without hit we treat as a successful escape

    # Game-theoretic launcher parameters
    candidate_launch_count: int = 40  # how many candidate launch points are evaluated each reset


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
