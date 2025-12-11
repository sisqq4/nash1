
"""Game-theoretic style launcher for the red missiles.

In this simplified implementation, we treat the red side's decision as
selecting launch positions for each missile inside a 3D box (the launch region).
Given the blue aircraft's initial position, the launcher evaluates a set of
candidate positions and selects those that are most threatening.

Threat is approximated here by the initial straight-line distance between
blue and candidate launch point (shorter distance = higher threat).

This module is designed so that a more sophisticated game-theoretic solver
(e.g., solving a static game over a discrete strategy set) can be plugged in
without touching the RL training code.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class LaunchRegion:
    region_min: float
    region_max: float
    num_missiles: int
    candidate_launch_count: int


class GameTheoreticLauncher:
    """Compute red's launch positions based on a simple one-shot game.

    Interface:
        launcher = GameTheoreticLauncher(region_cfg)
        launch_positions = launcher.compute_launch_positions(blue_initial_pos)

    where `launch_positions` has shape (num_missiles, 3).
    """

    def __init__(self, region: LaunchRegion, rng: np.random.Generator | None = None) -> None:
        self.region = region
        self.rng = rng or np.random.default_rng()

    def _sample_candidates(self) -> np.ndarray:
        """Sample candidate launch positions uniformly inside the cube region.

        Returns:
            positions: np.ndarray of shape (K, 3)
        """
        low = self.region.region_min
        high = self.region.region_max
        k = self.region.candidate_launch_count
        return self.rng.uniform(low=low, high=high, size=(k, 3))

    def compute_launch_positions(self, blue_initial_pos: np.ndarray) -> np.ndarray:
        """Return num_missiles launch positions inside the region.

        Args:
            blue_initial_pos: array-like of shape (3,)

        Returns:
            positions: np.ndarray of shape (num_missiles, 3)
        """
        blue_initial_pos = np.asarray(blue_initial_pos, dtype=float).reshape(3)
        candidates = self._sample_candidates()  # (K, 3)

        # Threat metric: initial distance (smaller is better for red).
        diffs = candidates - blue_initial_pos[None, :]
        dists = np.linalg.norm(diffs, axis=1)  # (K,)

        # Select the most threatening candidate positions.
        k = min(self.region.num_missiles, len(candidates))
        idx = np.argsort(dists)[:k]
        selected = candidates[idx]

        # If fewer candidates than missiles, tile them (unlikely with reasonable config).
        if selected.shape[0] < self.region.num_missiles:
            reps = int(np.ceil(self.region.num_missiles / selected.shape[0]))
            selected = np.tile(selected, (reps, 1))[: self.region.num_missiles]

        return selected
