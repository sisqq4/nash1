
"""Game-theoretic launcher for the red missiles (Nash-based).

In this implementation, the red side's decision is to select launch positions
for each missile inside a 3D box (the launch region). Given the blue aircraft's
initial position, the launcher builds a *discrete zero-sum game* between:

- Row player (red): chooses one of K candidate launch points within:
    x in [0, 20] km, y in [-10, 10] km, z in [-10, 10] km;
- Column player (blue): chooses one of B candidate initial escape headings.

The payoff matrix entry A[i, j] is defined as the *distance* between the i-th
candidate launch point and the blue aircraft's *predicted position* after a
short escape maneuver along heading j. Red wants to MINIMIZE this distance,
blue wants to MAXIMIZE it.

We then compute an approximate Nash equilibrium of this zero-sum game using
fictitious play, and sample red's launch positions from the equilibrium mixed
strategy.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class LaunchRegion:
    num_missiles: int
    candidate_launch_count: int
    num_blue_strategies: int = 8
    fictitious_iters: int = 200
    blue_escape_distance: float = 10.0  # km


class GameTheoreticLauncher:
    """Compute red's launch positions from a zero-sum Nash game.

    Interface:
        launcher = GameTheoreticLauncher(region_cfg)
        launch_positions = launcher.compute_launch_positions(blue_initial_pos)

    where `launch_positions` has shape (num_missiles, 3).
    """

    def __init__(self, region: LaunchRegion, rng: np.random.Generator | None = None) -> None:
        self.region = region
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_launch_positions(self, blue_initial_pos: np.ndarray) -> np.ndarray:
        """Return num_missiles launch positions inside the region.

        Args:
            blue_initial_pos: array-like of shape (3,)

        Returns:
            positions: np.ndarray of shape (num_missiles, 3)
        """
        blue_initial_pos = np.asarray(blue_initial_pos, dtype=float).reshape(3)
        candidates = self._sample_candidates()  # (K, 3)

        # Build zero-sum payoff matrix and approximate Nash equilibrium for red.
        A = self._build_payoff_matrix(candidates, blue_initial_pos)
        red_mixed = self._fictitious_play_minimax(A)

        # Sample missiles' launch positions from red's equilibrium mixed strategy.
        K = candidates.shape[0]
        probs = np.maximum(red_mixed, 0.0)
        if probs.sum() <= 0:
            probs = np.ones(K, dtype=float) / K
        else:
            probs = probs / probs.sum()

        # Ensure we don't request more distinct samples than non-zero support
        # when using replace=False.
        nonzero = int((probs > 1e-12).sum())
        replace = self.region.num_missiles > nonzero

        idx = self.rng.choice(
            K,
            size=self.region.num_missiles,
            replace=replace,
            p=probs,
        )
        selected = candidates[idx]
        return selected

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _sample_candidates(self) -> np.ndarray:
        """Sample candidate launch positions in the specified 3D box.

        x in [0, 20] km, y in [-10, 10] km, z in [-10, 10] km.

        Returns:
            positions: np.ndarray of shape (K, 3)
        """
        k = self.region.candidate_launch_count
        x = self.rng.uniform(0.0, 20.0, size=(k,))
        y = self.rng.uniform(-10.0, 10.0, size=(k,))
        z = self.rng.uniform(0, 10.0, size=(k,))
        return np.stack([x, y, z], axis=1)

    def _sample_blue_headings(self, blue_initial_pos: np.ndarray) -> np.ndarray:
        """Construct a set of candidate blue escape headings on the unit sphere.

        We use a small, fixed set of canonical directions (±x, ±y, ±z) plus
        the radial direction from the origin through the blue position.

        Returns:
            headings: (B, 3) unit vectors
        """
        base_dirs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
        ]

        # Radial direction from origin through blue position (outward).
        bp = np.asarray(blue_initial_pos, dtype=float)
        norm_bp = np.linalg.norm(bp)
        if norm_bp > 1e-6:
            radial = bp / norm_bp
        else:
            radial = np.array([1.0, 0.0, 0.0])
        base_dirs.append(radial)

        # Optionally add inward radial direction.
        base_dirs.append(-radial)

        # Normalize and assemble.
        dirs = []
        for d in base_dirs:
            n = np.linalg.norm(d)
            if n < 1e-6:
                continue
            dirs.append(d / n)

        # Ensure we have at least one heading.
        if not dirs:
            dirs = [np.array([1.0, 0.0, 0.0])]

        B = self.region.num_blue_strategies
        dirs = np.stack(dirs, axis=0)  # (D, 3)
        D = dirs.shape[0]

        if B <= D:
            return dirs[:B]
        else:
            # If more headings requested, sample with replacement.
            idx = self.rng.integers(0, D, size=B)
            return dirs[idx]

    def _build_payoff_matrix(self, candidates: np.ndarray, blue_initial_pos: np.ndarray) -> np.ndarray:
        """Construct payoff matrix A for the zero-sum game.

        A[i, j] = distance between candidate i and the predicted blue position
        after it moves along heading j for a fixed escape distance.

        Returns:
            A: np.ndarray of shape (K, B)
        """
        headings = self._sample_blue_headings(blue_initial_pos)  # (B, 3)
        B = headings.shape[0]
        K = candidates.shape[0]

        # Predicted blue positions for each heading.
        disp = self.region.blue_escape_distance  # km
        blue_future = blue_initial_pos[None, :] + disp * headings  # (B, 3)

        # Compute distances for all (i, j) pairs.
        A = np.zeros((K, B), dtype=float)
        for i in range(K):
            p = candidates[i][None, :]  # (1, 3)
            diffs = p - blue_future  # (B, 3)
            dists = np.linalg.norm(diffs, axis=1)  # (B,)
            A[i, :] = dists
        return A

    def _fictitious_play_minimax(self, A: np.ndarray) -> np.ndarray:
        """Approximate the minimax (Nash) strategy for the row player via fictitious play.

        We consider a zero-sum game where:
            - Row player (red) wants to MINIMIZE A;
            - Column player (blue) wants to MAXIMIZE A.

        Returns:
            red_mixed: np.ndarray of shape (K,) approximating red's equilibrium
                       mixed strategy over rows.
        """
        K, B = A.shape
        iters = max(1, self.region.fictitious_iters)

        # Counts of how many times each pure strategy has been played.
        row_counts = np.zeros(K, dtype=float)
        col_counts = np.zeros(B, dtype=float)

        # Initialize with uniform play.
        row_counts[self.rng.integers(0, K)] += 1.0
        col_counts[self.rng.integers(0, B)] += 1.0

        for t in range(1, iters + 1):
            # Average opponent strategies.
            row_mixed = row_counts / row_counts.sum()
            col_mixed = col_counts / col_counts.sum()

            # Row player best response to current column mixed strategy: minimize expected A.
            row_payoffs = A @ col_mixed  # (K,)
            best_row = int(np.argmin(row_payoffs))
            row_counts[best_row] += 1.0

            # Column player best response to current row mixed strategy: maximize expected A.
            col_payoffs = row_mixed @ A  # (B,)
            best_col = int(np.argmax(col_payoffs))
            col_counts[best_col] += 1.0

        red_mixed = row_counts / row_counts.sum()
        return red_mixed
