"""Small helpers for live training and evaluation progress reporting."""
from __future__ import annotations

from collections import deque
import sys
import time
from typing import Mapping


class ExperimentProgress:
    """A terminal bar with a rolling outcome rate and live optimization metrics."""

    def __init__(self, total: int, description: str, unit: str, window: int = 100) -> None:
        self.total = max(0, int(total))
        self.description = description
        self.unit = unit
        self._outcomes: deque[str] = deque(maxlen=window)
        self._position = 0
        self._started = time.monotonic()

    def record_outcomes(self, outcomes) -> None:
        self._outcomes.extend(str(outcome) for outcome in outcomes if outcome)

    def update(self, position: int, metrics: Mapping[str, object] | None = None) -> None:
        position = min(int(position), self.total)
        self._position = max(self._position, position)
        postfix: dict[str, str] = {}
        if self._outcomes:
            wins = sum(outcome == "success" for outcome in self._outcomes)
            postfix["win_rate"] = f"{wins / len(self._outcomes):.1%}"
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl",
                    "gradient_norm", "samples_per_second"):
            value = (metrics or {}).get(key)
            if isinstance(value, (int, float)):
                postfix[key] = f"{value:.4g}"
        fraction = self._position / self.total if self.total else 1.0
        filled = min(30, int(30 * fraction))
        elapsed = max(time.monotonic() - self._started, 1.0e-9)
        rate = self._position / elapsed
        details = " ".join(f"{key}={value}" for key, value in postfix.items())
        line = (f"\r{self.description}: [{'#' * filled}{'-' * (30 - filled)}] "
                f"{self._position}/{self.total} {self.unit} ({fraction:.1%}) "
                f"{rate:.1f} {self.unit}/s {details}")
        sys.stderr.write(line.rstrip())
        sys.stderr.flush()

    def close(self) -> None:
        if self._position or self.total:
            sys.stderr.write("\n")
            sys.stderr.flush()
