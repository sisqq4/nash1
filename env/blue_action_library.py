from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from . import action_space


STRATEGY_STEP_COUNT = 2


@dataclass(frozen=True)
class BlueStrategy:
    name: str
    actions: np.ndarray


def _repeat_action(action: np.ndarray, steps: int) -> np.ndarray:
    return np.tile(action.reshape(1, -1), (steps, 1))


def build_escape_strategies(step_count: int = STRATEGY_STEP_COUNT) -> List[BlueStrategy]:
    """Build defensive/escape strategies using the blue_plane_action ordering."""
    primitives = action_space.get_simple()
    if primitives.shape[0] < 11:
        raise ValueError("Expected at least 11 primitive actions in action_space.get_simple().")

    steady = _repeat_action(primitives[0], step_count)
    accelerate = _repeat_action(primitives[1], step_count)
    decelerate = _repeat_action(primitives[2], step_count)
    climb = _repeat_action(primitives[3], step_count)
    dive = _repeat_action(primitives[4], step_count)
    left_turn = _repeat_action(primitives[9], step_count)
    right_turn = _repeat_action(primitives[10], step_count)

    half = max(1, step_count // 2)
    s_turn = np.vstack(
        [
            _repeat_action(primitives[9], half),
            _repeat_action(primitives[10], step_count - half),
        ]
    )

    return [
        BlueStrategy(name="steady", actions=steady),
        BlueStrategy(name="accelerate", actions=accelerate),
        BlueStrategy(name="decelerate", actions=decelerate),
        BlueStrategy(name="climb", actions=climb),
        BlueStrategy(name="dive", actions=dive),
        BlueStrategy(name="left_turn", actions=left_turn),
        BlueStrategy(name="right_turn", actions=right_turn),
        BlueStrategy(name="s_turn", actions=s_turn),
    ]