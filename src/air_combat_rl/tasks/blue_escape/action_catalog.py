"""29-action catalog loader/validator for blue escape tasks."""
from __future__ import annotations

from dataclasses import dataclass
import math
from air_combat_rl.domain.commands import ManeuverCommand

SAFE_FALLBACK_ACTION_ID = 0


@dataclass(frozen=True, slots=True)
class BlueAction:
    action_id: int
    name: str
    command: ManeuverCommand
    expected_longitudinal_effect: str
    expected_vertical_effect: str
    expected_lateral_effect: str
    valid_platforms: tuple[str, ...]

    @property
    def id(self) -> int:
        return self.action_id

    @property
    def nx(self) -> float:
        return self.command.nx

    @property
    def nf(self) -> float:
        return self.command.nf

    @property
    def gamma_s(self) -> float:
        return self.command.gamma_s


class ActionCatalog:
    version = "blue_29/v2"

    def __init__(self, actions: list[BlueAction]) -> None:
        if len(actions) != 29:
            raise ValueError(f"blue action catalog must contain 29 actions, got {len(actions)}")
        self._actions = {action.action_id: action for action in actions}
        if set(self._actions) != set(range(29)):
            raise ValueError("blue action ids must be exactly 0..28")
        self._validate_direction_labels()

    @classmethod
    def from_yaml(cls, path: str) -> "ActionCatalog":
        """Load the repository's simple blue_29 YAML without coupling tests to PyYAML."""
        items: list[dict[str, object]] = []
        current: dict[str, object] | None = None
        with open(path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if line.startswith("- action_id:") or line.startswith("- id:"):
                    key, value = line[2:].split(":", 1)
                    current = {key.strip(): value.strip()}
                    items.append(current)
                elif current is not None and ":" in line:
                    key, value = line.split(":", 1)
                    value = value.strip()
                    if value.startswith("[") and value.endswith("]"):
                        current[key] = [part.strip() for part in value[1:-1].split(",") if part.strip()]
                    else:
                        current[key] = value
        actions = []
        for item in items:
            action_id = int(item.get("action_id", item.get("id")))
            actions.append(BlueAction(
                action_id=action_id,
                name=str(item["name"]),
                command=ManeuverCommand(float(item["nx"]), float(item["nf"]), math.radians(float(item["gamma_s_deg"]))),
                expected_longitudinal_effect=str(item["expected_longitudinal_effect"]),
                expected_vertical_effect=str(item["expected_vertical_effect"]),
                expected_lateral_effect=str(item["expected_lateral_effect"]),
                valid_platforms=tuple(item["valid_platforms"]),
            ))
        return cls(actions)

    def action(self, action_id: int) -> BlueAction:
        return self._actions[action_id]

    def command_for(self, action_id: int, platform: str) -> ManeuverCommand:
        mask = self.action_mask(platform)
        selected = action_id if mask[action_id] else SAFE_FALLBACK_ACTION_ID
        return self._actions[selected].command

    def action_mask(self, platform: str) -> tuple[bool, ...]:
        mask = tuple(platform in action.valid_platforms for action in self._actions_by_id())
        if any(mask):
            return mask
        fallback = [False] * 29
        fallback[SAFE_FALLBACK_ACTION_ID] = True
        return tuple(fallback)

    def _actions_by_id(self) -> list[BlueAction]:
        return [self._actions[action_id] for action_id in range(29)]

    def _validate_direction_labels(self) -> None:
        for action in self._actions.values():
            vertical_accel = action.nf * math.cos(action.gamma_s) - 1.0
            lateral_accel = action.nf * math.sin(action.gamma_s)
            if action.expected_longitudinal_effect == "accelerate" and not action.nx > 0:
                raise ValueError(f"action {action.action_id} expects acceleration but nx={action.nx}")
            if action.expected_longitudinal_effect == "decelerate" and not action.nx < 0:
                raise ValueError(f"action {action.action_id} expects deceleration but nx={action.nx}")
            if action.expected_lateral_effect == "left" and not lateral_accel < 0:
                raise ValueError(f"action {action.action_id} expects left turn but lateral={lateral_accel}")
            if action.expected_lateral_effect == "right" and not lateral_accel > 0:
                raise ValueError(f"action {action.action_id} expects right turn but lateral={lateral_accel}")
            if action.expected_vertical_effect == "climb" and not vertical_accel > 0:
                raise ValueError(f"action {action.action_id} expects climb but vertical={vertical_accel}")
            if action.expected_vertical_effect == "dive" and not vertical_accel < 0:
                raise ValueError(f"action {action.action_id} expects dive but vertical={vertical_accel}")
