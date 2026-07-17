"""29-action catalog loader/validator for blue escape tasks."""
from __future__ import annotations

from dataclasses import dataclass
import math
from air_combat_rl.domain.commands import ManeuverCommand

@dataclass(frozen=True, slots=True)
class BlueAction:
    id: int
    name: str
    command: ManeuverCommand
    expected_turn: str
    expected_vertical_motion: str
    allowed_platforms: tuple[str, ...]

    @property
    def ny(self) -> float:
        return self.command.ny

    @property
    def nz(self) -> float:
        return self.command.nz

class ActionCatalog:
    version = "blue_29/v1"

    def __init__(self, actions: list[BlueAction]) -> None:
        if len(actions) != 29:
            raise ValueError(f"blue action catalog must contain 29 actions, got {len(actions)}")
        self._actions = {action.id: action for action in actions}
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
                if line.startswith("- id:"):
                    current = {"id": line.split(":", 1)[1].strip()}
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
            actions.append(BlueAction(
                id=int(item["id"]), name=str(item["name"]),
                command=ManeuverCommand(float(item["nx"]), float(item["nf"]), math.radians(float(item["gamma_s_deg"]))),
                expected_turn=str(item["expected_turn"]),
                expected_vertical_motion=str(item["expected_vertical_motion"]),
                allowed_platforms=tuple(item["allowed_platforms"]),
            ))
        return cls(actions)

    def command_for(self, action_id: int, platform: str) -> ManeuverCommand:
        action = self._actions[action_id]
        if platform not in action.allowed_platforms:
            raise ValueError(f"action {action_id} is not allowed for platform {platform}")
        return action.command

    def _validate_direction_labels(self) -> None:
        for action in self._actions.values():
            if action.expected_turn == "left" and not action.ny < 0:
                raise ValueError(f"action {action.id} expects left turn but ny={action.ny}")
            if action.expected_turn == "right" and not action.ny > 0:
                raise ValueError(f"action {action.id} expects right turn but ny={action.ny}")
            if action.expected_vertical_motion == "climb" and not action.nz > 0:
                raise ValueError(f"action {action.id} expects climb but nz={action.nz}")
            if action.expected_vertical_motion == "dive" and not action.nz < 0:
                raise ValueError(f"action {action.id} expects dive but nz={action.nz}")
