"""Action hold state for PPO-rate choices applied across physics substeps."""
from __future__ import annotations

from dataclasses import dataclass

from src.air_combat_rl.core.timebase import SimulationClock
from src.air_combat_rl.domain.commands import ManeuverCommand
from src.air_combat_rl.tasks.blue_escape.action_catalog import ActionCatalog, SAFE_FALLBACK_ACTION_ID


@dataclass(slots=True)
class HeldAction:
    action_id: int = SAFE_FALLBACK_ACTION_ID
    command: ManeuverCommand | None = None
    remaining_substeps: int = 0

    def select(self, action_id: int, platform: str, catalog: ActionCatalog, clock: SimulationClock) -> ManeuverCommand:
        self.action_id = action_id if catalog.action_mask(platform)[action_id] else SAFE_FALLBACK_ACTION_ID
        self.command = catalog.command_for(self.action_id, platform)
        self.remaining_substeps = clock.substeps_per_policy_step
        return self.command

    def consume_substep(self) -> ManeuverCommand:
        if self.command is None or self.remaining_substeps <= 0:
            raise RuntimeError("no held action is available for this physics substep")
        self.remaining_substeps -= 1
        return self.command
