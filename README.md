# Air Combat RL

This repository is structured as a layered blue-escape research codebase:

1. **Domain simulation kernel** (`air_combat_rl.core`, `air_combat_rl.domain`, `air_combat_rl.simulation`) owns coordinates, units, physical state, 3-DoF dynamics, guidance, propulsion, collision checks, scenarios, events, and snapshots.
2. **Task layer** (`air_combat_rl.tasks.blue_escape`) adapts the simulation kernel into a blue escape RL task with replaceable actions, observations, rewards, masks, and termination semantics.
3. **Algorithm layer** (`air_combat_rl.algorithms`) contains scene-agnostic learning code such as discrete PPO.
4. **Application layer** (`air_combat_rl.training`, `air_combat_rl.evaluation`, `scripts`) wires configs, runners, evaluators, logging, and checkpointing.

Dependency direction is intentionally top-down: dynamics do not import training, PPO, observations, or rewards; PPO only depends on standard policy/environment interfaces; logging reads events and snapshots and never participates in state transition.
