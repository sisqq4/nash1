# 01 Target Architecture

The project is organized as a layered architecture: core foundations, domain physics, simulation orchestration, blue escape task components, scene-agnostic PPO algorithms, and application-layer training/evaluation scripts.

```mermaid
flowchart TD
    A[Training and evaluation entrypoints] --> B[PPO and baselines]
    A --> C[Blue escape task environment]
    B --> C
    C --> D[Simulation world and scheduler]
    C --> E[Observation, action, reward, termination]
    D --> F[3-DoF dynamics and guidance]
    D --> G[Scenarios, entities, events]
    F --> H[Coordinates, units, base types]
    E --> H
    G --> H
```
