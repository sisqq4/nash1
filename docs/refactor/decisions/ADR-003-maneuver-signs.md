# ADR-003 Maneuver Signs

## Status

Accepted for M3.

## Context

M2 fixed the canonical flight state and dynamics signs:

- `gamma` increases for climb.
- `psi` increases for right turn.
- Aircraft commands map through `dgamma/dt = g / V * (nf * cos(gamma_s) - cos(gamma))`.
- Aircraft commands map through `dpsi/dt = g * nf * sin(gamma_s) / (V * cos(gamma))`.

The source maneuver table lists 29 discrete maneuver classes, but its direction signs cannot be copied mechanically because the refactored coordinate system defines left/right and climb/dive through these derivative signs.

## Decision

The M3 blue action catalog keeps the table order as stable action ids `0..28` and chooses `gamma_s` by derivative semantics:

- left-turn actions use `nf * sin(gamma_s) < 0` so `dpsi/dt < 0`;
- right-turn actions use `nf * sin(gamma_s) > 0` so `dpsi/dt > 0`;
- climb actions use `nf * cos(gamma_s) > 1` from level flight so `dgamma/dt > 0`;
- dive actions use `nf * cos(gamma_s) < 1` from level flight so `dgamma/dt < 0`.

Pure level turns use `gamma_s = ±acos(1 / nf)` so the vertical derivative remains approximately level while heading changes. Compound maneuvers use quadrant choices that satisfy both required signs: left climb is negative `gamma_s` with positive cosine, right climb is positive `gamma_s` with positive cosine, left dive is negative `gamma_s` with negative cosine, and right dive is positive `gamma_s` with negative cosine.

## Consequences

The action names and expected effects are validated against actual 3-DOF derivative signs instead of against the source table's raw sign convention. Platform masks remain independent of these sign choices and invalidate actions whose absolute `nx` or `nf` exceeds the platform overload limit.
