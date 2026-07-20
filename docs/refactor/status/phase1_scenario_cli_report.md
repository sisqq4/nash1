# Phase 1 Status: Scenario CLI, JSONL Trajectory, Single-Episode Summary

## Scope
Implemented the reusable single-scenario execution path for BlueEscapeEnv. This phase intentionally does not implement training, batch evaluation, plotting, or ACMI export.

## Delivered
- Shared runtime environment builder for scenario/action/platform/seed assembly.
- `scripts/run_scenario.py` CLI with `constant` and `random_valid` policies.
- Streaming UTF-8 JSONL trajectory writer with JSON normalization.
- `manifest.json`, `steps.jsonl`, and `episode_summary.json` artifacts.
- Reset restoration for the environment's world state to avoid cross-episode leakage.
- README usage documentation.

## Verification
Covered JSON normalization, writer lifecycle, constant action legality, random-valid masking, seed reproducibility, CLI help, smoke execution, JSONL parseability, XZY altitude ordering, overwrite protection, and reset isolation in `tests/unit/test_phase1_trajectory_cli.py`.

## Known limitations
- Minimum distance in summaries is policy-step sampled and is therefore named `min_sampled_distance_m`.
- This phase writes JSON artifacts only; no plots, aggregate reports, training, evaluation suite, or ACMI are included.
