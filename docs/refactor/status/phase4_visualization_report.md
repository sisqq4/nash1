# Phase 4 Status: Offline Result Visualization

## Scope

Implemented only Phase 4: trajectory, reward, Projected PPO diagnostic, and
fair three-algorithm comparison plots. The plotting layer consumes existing
`steps.jsonl`, `evaluation_steps.jsonl`, `episodes.csv`, and `metrics.json`
artifacts. It does not run training or simulation, and this phase does not
implement ACMI.

## Delivered

1. Reusable `air_combat_rl.visualization` readers and plotting functions.
2. Headless Matplotlib `Agg` rendering with non-empty PNG validation.
3. Single-episode x-z horizontal, x-z-y 3D, y-altitude, per-missile distance,
   step/cumulative reward, and optional reward-component charts.
4. Stable blue-aircraft and per-missile colors plus hit, closest-approach,
   ground-collision, and timeout annotations when recorded.
5. Projected PPO projection-distance timeline/distribution, projected action
   histogram, bounded continuous-output distributions, saturation summary, and
   physical `[nx, nf, gamma_s]` continuous/projected command diagnostic. Discrete algorithms safely
   omit projection-only charts.
6. Aggregate algorithm charts covering survival, escape completion, outcomes,
   reward, duration, minimum sampled distance, lowest y-altitude, and scenario
   performance across the same six headline measures.
7. Manifest-based comparison validation for scenarios, platform, episode count,
   seeds, deterministic mode, actions, and action-catalog version.
8. Thin `plot_run.py`, `plot_evaluation.py`, and `compare_algorithms.py` CLIs.

## Dependency audit

Matplotlib was used by the new plotting code but was not declared, so
`matplotlib>=3.7` was added to project dependencies. NumPy and PyYAML were
already declared and were not duplicated.

## Tests

`tests/unit/test_phase4_visualization.py` covers trajectory parsing, canonical
XZY mapping, single/multiple missiles, optional reward components, Projected PPO
fields, absent discrete projection fields, empty trajectories, non-empty/readable
PNG output, and exact three-algorithm/fair-condition validation.

## Verification performed

- `python -m py_compile src/air_combat_rl/evaluation/evaluator.py
  src/air_combat_rl/visualization/__init__.py
  src/air_combat_rl/visualization/plotting.py scripts/plot_run.py
  scripts/plot_evaluation.py scripts/compare_algorithms.py
  tests/unit/test_phase4_visualization.py` passed.
- `PYTHONPATH=src pytest tests/unit/test_phase4_visualization.py -q` was attempted,
  but collection could not start because Matplotlib is not installed.
- `PYTHONPATH=src:. pytest -q --ignore=tests/unit/test_phase4_visualization.py`
  passed all 70 pre-existing tests.
- `python -m pip install -e '.[test]'` was attempted, but the package-index
  tunnel returned HTTP 403 while resolving the build dependency.

Consequently, dynamic image generation and the Matplotlib-dependent Phase 4
suite were not executed in this environment. Install with
`python -m pip install -e '.[test]'` on a network-enabled worker, then run
`PYTHONPATH=src:. pytest -q`.

## Follow-up correctness audit

The post-implementation audit tightened the comparison contract to require the
three supported algorithms exactly once, expanded per-scenario plots from only
survival rate to all six headline measures, prevented projection-distance lines
from joining unrelated episodes, and added event markers to the 3D trajectory.

## Known limitations

- Older Phase 3 artifacts persist bounded continuous actions and final action IDs,
  but not the complete physical continuous/projected command pair. For those old
  artifacts, the comparison diagnostic therefore presents bounded
  continuous outputs against projected action IDs without inventing physical
  command values.
- Plots use policy-step samples rather than high-rate physics substeps.
- PNG is supported; SVG and HTML reports are not implemented.

## Next-stage input conditions

Before a later stage, install declared dependencies and run the full test suite.
Any future ACMI stage should consume existing snapshots/events through the I/O
layer and must not alter this offline plotting contract.
