# Phase 5 Status: ACMI, End-to-End Acceptance, and Final Documentation

## Scope and delivered capability

Phase 5 adds a reusable, streaming Tacview ACMI 2.2 writer, local XZY-to-geodetic
conversion, a Phase-1 JSONL offline converter and CLI, strict unified checkpoint
resume for training, ACMI tests, and a complete README quick start. ACMI remains
optional and does not participate in simulation state transitions.

The coordinate converter requires a caller-supplied latitude, longitude, and
altitude origin. It maps x to north, z to east, and y to up using the documented
small-area Earth-radius approximation. Heading maps north-zero,
clockwise-positive `psi` into `[0, 360)` and pitch maps `gamma` to degrees. Roll
is explicitly exported as zero because the simulator has no six-DoF roll state.

The writer emits the ACMI header, version 2.2, reference time, monotonic time
frames, deterministic numeric formatting, stable IDs, blue and any number of
missiles, type/name/color metadata, removals, and representable hit and ground
collision events. Files are UTF-8 and streamed rather than accumulated in RAM.

## Files changed

- `src/air_combat_rl/io/coordinate_transform.py`: validated geodetic origin,
  XZY local conversion, heading, and pitch conversions.
- `src/air_combat_rl/io/acmi_writer.py`: context-managed streaming writer and
  JSONL-to-ACMI conversion API.
- `scripts/export_acmi.py`: thin offline conversion CLI with mandatory origin.
- `src/air_combat_rl/training/runner.py` and `scripts/train.py`: strict
  same-algorithm checkpoint restoration and `--resume` wiring.
- `tests/unit/test_phase5_acmi.py`: required ACMI and coordinate contracts.
- `README.md`: install, run, all three training modes, resume, evaluation,
  plotting, ACMI, artifacts, algorithm relationship, and outcome definitions.

## Commands executed and results

1. `python -m pip install -e '.[test]'` — **not completed**. The package-index
   proxy returned HTTP 403 while pip attempted to resolve the already-declared
   `setuptools>=68` build requirement. No dependency declaration was duplicated.
2. `PYTHONPATH=src:. pytest tests/unit/test_phase5_acmi.py -q` — initially found
   a real hexadecimal removal-ID formatting defect (4 passed, 1 failed). The
   defect was corrected. A follow-up format audit corrected the Tacview `T`
   transform to `lon|lat|alt|roll|pitch|yaw`, standardized event records, and
   added finite-value and JSONL fallback coverage; the final rerun passed all
   7 tests.
3. `PYTHONPATH=src:. pytest -q` — collection could not complete because NumPy
   and Matplotlib are not installed in this container; both are already declared
   in `pyproject.toml`.
4. `python -m py_compile src/air_combat_rl/io/acmi_writer.py
   src/air_combat_rl/io/coordinate_transform.py
   src/air_combat_rl/training/runner.py scripts/export_acmi.py scripts/train.py`
   — passed.
5. `PYTHONPATH=src python scripts/export_acmi.py --trajectory <smoke-jsonl>
   --origin-lat-deg 34 --origin-lon-deg 108 --origin-alt-m 0
   --reference-time 2026-01-01T00:00:00Z --output <smoke-acmi>` — passed and
   produced the required header plus correctly transformed blue/missile rows.
6. `git diff --check` — passed.

## End-to-end acceptance status

The requested fixed/random scenario executions, three short training runs,
Projected PPO checkpoint resume, three evaluations, reports, plots, exported
simulation ACMI, and dependency-dependent full suite could not truthfully be
executed because installation was blocked and the active interpreter lacks
NumPy/Matplotlib. Static compilation passed and the pure-Python ACMI suite ran;
the complete dynamic matrix remains pending on a worker able to install the
declared dependencies.

Run the following after dependency installation, using unique output paths:

1. Run `scripts/run_scenario.py` with constant and `random_valid` policies.
2. Run `scripts/train.py` for `ppo_projected`, resume its `latest.pt`, then run
   `ppo_discrete` and `rainbow_dqn` short jobs.
3. Run `scripts/evaluate.py` for each matching config/checkpoint and confirm
   `metrics.json`, `episodes.csv`, and `report.md`.
4. Run `scripts/plot_run.py`, `scripts/plot_evaluation.py`, and
   `scripts/compare_algorithms.py`.
5. Run `scripts/export_acmi.py` with an explicit real origin.
6. Run `PYTHONPATH=src:. pytest -q`.

## Final consistency audit

- `ppo_projected` remains the runtime default and documented primary method;
  Discrete PPO and Rainbow remain independent comparisons.
- Existing Projected PPO tests/contracts retain continuous-sample log
  probabilities while environment action IDs denote projected execution, and
  projection uses the legal action mask.
- All three algorithms continue to use `BlueEscapeEnv`, its reward/termination
  semantics, and the same scenario/action configuration path.
- ACMI and existing trajectory code preserve XZY: y altitude, x-z horizontal,
  and `[x,z,y]` serialization.
- Scenario configuration continues to own physics/policy time steps, and
  powered-launch and terminal-intercept presets remain separate files.
- Timeout remains distinct from success, and checkpoint restore rejects an
  algorithm mismatch. No old weights, data, logs, or results were deleted.

## Known limitations

- The local geodetic approximation is intended for limited geographic extents;
  it is not a global ellipsoidal Earth transform.
- ACMI samples recorded policy-step snapshots, not every physics substep.
- Only hit and ground-collision event kinds are exported as ACMI events; other
  simulator events remain available in the source JSONL.
- Unified checkpoints currently restore model state and global progress, but
  the lightweight NumPy trainers do not persist optimizer/replay contents.
- Full end-to-end runtime acceptance is pending dependency availability.

## Next-stage input conditions

Install with `python -m pip install -e '.[test]'` on a worker with package-index
access, execute the six-step acceptance matrix above, and retain its output
directories. Any next milestone should begin only after the full suite and all
three algorithm evaluations pass; it should receive the three checkpoint paths,
evaluation directories, chosen real-world ACMI origin, and acceptance logs.
