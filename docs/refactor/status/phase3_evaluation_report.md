# Phase 3 Evaluation Report

## Scope
Implemented batch evaluation for `ppo_projected`, `ppo_discrete`, `rainbow_dqn`, `constant`, and `random_valid` over multiple scenarios, seeds, and episodes. This phase intentionally does not implement charts or ACMI.

## Implemented
1. Added `scripts/evaluate.py` CLI for batch evaluation.
2. Added reusable evaluator logic in `air_combat_rl.evaluation.evaluator`.
3. Added unified metrics in `air_combat_rl.evaluation.metrics`.
4. Added Markdown report rendering in `air_combat_rl.evaluation.report`.
5. Added baseline algorithm configs for `constant` and `random_valid`.
6. Added Phase 3 unit tests covering metric formulas, grouped metrics, output files, checkpoint compatibility, deterministic projected PPO projection, seed reproducibility, episode isolation, error conditions, and short evaluation paths for the three learning algorithms.

## Output Files
Each evaluation writes:

- `manifest.json`
- `episodes.csv`
- `metrics.json`
- `report.md`
- `trajectories/episode_*.jsonl`
- `evaluation_steps.jsonl`

## Fair Comparison Recording
`manifest.json` records scenarios, seeds, episode count, platform, action catalog version, deterministic setting, and a fair-comparison condition block.

## Projected PPO Diagnostics
Metrics include projection distance mean/std/max, exact projection rate, projected action distribution, continuous action mean/std, saturation rate, valid candidate action counts, and continuous-to-discrete mapping frequency.

## Known Limitations
- No charts are generated in this phase.
- No ACMI files are generated in this phase.
- Trajectory files contain evaluation step records rather than a full high-rate physics-state export.
- The current environment in this session did not have NumPy/PyYAML installed and network package installation was blocked by a 403 tunnel error, so dynamic pytest execution could not complete here.

## Tests/Checks Run
- `python -m py_compile src/air_combat_rl/evaluation/metrics.py src/air_combat_rl/evaluation/report.py src/air_combat_rl/evaluation/evaluator.py scripts/evaluate.py` passed.
- `PYTHONPATH=src pytest tests/unit/test_phase3_evaluation.py -q` could not run because NumPy is missing in the active interpreter.
- `python -m pip install -e '.[test]'` failed due blocked package index access.

## Next Phase Input Conditions
Before the next phase, run the Phase 3 pytest suite in an environment with the declared dependencies installed:

```bash
python -m pip install -e '.[test]'
PYTHONPATH=src pytest tests/unit/test_phase3_evaluation.py -q
```

The next phase can then add visualization and/or ACMI export using the Phase 3 `episodes.csv`, `metrics.json`, and `evaluation_steps.jsonl` outputs as inputs.


## Follow-up Audit Fixes
A follow-up correctness review found and fixed these issues:

1. Projected PPO evaluation now sends the actor's tanh-squashed action into `ProjectedContinuousActionWrapper.step()`, matching the training path, instead of incorrectly sending the raw pre-tanh action.
2. Checkpoint weights are loaded into every freshly rebuilt per-episode runtime rather than only the first runtime.
3. Episode records now track sampled minimum altitude across policy steps, finite closest-distance values, initial missile count, final alive/locked counts, per-missile alive/locked outcomes, maximum simultaneous alive+locked threats, reward component totals, and threat step references.
4. Projected PPO checkpoint validation now also checks the wrapper action bounds metadata when present.
