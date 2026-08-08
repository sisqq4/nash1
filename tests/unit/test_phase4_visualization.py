import csv
import json
from pathlib import Path

import matplotlib.image as mpimg
import pytest

from air_combat_rl.visualization import PlotDataError, compare_evaluations, plot_evaluation, plot_run


def _step(t, blue, missiles, reward=1.0, components=None, outcome="running"):
    return {"time_s": t, "reward": reward, "outcome": outcome, "events": [],
            "reward_components": components,
            "world_snapshot": {"blue": {"position_xzy_m": blue},
                               "missiles": [{"id": f"missile_{i}", "position_xzy_m": p} for i, p in enumerate(missiles)]}}


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize("missiles", [[[10, 0, 100]], [[10, 0, 100], [20, 10, 200]]])
def test_trajectory_xzy_single_and_multi_missile_missing_components(tmp_path, missiles):
    rows = [_step(0.1, [1, 2, 300], missiles, components=None),
            _step(0.2, [2, 4, 310], [[p[0]-1, p[1], p[2]] for p in missiles], components={"safe": .5}, outcome="timeout")]
    _write_jsonl(tmp_path / "steps.jsonl", rows)
    paths = plot_run(tmp_path)
    assert len(paths) == 6
    assert all(path.stat().st_size > 0 and mpimg.imread(path).size for path in paths)
    # Explicitly exercise the canonical serialization mapping: index 1 is z, index 2 is y.
    assert rows[0]["world_snapshot"]["blue"]["position_xzy_m"] == [1, 2, 300]


def test_empty_trajectory_is_clear_error(tmp_path):
    (tmp_path / "steps.jsonl").write_text("")
    with pytest.raises(PlotDataError, match="empty"):
        plot_run(tmp_path)


def _evaluation(root, algorithm, projected=False, scenario="same.yaml"):
    root.mkdir()
    manifest={"algorithm":algorithm,"scenarios":[scenario],"platform":"zdj","episodes":1,"seeds":[0],"deterministic":True,"actions":"actions.yaml","action_catalog_version":"v1"}
    (root/"manifest.json").write_text(json.dumps(manifest))
    diag={"projection_distances":[.1,.2],"continuous_actions":[[0,1,-1],[.2,.3,.4]],
          "continuous_commands":[[3,9,-3.14],[4,6,1]],"projected_commands":[[0,9,-1.4],[4.5,6,.78]],
          "projected_action_distribution":{"0":1,"4":1}} if projected else {}
    fields=["outcome","projected_ppo"]
    with (root/"episodes.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerow({"outcome":"timeout","projected_ppo":json.dumps(diag)})
    _write_jsonl(root/"evaluation_steps.jsonl",[{"time_s":.1,"action_id":0},{"time_s":.2,"action_id":4}])
    overall={"survival_rate":1,"escape_completion_rate":0,"hit_rate":0,"crash_rate":0,"timeout_rate":1,"exhausted_rate":0,"reward":{"mean":1},"duration_s":{"mean":2},"min_sampled_distance_m":{"mean":3},"lowest_altitude_y_m":4}
    (root/"metrics.json").write_text(json.dumps({"overall":overall,"groups":[{"keys":{"scenario":scenario},"metrics":overall}]}))


def test_projected_fields_and_discrete_absence(tmp_path):
    projected=tmp_path/"projected"; discrete=tmp_path/"discrete"
    _evaluation(projected,"ppo_projected",True); _evaluation(discrete,"ppo_discrete",False)
    assert len(plot_evaluation(projected)) == 5
    paths=plot_evaluation(discrete)
    assert [path.name for path in paths] == ["outcome_distribution.png"]
    assert all(mpimg.imread(path).size for path in [*paths, *(projected/"plots").glob("*.png")])


def test_algorithm_condition_validation_and_comparison(tmp_path):
    roots=[]
    for algorithm in ("ppo_projected","ppo_discrete","rainbow_dqn"):
        root=tmp_path/algorithm; _evaluation(root,algorithm,algorithm=="ppo_projected"); roots.append(root)
    outputs=compare_evaluations(roots,tmp_path/"comparison")
    assert {p.name for p in outputs} == {"algorithm_comparison.png","scenario_comparison.png"}
    assert all(mpimg.imread(path).size for path in outputs)
    bad=tmp_path/"bad"; _evaluation(bad,"other",False,scenario="different.yaml")
    with pytest.raises(PlotDataError,match="conditions differ"):
        compare_evaluations([roots[0],roots[1],bad],tmp_path/"nope")


def test_algorithm_comparison_requires_the_three_supported_algorithms(tmp_path):
    projected=tmp_path/"projected"; first=tmp_path/"first"; second=tmp_path/"second"
    _evaluation(projected,"ppo_projected",True)
    _evaluation(first,"ppo_discrete")
    _evaluation(second,"ppo_discrete")
    with pytest.raises(PlotDataError,match="exactly once"):
        compare_evaluations([projected,first,second],tmp_path/"duplicate")
    with pytest.raises(PlotDataError,match="exactly three"):
        compare_evaluations([projected,first],tmp_path/"two")
