import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.air_combat_rl.io.trajectory_writer import SerializationError, TrajectoryWriter, normalize_json
from src.air_combat_rl.runtime import build_blue_escape_env
from scripts.run_scenario import main


def test_json_normalization_numpy_dataclass_enum_path_error(tmp_path):
    payload = normalize_json({"array": np.asarray([1, 2]), "scalar": np.float64(1.5)})
    assert payload == {"array": [1, 2], "scalar": 1.5}
    with pytest.raises(SerializationError, match=r"\$\.bad"):
        normalize_json({"bad": object()})


def test_writer_writes_and_closes(tmp_path):
    path = tmp_path / "steps.jsonl"
    with TrajectoryWriter(path) as writer:
        writer.write_step({"episode": 0, "value": np.int64(3)})
    assert writer._fh is None
    record = json.loads(path.read_text().splitlines()[0])
    assert record["schema_version"] == TrajectoryWriter.schema_version
    assert record["value"] == 3


def test_runtime_builder_sets_requested_blue_platform():
    env, _ = build_blue_escape_env("configs/scenario/fixed_1v1.yaml", "configs/actions/blue_29.yaml", "yjj", 0, max_policy_steps=1)
    assert env.world.blue.platform == "yjj"
    assert env.reset()[0].shape[0] > 0


def test_constant_action_legality_error(tmp_path):
    with pytest.raises(SystemExit, match="illegal"):
        main([
            "--scenario", "configs/scenario/fixed_1v1.yaml", "--actions", "configs/actions/blue_29.yaml",
            "--platform", "yjj", "--policy", "constant", "--action-id", "2", "--seed", "0",
            "--max-policy-steps", "1", "--output-dir", str(tmp_path / "run"),
        ])


def test_random_valid_seed_reproducible_and_masked(tmp_path):
    args = ["--scenario", "configs/scenario/fixed_1v1.yaml", "--actions", "configs/actions/blue_29.yaml", "--platform", "yjj", "--policy", "random_valid", "--seed", "5", "--max-policy-steps", "3"]
    main([*args, "--output-dir", str(tmp_path / "a")])
    main([*args, "--output-dir", str(tmp_path / "b")])
    lines_a = [json.loads(x) for x in (tmp_path / "a" / "steps.jsonl").read_text().splitlines()]
    lines_b = [json.loads(x) for x in (tmp_path / "b" / "steps.jsonl").read_text().splitlines()]
    assert [r["action_id"] for r in lines_a] == [r["action_id"] for r in lines_b]
    assert all(r["action_mask"][r["action_id"]] for r in lines_a)


def test_cli_help(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_scenario.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0
    assert "--scenario" in proc.stdout


def test_short_episode_cli_smoke_jsonl_xzy_height_and_summary(tmp_path):
    out = tmp_path / "smoke"
    main([
        "--scenario", "configs/scenario/fixed_1v1.yaml", "--actions", "configs/actions/blue_29.yaml",
        "--platform", "zdj", "--policy", "constant", "--action-id", "0", "--seed", "0",
        "--max-policy-steps", "2", "--output-dir", str(out),
    ])
    records = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]
    assert len(records) == 2
    pos = records[0]["world_snapshot"]["blue"]["position_xzy_m"]
    assert len(pos) == 3 and pos[2] > 0.0
    summary = json.loads((out / "episode_summary.json").read_text())
    assert summary["truncated"] is True and summary["policy_steps"] == 2
    assert summary["min_altitude_y_m"] == pytest.approx(pos[2])


def test_output_directory_overwrite_protection(tmp_path):
    out = tmp_path / "exists"; out.mkdir(); (out / "keep.txt").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        main(["--scenario", "configs/scenario/fixed_1v1.yaml", "--actions", "configs/actions/blue_29.yaml", "--platform", "zdj", "--policy", "constant", "--action-id", "0", "--output-dir", str(out)])


def test_reset_has_no_cross_episode_state_leakage():
    env, _ = build_blue_escape_env("configs/scenario/fixed_1v1.yaml", "configs/actions/blue_29.yaml", "zdj", 0, max_policy_steps=1)
    obs0, _ = env.reset()
    env.step(0)
    assert env.world.time_s > 0
    obs1, _ = env.reset()
    assert env.world.time_s == 0.0
    assert env.world.substeps_last_interval == 0
    assert env.world.min_missile_distances == [float("inf")]
    assert np.allclose(obs0, obs1)
