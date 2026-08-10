from src.air_combat_rl.io.progress import ExperimentProgress


def test_progress_reports_win_rate_and_convergence_metrics(capsys):
    progress = ExperimentProgress(10, "training", "step")
    progress.record_outcomes(["success", "hit"])
    progress.update(5, {"policy_loss": 0.125, "value_loss": 0.25})
    progress.close()
    output = capsys.readouterr().err
    assert "5/10 step" in output
    assert "win_rate=50.0%" in output
    assert "policy_loss=0.125" in output
    assert "value_loss=0.25" in output
