from pathlib import Path

from air_combat_rl.tasks.blue_escape.action_catalog import ActionCatalog

ROOT = Path(__file__).resolve().parents[2]


def test_blue_29_catalog_loads_and_validates():
    catalog = ActionCatalog.from_yaml(str(ROOT / "configs/actions/blue_29.yaml"))
    assert catalog.version == "blue_29/v2"
    assert catalog.command_for(5, "zdj").gamma_s < 0


def test_domain_dynamics_do_not_import_algorithm_or_rewards():
    forbidden = ("air_combat_rl.algorithms", "air_combat_rl.training", "reward", "replay")
    for path in (ROOT / "src/air_combat_rl/domain").rglob("*.py"):
        import_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(("import ", "from "))]
        assert not any(term in line for line in import_lines for term in forbidden), path
