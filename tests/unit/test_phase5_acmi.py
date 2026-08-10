from __future__ import annotations

import json
import math

import pytest

from air_combat_rl.io.acmi_writer import AcmiFormatError, AcmiWriter, trajectory_jsonl_to_acmi
from air_combat_rl.io.coordinate_transform import GeodeticOrigin, heading_deg, xzy_to_geodetic


ORIGIN = GeodeticOrigin(45.0, 10.0, 100.0)


def state(x=0.0, z=0.0, y=0.0, psi=0.0, gamma=0.0, alive=True, **extra):
    return {"position_xzy_m": [x, z, y], "psi_rad": psi,
            "gamma_rad": gamma, "alive": alive, **extra}


def test_coordinate_conversion_x_north_z_east_y_up_and_heading():
    lat, lon, alt = xzy_to_geodetic(1000.0, 1000.0, 50.0, ORIGIN)
    assert lat == pytest.approx(45.0 + math.degrees(1000.0 / 6_371_000.0))
    assert lon == pytest.approx(10.0 + math.degrees(1000.0 / (6_371_000.0 * math.cos(math.radians(45.0)))))
    assert alt == 150.0
    assert heading_deg(0.0) == 0.0
    assert heading_deg(math.pi / 2) == 90.0
    assert heading_deg(-math.pi / 2) == 270.0


def test_origin_is_required(tmp_path):
    with pytest.raises(ValueError, match="explicit geodetic origin"):
        AcmiWriter(tmp_path / "x.acmi", origin=None)


def test_header_stable_ids_multiple_missiles_removal_events_and_time(tmp_path):
    path = tmp_path / "trajectory.acmi"
    with AcmiWriter(path, origin=ORIGIN, reference_time="2026-01-02T03:04:05Z") as writer:
        writer.write_snapshot(0.1, state(100, 200, 300, psi=math.pi / 2), [
            state(id="missile_a"), state(id="missile_b", z=10)
        ])
        writer.write_snapshot(0.2, state(101, 201, 301), [
            state(id="missile_a", alive=False), state(id="missile_b", z=20)
        ], [{"kind": "hit", "entity_id": "missile_a"},
            {"kind": "ground_collision", "entity_id": "blue"}])
    text = path.read_text(encoding="utf-8")
    assert text.startswith("FileType=text/acmi/tacview\nFileVersion=2.1\n")
    assert "0,ReferenceTime=2026-01-02T03:04:05Z" in text
    assert text.index("#0.1") < text.index("#0.2")
    assert text.count("a1,T=") == 2
    assert text.count("b1,T=") == 1
    assert text.count("b2,T=") == 2
    assert "-b1" in text
    assert "|0.0|0.0|90.0" in text
    assert text.count("Name=F16,Color=Blue") == 2
    assert text.count("Name=AIM-120,Color=Red") == 3
    assert "0,Event=Message|b1|||||Hit" in text
    assert "0,Event=Message|a1|||||Ground collision" in text


def test_non_monotonic_time_rejected(tmp_path):
    with AcmiWriter(tmp_path / "x.acmi", origin=ORIGIN) as writer:
        writer.write_snapshot(1.0, state(), [])
        with pytest.raises(AcmiFormatError, match="monotonic"):
            writer.write_snapshot(0.9, state(), [])


def test_jsonl_to_acmi_smoke_single_missile(tmp_path):
    source = tmp_path / "steps.jsonl"
    source.write_text(json.dumps({
        "time_s": 0.1,
        "events": [],
        "world_snapshot": {"time_s": 0.1, "blue": state(y=500),
                           "missiles": [state(id="missile_0", x=1000)]},
    }) + "\n", encoding="utf-8")
    output = tmp_path / "out.acmi"
    assert trajectory_jsonl_to_acmi(source, output, origin=ORIGIN,
                                    reference_time="2026-01-01T00:00:00Z") == output
    text = output.read_text(encoding="utf-8")
    assert "#0.1" in text
    assert "a1,T=10.0|45.0|600.0|0.0|0.0|0.0,Name=F16,Color=Blue" in text
    assert "b1,T=10.0|45.00899321605919|100.0|0.0|0.0|0.0,Name=AIM-120,Color=Red" in text


def test_snapshot_time_does_not_require_redundant_top_level_time(tmp_path):
    source = tmp_path / "steps.jsonl"
    source.write_text(json.dumps({
        "world_snapshot": {"time_s": 0.25, "blue": state(), "missiles": []}
    }) + "\n", encoding="utf-8")
    output = tmp_path / "out.acmi"
    trajectory_jsonl_to_acmi(source, output, origin=ORIGIN)
    assert "#0.25" in output.read_text(encoding="utf-8")


def test_non_finite_values_and_reserved_id_rejected(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        GeodeticOrigin(float("nan"), 10.0)
    with AcmiWriter(tmp_path / "x.acmi", origin=ORIGIN) as writer:
        with pytest.raises(AcmiFormatError, match="finite"):
            writer.write_snapshot(float("nan"), state(), [])
    with AcmiWriter(tmp_path / "y.acmi", origin=ORIGIN) as writer:
        with pytest.raises(AcmiFormatError, match="reserved"):
            writer.write_snapshot(0.0, state(), [state(id="blue")])
