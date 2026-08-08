"""Streaming Tacview ACMI 2.2 export for recorded XZY trajectories."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, TextIO

from air_combat_rl.io.coordinate_transform import (
    GeodeticOrigin,
    heading_deg,
    pitch_deg,
    xzy_to_geodetic,
)


class AcmiFormatError(ValueError):
    """Raised for invalid or non-monotonic trajectory input."""


class AcmiWriter:
    """Context-managed, streaming Tacview text writer.

    Object IDs are assigned once from source IDs: blue is ``1`` and missiles
    start at ``100``. Three-DoF states export no roll value; ``T`` contains
    longitude, latitude, altitude, roll, pitch and yaw in the ACMI transform.
    Roll is explicitly zero because the three-DoF model has no roll state.
    """

    def __init__(self, path: str | Path, *, origin: GeodeticOrigin | None,
                 reference_time: datetime | str | None = None) -> None:
        if origin is None:
            raise ValueError("an explicit geodetic origin is required for ACMI export")
        self.path = Path(path)
        self.origin = origin
        self.reference_time = _reference_time(reference_time)
        self._fh: TextIO | None = None
        self._last_time = -float("inf")
        self._ids: dict[str, int] = {"blue": 1}
        self._next_missile_id = 100
        self._active: set[str] = set()

    def __enter__(self) -> "AcmiWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8", newline="\n")
        self._write("FileType=text/acmi/tacview\nFileVersion=2.2\n")
        self._write(f"0,ReferenceTime={self.reference_time}\n")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def write_snapshot(self, time_s: float, blue: dict[str, Any],
                       missiles: Iterable[dict[str, Any]],
                       events: Iterable[dict[str, Any]] = ()) -> None:
        """Write one policy-time snapshot and associated simulation events."""
        if self._fh is None:
            raise RuntimeError("AcmiWriter is not open")
        time_s = float(time_s)
        if not math.isfinite(time_s):
            raise AcmiFormatError("ACMI time must be finite")
        if time_s < self._last_time:
            raise AcmiFormatError(
                f"ACMI time must be monotonic: {time_s} follows {self._last_time}"
            )
        self._last_time = time_s
        self._write(f"#{time_s:.3f}\n")
        self._write_entity("blue", blue, name="Blue", kind="Air+FixedWing", color="Blue")
        seen = {"blue"}
        for index, missile in enumerate(missiles):
            source_id = str(missile.get("id", f"missile_{index}"))
            if source_id == "blue":
                raise AcmiFormatError("missile source ID 'blue' is reserved")
            seen.add(source_id)
            self._write_entity(source_id, missile, name=source_id,
                               kind="Weapon+Missile", color="Red")
        # Explicit removal is understood by Tacview and preserves stable IDs.
        for source_id in sorted(self._active - seen):
            self._write(f"-{self._object_id(source_id):X}\n")
            self._active.remove(source_id)
        for event in events:
            self._write_event(event)

    def _write_entity(self, source_id: str, state: dict[str, Any], *,
                      name: str, kind: str, color: str) -> None:
        object_id = self._object_id(source_id)
        if not bool(state.get("alive", True)):
            if source_id in self._active:
                self._write(f"-{object_id:X}\n")
                self._active.remove(source_id)
            return
        position = state.get("position_xzy_m")
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise AcmiFormatError(f"{source_id} lacks position_xzy_m [x,z,y]")
        latitude, longitude, altitude = xzy_to_geodetic(*position, self.origin)
        heading = heading_deg(float(state.get("psi_rad", 0.0)))
        pitch = pitch_deg(float(state.get("gamma_rad", 0.0)))
        # ACMI's T property is lon|lat|alt|roll|pitch|yaw. Supplying attitude
        # as ad-hoc Heading/Pitch properties is not interpreted by Tacview.
        fields = [
            f"T={longitude:.8f}|{latitude:.8f}|{altitude:.3f}|0.000|{pitch:.3f}|{heading:.3f}"
        ]
        if source_id not in self._active:
            fields.extend((f"Name={_clean(name)}", f"Type={kind}", f"Color={color}"))
            self._active.add(source_id)
        self._write(f"{object_id:X}," + ",".join(fields) + "\n")

    def _write_event(self, event: dict[str, Any]) -> None:
        kind = str(event.get("kind", "event"))
        if kind not in {"hit", "ground_collision"}:
            return
        entity = str(event.get("entity_id", ""))
        label = "Hit" if kind == "hit" else "Ground collision"
        object_id = self._ids.get(entity)
        object_field = f"{object_id:X}" if object_id is not None else ""
        # Event fields are Type|ObjectId|Longitude|Latitude|Altitude|Radius|Text.
        # Message is a standard ACMI event and retains the simulator event name.
        self._write(f"0,Event=Message|{object_field}|||||{label}\n")

    def _object_id(self, source_id: str) -> int:
        if source_id not in self._ids:
            self._ids[source_id] = self._next_missile_id
            self._next_missile_id += 1
        return self._ids[source_id]

    def _write(self, text: str) -> None:
        assert self._fh is not None
        self._fh.write(text)
        self._fh.flush()


def trajectory_jsonl_to_acmi(trajectory: str | Path, output: str | Path, *,
                             origin: GeodeticOrigin | None,
                             reference_time: datetime | str | None = None) -> Path:
    """Stream a Phase-1 ``steps.jsonl`` file into an ACMI file."""
    if origin is None:
        raise ValueError("an explicit geodetic origin is required for ACMI export")
    trajectory = Path(trajectory)
    with trajectory.open("r", encoding="utf-8") as source, AcmiWriter(
        output, origin=origin, reference_time=reference_time
    ) as writer:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                snapshot = record["world_snapshot"]
                time_s = snapshot["time_s"] if "time_s" in snapshot else record["time_s"]
                writer.write_snapshot(time_s,
                                      snapshot["blue"], snapshot.get("missiles", ()),
                                      record.get("events", ()))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise AcmiFormatError(
                    f"invalid trajectory record at {trajectory}:{line_number}: {exc}"
                ) from exc
    return Path(output)


def _reference_time(value: datetime | str | None) -> str:
    if value is None:
        value = datetime.now(timezone.utc)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _clean(value: str) -> str:
    return value.replace(",", "_").replace("\n", " ").replace("\r", " ")
