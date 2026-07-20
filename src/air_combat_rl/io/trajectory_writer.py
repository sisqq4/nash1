"""JSONL trajectory writer for single-scenario rollouts."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "trajectory.v1"


class SerializationError(TypeError):
    """Raised when a value cannot be converted to JSON-safe data."""


def normalize_json(value: Any, path: str = "$") -> Any:
    """Convert common scientific Python objects into JSON-safe values.

    Raises SerializationError with the offending field path when conversion fails.
    """
    try:
        import numpy as _np
    except Exception:  # pragma: no cover - numpy is a declared dependency
        _np = None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if _np is not None and isinstance(value, _np.generic):
        return value.item()
    if _np is not None and isinstance(value, _np.ndarray):
        return [normalize_json(item, f"{path}[{i}]") for i, item in enumerate(value.tolist())]
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return normalize_json(asdict(value), path)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise SerializationError(f"cannot serialize non-scalar key at {path}: {key!r}")
            out[str(key)] = normalize_json(item, f"{path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [normalize_json(item, f"{path}[{i}]") for i, item in enumerate(value)]
    raise SerializationError(f"cannot serialize value at {path}: {type(value).__name__}")


class TrajectoryWriter:
    """Streaming UTF-8 JSONL writer with a shared schema version."""

    schema_version = SCHEMA_VERSION

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = None

    def __enter__(self) -> "TrajectoryWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write_step(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("TrajectoryWriter is not open")
        payload = {"schema_version": self.schema_version, **record}
        line = json.dumps(normalize_json(payload), ensure_ascii=False, sort_keys=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
