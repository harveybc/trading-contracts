from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel


def _normalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python", exclude_none=True)
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("canonical timestamps must be timezone-aware")
        utc_value = value.astimezone(timezone.utc)
        return utc_value.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _normalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not allow NaN or infinity")
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_data(value: Any) -> Any:
    """Return a JSON-compatible, recursively normalized value."""

    return _normalize(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonical_data(value),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def content_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()

