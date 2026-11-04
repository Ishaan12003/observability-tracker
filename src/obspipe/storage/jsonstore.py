"""Trace files on disk.

One JSON document per run, laid out by date. Human-readable and diffable is a
deliberate requirement: the file is the thing you open when a run went wrong,
before any UI exists.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from obspipe.tracing.payloads import REF_PREFIX
from obspipe.tracing.trace import Trace

DEFAULT_ROOT = Path("traces")


def trace_path(trace: Trace, root: Path | str = DEFAULT_ROOT) -> Path:
    day = trace.started_at.astimezone().strftime("%Y-%m-%d")
    return Path(root) / day / f"{trace.trace_id}.json"


def write_trace(trace: Trace, root: Path | str = DEFAULT_ROOT) -> Path:
    path = trace_path(trace, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(trace.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def read_trace(path: Path | str) -> Trace:
    return Trace.model_validate_json(Path(path).read_text(encoding="utf-8"))


def iter_traces(root: Path | str = DEFAULT_ROOT) -> Iterator[Path]:
    root = Path(root)
    if not root.is_dir():
        return
    yield from sorted(root.glob("*/*.json"))


_VOLATILE_KEYS = frozenset(
    {
        "trace_id",
        "span_id",
        "parent_span_id",
        "call_id",
        "run_id",
        "started_at",
        "ingested_at",
        "duration_ms",
    }
)
"""Fields that differ between two otherwise identical runs.

Ids and clock readings, wherever they appear - including inside a serialized
artifact nested in a span input, which is where ``RawDocument.ingested_at``
hides.
"""

_REF_PLACEHOLDER = "@sha256:<ref>"


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items() if k not in _VOLATILE_KEYS}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, str) and value.startswith(REF_PREFIX):
        # The hash depends on the volatile fields inside the payload, so it
        # moves every run. Dedup itself is asserted elsewhere.
        return _REF_PLACEHOLDER
    return value


def normalize_for_diff(trace: Trace) -> dict:
    """Strip everything that changes between two identical runs.

    What survives is the shape - which spans ran, what status, which warnings,
    what confidence - which is exactly what a regression test should be
    sensitive to. Payload bytes are dropped: they are derivable, and their
    hashes are not stable across runs.
    """
    data = _scrub(trace.model_dump(mode="json"))
    data.pop("result", None)
    data.pop("payloads", None)
    return data


def write_golden(trace: Trace, path: Path | str) -> Path:
    """Write a normalized trace for use as a committed regression fixture."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalize_for_diff(trace), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def parse_day(value: str) -> datetime:  # pragma: no cover - helper for CLI filters
    return datetime.strptime(value, "%Y-%m-%d")
