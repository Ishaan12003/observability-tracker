"""Trace persistence: JSON for fidelity, SQLite for queries."""

from __future__ import annotations

from pathlib import Path

from obspipe.storage.index import DEFAULT_DB, TraceIndex
from obspipe.storage.jsonstore import (
    DEFAULT_ROOT,
    iter_traces,
    normalize_for_diff,
    read_trace,
    trace_path,
    write_golden,
    write_trace,
)
from obspipe.tracing.trace import Trace

__all__ = [
    "DEFAULT_DB",
    "DEFAULT_ROOT",
    "TraceIndex",
    "TraceStore",
    "iter_traces",
    "normalize_for_diff",
    "read_trace",
    "trace_path",
    "write_golden",
    "write_trace",
]


class TraceStore:
    """Writes the JSON file and the index row together.

    Keeping the two in one call is what stops them drifting apart; the index is
    always derivable from the files, and :meth:`reindex` rebuilds it if it does.
    """

    def __init__(
        self, root: Path | str = DEFAULT_ROOT, db_path: Path | str | None = None
    ) -> None:
        self.root = Path(root)
        self.index = TraceIndex(db_path or self.root / "index.db")

    def save(self, trace: Trace) -> Path:
        path = write_trace(trace, self.root)
        self.index.upsert(trace, path)
        return path

    def load(self, trace_id: str) -> Trace:
        row = self.index.get(trace_id)
        if row and row["path"] and Path(row["path"]).exists():
            return read_trace(row["path"])
        for path in iter_traces(self.root):
            if path.stem == trace_id:
                return read_trace(path)
        raise KeyError(f"no stored trace with id {trace_id}")

    def reindex(self) -> int:
        """Rebuild the index from the JSON files on disk."""
        count = 0
        for path in iter_traces(self.root):
            self.index.upsert(read_trace(path), path)
            count += 1
        return count
