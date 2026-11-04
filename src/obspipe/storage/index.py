"""SQLite index over stored traces.

The JSON files hold everything; this holds the handful of columns you actually
filter and aggregate on. The spans table is the reason it exists - "which step
breaks most often" is the project's central question, and it should be one
query rather than a directory walk.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable

from obspipe.tracing.models import SpanStatus
from obspipe.tracing.trace import Trace

DEFAULT_DB = Path("traces/index.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id            TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    doc_id              TEXT NOT NULL,
    source_name         TEXT,
    backend             TEXT,
    model               TEXT,
    status              TEXT NOT NULL,
    failed_step         TEXT,
    first_degraded_step TEXT,
    duration_ms         REAL,
    total_tokens        INTEGER,
    warning_codes       TEXT,
    final_score         REAL,
    path                TEXT,
    pipeline_version    TEXT,
    batch               TEXT,
    tags                TEXT
);
CREATE INDEX IF NOT EXISTS traces_created_at ON traces (created_at DESC);
CREATE INDEX IF NOT EXISTS traces_status ON traces (status);
CREATE INDEX IF NOT EXISTS traces_doc ON traces (doc_id);

CREATE TABLE IF NOT EXISTS spans (
    trace_id            TEXT NOT NULL,
    span_id             TEXT NOT NULL,
    name                TEXT NOT NULL,
    kind                TEXT NOT NULL,
    status              TEXT NOT NULL,
    duration_ms         REAL,
    self_confidence     INTEGER,
    computed_confidence REAL,
    suspicion           REAL,
    warning_codes       TEXT,
    total_tokens        INTEGER,
    PRIMARY KEY (trace_id, span_id),
    FOREIGN KEY (trace_id) REFERENCES traces (trace_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS spans_name_status ON spans (name, status);
CREATE INDEX IF NOT EXISTS spans_suspicion ON spans (suspicion DESC);
"""


_ADDED_COLUMNS = (("pipeline_version", "TEXT"), ("batch", "TEXT"), ("tags", "TEXT"))
"""Columns added after Phase 2. An index built before them is upgraded in place."""


def _migrate(conn: sqlite3.Connection) -> None:
    present = {row[1] for row in conn.execute("PRAGMA table_info(traces)")}
    for name, kind in _ADDED_COLUMNS:
        if name not in present:
            conn.execute(f"ALTER TABLE traces ADD COLUMN {name} {kind}")
    conn.commit()


class TraceIndex:
    """Thin wrapper over the SQLite index. Safe to construct per use."""

    def __init__(self, db_path: Path | str = DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            _migrate(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # -- writes ----------------------------------------------------------

    def upsert(self, trace: Trace, path: Path | str | None = None) -> None:
        codes = sorted({w.code for _, w in trace.all_warnings()})
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO traces (trace_id, created_at, doc_id, source_name, backend, model,
                                    status, failed_step, first_degraded_step, duration_ms,
                                    total_tokens, warning_codes, final_score, path,
                                    pipeline_version, batch, tags)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(trace_id) DO UPDATE SET
                    pipeline_version=excluded.pipeline_version,
                    batch=excluded.batch,
                    tags=excluded.tags,
                    status=excluded.status,
                    failed_step=excluded.failed_step,
                    first_degraded_step=excluded.first_degraded_step,
                    duration_ms=excluded.duration_ms,
                    total_tokens=excluded.total_tokens,
                    warning_codes=excluded.warning_codes,
                    path=excluded.path
                """,
                (
                    trace.trace_id,
                    trace.started_at.isoformat(),
                    trace.doc_id,
                    trace.source_name,
                    trace.backend,
                    trace.model,
                    trace.status.value,
                    trace.failed_step.value if trace.failed_step else None,
                    trace.first_degraded_step.value if trace.first_degraded_step else None,
                    trace.duration_ms,
                    trace.total_tokens,
                    json.dumps(codes),
                    trace.final_score,
                    str(path) if path else None,
                    trace.tags.get("pipeline_version"),
                    trace.tags.get("batch"),
                    json.dumps(trace.tags, sort_keys=True),
                ),
            )
            conn.execute("DELETE FROM spans WHERE trace_id = ?", (trace.trace_id,))
            conn.executemany(
                """
                INSERT INTO spans (trace_id, span_id, name, kind, status, duration_ms,
                                   self_confidence, computed_confidence, suspicion,
                                   warning_codes, total_tokens)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        trace.trace_id,
                        span.span_id,
                        span.name,
                        span.kind.value,
                        span.status.value,
                        span.duration_ms,
                        span.self_confidence,
                        span.computed_confidence,
                        span.suspicion,
                        json.dumps(span.warning_codes()),
                        span.total_tokens,
                    )
                    for span in trace.spans
                ],
            )

    def set_score(self, trace_id: str, score: float, notes: str | None = None) -> bool:
        """Record a human verdict. Returns False if the trace is not indexed."""
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "UPDATE traces SET final_score = ? WHERE trace_id = ?", (score, trace_id)
            )
            return cursor.rowcount > 0

    # -- reads -----------------------------------------------------------

    def recent(
        self,
        limit: int = 20,
        status: str | None = None,
        batch: str | None = None,
        doc_id: str | None = None,
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM traces"
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (("status", status), ("batch", batch), ("doc_id", doc_id)):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with closing(self._connect()) as conn:
            return list(conn.execute(sql, params))

    def get(self, trace_id: str) -> sqlite3.Row | None:
        with closing(self._connect()) as conn:
            return conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()

    def failure_counts_by_step(self) -> list[sqlite3.Row]:
        """The project's central question, as one query."""
        with closing(self._connect()) as conn:
            return list(
                conn.execute(
                    """
                    SELECT name AS step,
                           COUNT(*) AS occurrences,
                           SUM(status = 'failed') AS failures,
                           SUM(status = 'degraded') AS degraded,
                           ROUND(AVG(suspicion), 3) AS avg_suspicion,
                           ROUND(AVG(self_confidence), 2) AS avg_self_confidence
                    FROM spans
                    WHERE kind = 'step' AND status != 'ok'
                    GROUP BY name
                    ORDER BY failures DESC, degraded DESC
                    """
                )
            )

    def warning_frequency(self, limit: int = 20) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        with closing(self._connect()) as conn:
            for row in conn.execute("SELECT warning_codes FROM spans WHERE kind = 'step'"):
                for code in json.loads(row["warning_codes"] or "[]"):
                    counts[code] = counts.get(code, 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])[:limit]

    def all_rows(self) -> list[sqlite3.Row]:
        with closing(self._connect()) as conn:
            return list(conn.execute("SELECT * FROM traces ORDER BY created_at"))

    def count(self) -> int:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]

    def resolve_id(self, prefix: str) -> list[str]:
        """Accept an unambiguous trace-id prefix, the way git accepts short SHAs."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT trace_id FROM traces WHERE trace_id LIKE ?", (f"{prefix}%",)
            ).fetchall()
        return [row["trace_id"] for row in rows]
