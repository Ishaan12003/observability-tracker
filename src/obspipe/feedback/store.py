"""Persistence for the feedback loop.

Two stores, on purpose. The eval dataset is a JSONL file - append-only,
diffable, reviewable in a pull request, and the thing you would commit. Flags,
cached diagnoses and eval-run results live in SQLite next to the trace index,
because those are what the dashboard aggregates over.

Unlike ``index.db``, ``feedback.db`` is *not* derivable from the trace files:
it holds human decisions. It is never rebuilt, only migrated.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Iterator

from obspipe.analysis.evidence import Diagnosis
from obspipe.feedback.models import CaseResult, EvalCase, EvalRun, Flag

DEFAULT_DATASET = Path("evals/dataset.jsonl")

SCHEMA = """
CREATE TABLE IF NOT EXISTS flags (
    flag_id     TEXT PRIMARY KEY,
    trace_id    TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    status      TEXT NOT NULL,
    body        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS flags_trace ON flags (trace_id);

CREATE TABLE IF NOT EXISTS diagnoses (
    trace_id         TEXT PRIMARY KEY,
    doc_id           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    verdict          TEXT NOT NULL,
    root_step        TEXT,
    category         TEXT,
    confidence       REAL,
    method           TEXT,
    computed_ms      REAL,
    body             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id            TEXT PRIMARY KEY,
    started_at        TEXT NOT NULL,
    pipeline_version  TEXT NOT NULL,
    label             TEXT,
    total             INTEGER NOT NULL,
    passed            INTEGER NOT NULL,
    duration_ms       REAL
);

CREATE TABLE IF NOT EXISTS eval_results (
    run_id     TEXT NOT NULL,
    case_id    TEXT NOT NULL,
    passed     INTEGER NOT NULL,
    body       TEXT NOT NULL,
    PRIMARY KEY (run_id, case_id)
);
"""


class FeedbackStore:
    def __init__(self, db_path: Path | str, dataset_path: Path | str = DEFAULT_DATASET) -> None:
        self.db_path = Path(db_path)
        self.dataset_path = Path(dataset_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- flags ---------------------------------------------------------------

    def save_flag(self, flag: Flag) -> Flag:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO flags (flag_id, trace_id, created_at, status, body)
                VALUES (?,?,?,?,?)
                ON CONFLICT(flag_id) DO UPDATE SET status=excluded.status, body=excluded.body
                """,
                (
                    flag.flag_id,
                    flag.trace_id,
                    flag.created_at.isoformat(),
                    flag.status.value,
                    flag.model_dump_json(),
                ),
            )
        return flag

    def get_flag(self, flag_id: str) -> Flag | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT body FROM flags WHERE flag_id = ?", (flag_id,)).fetchone()
        return Flag.model_validate_json(row["body"]) if row else None

    def flags(self, trace_id: str | None = None) -> list[Flag]:
        sql = "SELECT body FROM flags"
        params: tuple = ()
        if trace_id:
            sql += " WHERE trace_id = ?"
            params = (trace_id,)
        sql += " ORDER BY created_at DESC"
        with closing(self._connect()) as conn:
            return [Flag.model_validate_json(r["body"]) for r in conn.execute(sql, params)]

    # -- diagnoses -----------------------------------------------------------

    def save_diagnosis(self, diagnosis: Diagnosis, created_at: datetime) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO diagnoses
                    (trace_id, doc_id, created_at, verdict, root_step, category,
                     confidence, method, computed_ms, body)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    diagnosis.trace_id,
                    diagnosis.doc_id,
                    created_at.isoformat(),
                    diagnosis.verdict,
                    diagnosis.root_cause_step.value if diagnosis.root_cause_step else None,
                    diagnosis.category.value if diagnosis.category else None,
                    diagnosis.confidence,
                    diagnosis.method,
                    diagnosis.computed_ms,
                    diagnosis.model_dump_json(),
                ),
            )

    def get_diagnosis(self, trace_id: str) -> Diagnosis | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT body FROM diagnoses WHERE trace_id = ?", (trace_id,)).fetchone()
        return Diagnosis.model_validate_json(row["body"]) if row else None

    def diagnosis_rows(self) -> list[sqlite3.Row]:
        with closing(self._connect()) as conn:
            return list(
                conn.execute(
                    "SELECT trace_id, doc_id, created_at, verdict, root_step, category, "
                    "confidence, method, computed_ms FROM diagnoses"
                )
            )

    # -- eval dataset ----------------------------------------------------------

    def append_case(self, case: EvalCase) -> EvalCase:
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with self.dataset_path.open("a", encoding="utf-8") as handle:
            handle.write(case.model_dump_json() + "\n")
        return case

    def cases(self) -> list[EvalCase]:
        return list(self._iter_cases())

    def _iter_cases(self) -> Iterator[EvalCase]:
        if not self.dataset_path.exists():
            return
        for line in self.dataset_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield EvalCase.model_validate_json(line)

    def get_case(self, case_id: str) -> EvalCase | None:
        return next((c for c in self._iter_cases() if c.case_id == case_id), None)

    # -- eval runs ---------------------------------------------------------------

    def save_run(self, run: EvalRun) -> EvalRun:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO eval_runs VALUES (?,?,?,?,?,?,?)",
                (
                    run.run_id,
                    run.started_at.isoformat(),
                    run.pipeline_version,
                    run.label,
                    run.total,
                    run.passed,
                    run.duration_ms,
                ),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO eval_results VALUES (?,?,?,?)",
                [(run.run_id, r.case_id, int(r.passed), r.model_dump_json()) for r in run.results],
            )
        return run

    def runs(self) -> list[sqlite3.Row]:
        with closing(self._connect()) as conn:
            return list(conn.execute("SELECT * FROM eval_runs ORDER BY started_at"))

    def run_results(self, run_id: str) -> list[CaseResult]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT body FROM eval_results WHERE run_id = ?", (run_id,))
            return [CaseResult.model_validate_json(r["body"]) for r in rows]

    def latest_results(self) -> dict[str, CaseResult]:
        """Each case's result in the most recent run that included it."""
        latest: dict[str, CaseResult] = {}
        for run in self.runs():
            for result in self.run_results(run["run_id"]):
                latest[result.case_id] = result
        return latest
