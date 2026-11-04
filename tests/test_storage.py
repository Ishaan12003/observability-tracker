"""Trace persistence: JSON files, SQLite index, and golden-trace regression."""

from __future__ import annotations

import json
import sqlite3

import pytest

from obspipe.corpus import load_corpus
from obspipe.models.common import StepName
from obspipe.pipeline import run_traced
from obspipe.storage import TraceIndex, TraceStore, normalize_for_diff, read_trace, write_trace
from obspipe.tracing.models import SpanStatus


@pytest.fixture
def store(tmp_path) -> TraceStore:
    return TraceStore(tmp_path / "traces")


@pytest.fixture
def traced_corpus(store, backend):
    traces = [run_traced(entry.document, backend) for entry in load_corpus()]
    for trace in traces:
        store.save(trace)
    return traces


# -- JSON round trip -----------------------------------------------------


def test_a_trace_survives_a_write_and_read(tmp_path, backend, doc_of):
    trace = run_traced(doc_of("invoice_mixed_currency.txt"), backend)
    path = write_trace(trace, tmp_path)
    restored = read_trace(path)

    assert restored.trace_id == trace.trace_id
    assert restored.status is trace.status
    assert len(restored.spans) == len(trace.spans)
    assert restored.result.summary.headline == trace.result.summary.headline
    # Payload references must still resolve after a round trip.
    span = restored.span_for(StepName.EXTRACTION)
    assert restored.resolve(span.inputs)["intake"]["doc_id"] == trace.doc_id


def test_traces_are_laid_out_by_date(tmp_path, backend, doc_of):
    trace = run_traced(doc_of("contract_no_dates.md"), backend)
    path = write_trace(trace, tmp_path)
    assert path.parent.name == trace.started_at.astimezone().strftime("%Y-%m-%d")
    assert path.stem == trace.trace_id


def test_stored_json_is_human_readable(tmp_path, backend, doc_of):
    """The file is what you open when a run went wrong and no UI exists yet."""
    trace = run_traced(doc_of("invoice_mixed_currency.txt"), backend)
    text = write_trace(trace, tmp_path).read_text()
    assert text.startswith("{\n"), "indented, not minified"
    assert "mixed_currencies" in text, "warnings readable without resolving anything"


# -- SQLite index --------------------------------------------------------


def test_index_records_one_row_per_trace_and_span(store, traced_corpus):
    assert store.index.count() == len(traced_corpus)
    with sqlite3.connect(store.index.db_path) as conn:
        spans = conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
    assert spans == sum(len(t.spans) for t in traced_corpus)


def test_recent_returns_newest_first_and_filters_by_status(store, traced_corpus):
    rows = store.index.recent(limit=50)
    assert [r["created_at"] for r in rows] == sorted(
        (r["created_at"] for r in rows), reverse=True
    )
    failed = store.index.recent(limit=50, status="failed")
    assert failed and all(r["status"] == "failed" for r in failed)
    assert {r["doc_id"] for r in failed} == {"empty"}


def test_failure_counts_answer_which_step_breaks_most(store, traced_corpus):
    rows = {r["step"]: r for r in store.index.failure_counts_by_step()}
    assert "extraction" in rows
    assert rows["intake"]["failures"] == 1, "only the empty document fails outright"
    assert all(r["occurrences"] >= r["failures"] for r in rows.values())


def test_warning_frequency_aggregates_across_traces(store, traced_corpus):
    frequency = dict(store.index.warning_frequency(limit=50))
    assert frequency["no_dates_found"] >= 3
    assert all(count >= 1 for count in frequency.values())


def test_upsert_is_idempotent(store, backend, doc_of):
    trace = run_traced(doc_of("memo_ambiguous.md"), backend)
    store.save(trace)
    store.save(trace)
    assert store.index.count() == 1
    with sqlite3.connect(store.index.db_path) as conn:
        spans = conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
    assert spans == len(trace.spans), "re-saving must not duplicate spans"


def test_a_score_can_be_attached_to_a_stored_trace(store, backend, doc_of):
    """Phase 4's feedback loop writes here."""
    trace = run_traced(doc_of("memo_ambiguous.md"), backend)
    store.save(trace)
    assert store.index.set_score(trace.trace_id, 2.0)
    assert store.index.get(trace.trace_id)["final_score"] == 2.0
    assert not store.index.set_score("nope", 1.0)


def test_trace_ids_resolve_from_an_unambiguous_prefix(store, backend, doc_of):
    trace = run_traced(doc_of("memo_ambiguous.md"), backend)
    store.save(trace)
    assert store.index.resolve_id(trace.trace_id[:8]) == [trace.trace_id]
    assert store.index.resolve_id("zzzz") == []


def test_the_index_is_rebuildable_from_the_files(store, traced_corpus):
    """The JSON files are the source of truth; the index is derived."""
    store.index.db_path.unlink()
    rebuilt = TraceStore(store.root, db_path=store.index.db_path)
    assert rebuilt.reindex() == len(traced_corpus)
    assert rebuilt.index.count() == len(traced_corpus)


def test_load_raises_for_an_unknown_trace(store):
    with pytest.raises(KeyError):
        store.load("does-not-exist")


# -- golden traces -------------------------------------------------------


def test_normalizing_strips_everything_that_changes_between_runs(backend, doc_of):
    """Two identical runs must normalize to the same thing, or golden files churn."""
    document = doc_of("invoice_mixed_currency.txt")
    first = normalize_for_diff(run_traced(document, backend))
    second = normalize_for_diff(run_traced(document, backend))
    assert first == second
    assert "trace_id" not in first
    assert all("span_id" not in span for span in first["spans"])


def test_normalized_trace_still_carries_the_evidence(backend, doc_of):
    normalized = normalize_for_diff(run_traced(doc_of("contract_long_truncated.md"), backend))
    codes = {w["code"] for span in normalized["spans"] for w in span["warnings"]}
    assert "document_truncated" in codes
    statuses = {span["name"]: span["status"] for span in normalized["spans"]}
    assert statuses["intake"] == SpanStatus.DEGRADED.value


def test_golden_traces_match_the_committed_fixtures(backend, doc_of, golden_dir):
    """Regression guard on trace shape, without committing run output.

    Regenerate deliberately with OBSPIPE_UPDATE_GOLDEN=1 when a change to the
    trace format is intended.
    """
    import os

    from obspipe.storage import write_golden

    for name in ("invoice_mixed_currency.txt", "contract_long_truncated.md", "empty.txt"):
        trace = run_traced(doc_of(name), backend)
        path = golden_dir / f"{trace.doc_id}.json"
        if os.environ.get("OBSPIPE_UPDATE_GOLDEN"):
            write_golden(trace, path)
            continue
        assert path.exists(), f"missing golden file {path}; run with OBSPIPE_UPDATE_GOLDEN=1"
        expected = json.loads(path.read_text())
        assert normalize_for_diff(trace) == expected, f"trace shape changed for {trace.doc_id}"
