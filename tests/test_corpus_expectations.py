"""The fixture corpus is the contract for Phase 1.

Each document declares, in its sidecar, the failure it is designed to cause.
These tests assert the pipeline actually produces that failure - a corpus whose
documents quietly stop breaking is worthless as a source of traces.

Assertions run against the deterministic fake backend, so they are exact. Phase
4 will re-run the same corpus against a real model, where an expectation
becomes a scored eval case rather than a hard assertion.
"""

from __future__ import annotations

import pytest

from obspipe.corpus import load_corpus
from obspipe.models.pipeline import StepStatus
from obspipe.pipeline import run_pipeline

CORPUS = load_corpus()
IDS = [entry.document.doc_id for entry in CORPUS]

LABELLED = [entry for entry in CORPUS if entry.expectation is not None]
LABELLED_IDS = [entry.document.doc_id for entry in LABELLED]

UNLABELLED = [entry for entry in CORPUS if entry.expectation is None]
UNLABELLED_IDS = [entry.document.doc_id for entry in UNLABELLED]


def test_corpus_covers_every_designed_failure_mode():
    failures = {e.expectation.expected_failure for e in CORPUS if e.expectation}
    assert failures >= {
        "no_dates",
        "mixed_currency_total",
        "ambiguous_type",
        "ocr_damage",
        "truncation_entity_loss",
        "empty_document",
        "no_total",
        "non_english_clause",
    }


@pytest.mark.parametrize("entry", LABELLED, ids=LABELLED_IDS)
def test_document_produces_its_designed_failure(entry, backend):
    result = run_pipeline(entry.document, backend)
    expectation = entry.expectation
    observed = {warning.code for _, warning in result.all_warnings()}

    missing = set(expectation.expected_warning_codes) - observed
    assert not missing, (
        f"{entry.document.doc_id}: expected warning(s) {sorted(missing)} were not raised; "
        f"observed {sorted(observed)}"
    )

    if expectation.expected_failure == "empty_document":
        assert result.status is StepStatus.FAILED
        return

    assert result.status is StepStatus.DEGRADED, (
        f"{entry.document.doc_id} was designed to break but ran clean"
    )
    assert result.doc_type is expectation.expected_type


@pytest.mark.parametrize("entry", CORPUS, ids=IDS)
def test_every_run_is_fully_traced(entry, backend):
    """Even a run that dies in step 1 records all four steps."""
    result = run_pipeline(entry.document, backend)
    assert len(result.spans) == 4
    assert result.first_degraded_step is not None


@pytest.mark.skipif(not UNLABELLED, reason="no unlabelled documents in the corpus")
@pytest.mark.parametrize("entry", UNLABELLED, ids=UNLABELLED_IDS)
def test_unlabelled_documents_still_run_end_to_end(entry, backend):
    """A document dropped in without a sidecar is unlabelled, not invalid.

    Nothing can be asserted about what it *should* produce, but it must still
    run all four steps and come back fully traced. The designed-failure
    guarantee lives in test_corpus_covers_every_designed_failure_mode, which
    fails if a curated fixture ever loses its sidecar.
    """
    result = run_pipeline(entry.document, backend)
    assert len(result.spans) == 4
    assert result.run_id
