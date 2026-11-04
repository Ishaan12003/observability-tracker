"""Live Groq tests.

Excluded from the default run (see ``addopts`` in pyproject). Enable with::

    pytest -m live

A real model is non-deterministic, so nothing here asserts exact output. What
is asserted is the contract every step must honour regardless of what the model
says: the run is fully traced, spans are grounded in the source, the summary
schema matches the classified type, and gaps are disclosed rather than filled.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from obspipe.corpus import load_entry
from obspipe.llm.backends import GroqBackend
from obspipe.models.common import StepName
from obspipe.models.document import DocType
from obspipe.models.pipeline import StepStatus
from obspipe.pipeline import STEP_ORDER, run_pipeline

load_dotenv()

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set"),
]

CORPUS = "tests/fixtures/documents"


@pytest.fixture(scope="module")
def groq() -> GroqBackend:
    return GroqBackend()


@pytest.fixture(scope="module")
def invoice_run(groq):
    return run_pipeline(load_entry(f"{CORPUS}/invoice_mixed_currency.txt").document, groq)


@pytest.fixture(scope="module")
def contract_run(groq):
    return run_pipeline(load_entry(f"{CORPUS}/contract_no_dates.md").document, groq)


def codes(result) -> set[str]:
    return {warning.code for _, warning in result.all_warnings()}


def test_a_real_run_completes_and_is_fully_traced(invoice_run):
    assert [record.step for record in invoice_run.spans] == list(STEP_ORDER)
    assert invoice_run.status is not StepStatus.FAILED
    assert invoice_run.model == GroqBackend().model
    assert all(record.duration_ms > 0 for record in invoice_run.spans)


def test_extracted_spans_are_grounded_in_the_source(invoice_run):
    """No hallucination errors on a document whose entities are all plainly present."""
    assert "hallucinated_amount_span" not in codes(invoice_run)
    assert "hallucinated_date_span" not in codes(invoice_run)
    assert "hallucinated_party_span" not in codes(invoice_run)


def test_amounts_are_parsed_across_notations(invoice_run):
    amounts = invoice_run.extraction.entities.amounts
    assert amounts
    assert invoice_run.extraction.entities.currencies() >= {"USD", "EUR"}
    # Whatever the model returned, every span it claimed must have survived parsing
    # or been explicitly flagged - never silently zeroed.
    for amount in amounts:
        assert amount.value is not None or "unparseable_amount" in codes(invoice_run)


def test_classification_returns_a_full_score_vector(invoice_run):
    scores = invoice_run.classification.scores
    assert {score.doc_type for score in scores} == {
        DocType.CONTRACT,
        DocType.INVOICE,
        DocType.REPORT,
        DocType.CORRESPONDENCE,
    }
    assert invoice_run.classification.rationale.strip()
    assert 0.0 <= invoice_run.classification.margin <= 1.0


def test_summary_schema_follows_the_classified_type(invoice_run):
    doc_type = invoice_run.classification.doc_type
    fields = set(invoice_run.summary.type_specific)
    expected = {
        DocType.INVOICE: {"total_stated", "currencies_seen"},
        DocType.CONTRACT: {"effective_date", "governing_law"},
        DocType.REPORT: {"key_findings", "recommendations"},
        DocType.CORRESPONDENCE: {"sender", "recipient"},
        DocType.UNKNOWN: {"subject"},
    }[doc_type]
    assert expected <= fields


def test_missing_dates_are_disclosed_not_invented(contract_run):
    """The contract has no calendar dates; the model must not supply one."""
    assert contract_run.extraction.entities.dates == []
    assert "no_dates_found" in codes(contract_run)
    if contract_run.summary.doc_type is DocType.CONTRACT:
        assert contract_run.summary.type_specific["effective_date"] is None
    assert contract_run.summary.unresolved_questions


def test_empty_document_fails_before_any_paid_call(groq):
    """Three LLM calls on a blank page is a bug, not a trace."""
    result = run_pipeline(load_entry(f"{CORPUS}/empty.txt").document, groq)
    assert result.failed_step is StepName.INTAKE
    assert [record.status for record in result.spans[1:]] == [StepStatus.SKIPPED] * 3
