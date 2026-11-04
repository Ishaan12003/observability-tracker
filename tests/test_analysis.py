"""Phase 3: backward analysis, ground-truth comparison and the evidence chain."""

from __future__ import annotations

import pytest

from obspipe.analysis import FailureCategory, GoldenRecord, compare, diagnose, project, step_diffs
from obspipe.analysis.evidence import StepHealth
from obspipe.llm.backends import FakeBackend
from obspipe.models.common import StepName
from obspipe.pipeline import get_config, run_traced
from tests.helpers import text_doc

CONTRACT = """# SERVICES AGREEMENT

This Services Agreement is made between Acme Robotics Inc (the "Client") and Beta Labs LLC (the "Contractor").

The term of this Agreement is 12 months, commencing on the Effective Date of March 3, 2025.
The Client shall pay the Contractor a fixed monthly fee of $4,500.00 within thirty days.
Either party may terminate this Agreement for material breach. This Agreement shall be governed
by the laws of the State of Delaware. Each party shall keep confidential all information.

IN WITNESS WHEREOF the parties have executed this Agreement on February 20, 2025.

Client: Acme Robotics Inc
Contractor: Beta Labs LLC
"""

GOLDEN = GoldenRecord(
    doc_id="contract",
    expected={
        StepName.INTAKE: {"truncated": False},
        StepName.EXTRACTION: {
            "parties": ["Acme Robotics Inc", "Beta Labs LLC"],
            "dates": ["2025-02-20", "2025-03-03"],
            "amounts": ["4500.00 USD"],
        },
        StepName.CLASSIFICATION: {"doc_type": "contract"},
        StepName.SUMMARIZATION: {"doc_type": "contract", "effective_date": "2025-03-03"},
    },
)


def _run(faults=(), config=None, content=CONTRACT, name="contract.md"):
    doc = text_doc(content, name)
    return run_traced(doc, FakeBackend(set(faults)), config=config)


# -- compare -------------------------------------------------------------------


def test_compare_reports_missing_unexpected_and_mismatched_fields():
    found = compare(
        {"parties": ["A", "B"], "doc_type": "invoice", "total": "10.00 USD"},
        {"parties": ["a", "C"], "doc_type": "report", "total": "10.00 USD"},
    )
    kinds = {(d.path, d.kind) for d in found}
    assert kinds == {("parties", "missing"), ("parties", "unexpected"), ("doc_type", "mismatch")}


def test_compare_checks_only_the_keys_ground_truth_names():
    assert compare({"doc_type": "contract"}, {"doc_type": "contract", "margin": 0.01}) == []


def test_compare_accepts_any_of_for_defensible_alternatives():
    expected = {"doc_type": {"any_of": ["report", "correspondence"]}}
    assert compare(expected, {"doc_type": "correspondence"}) == []
    assert compare(expected, {"doc_type": "invoice"})[0].kind == "mismatch"


def test_projection_normalizes_money_and_dates():
    trace = _run()
    output = trace.resolve(trace.span_for(StepName.EXTRACTION).output)
    view = project(StepName.EXTRACTION, output)
    assert view["amounts"] == ["4500.00 USD"]
    assert view["dates"] == ["2025-02-20", "2025-03-03"]


# -- diagnosis -------------------------------------------------------------------


def test_a_clean_run_is_healthy_and_matches_ground_truth():
    diagnosis = diagnose(_run(), GOLDEN)
    assert diagnosis.verdict == "healthy"
    assert all(s.health is StepHealth.HEALTHY for s in diagnosis.steps)


def test_hallucinated_party_is_traced_from_extraction_into_the_summary():
    diagnosis = diagnose(_run(["hallucinate_party"]), GOLDEN)
    assert diagnosis.root_cause_step is StepName.EXTRACTION
    assert diagnosis.category is FailureCategory.HALLUCINATION
    root, *propagation = diagnosis.chain
    assert "hallucinated the party 'John Smith'" in root.finding
    assert "does not appear in the source document" in root.finding
    assert root.produced["occurrences_in_source"] == {"John Smith": 0}
    summary = next(link for link in propagation if link.step is StepName.SUMMARIZATION)
    assert "John Smith" in summary.traced_values
    assert "This propagated to Step 4 (Summarization)" in diagnosis.narrative


def test_hallucinated_date_reaches_the_effective_date_field():
    diagnosis = diagnose(_run(["hallucinate_date"]), GOLDEN)
    assert diagnosis.root_cause_step is StepName.EXTRACTION
    assert "type_specific.effective_date" in diagnosis.narrative


def test_a_confident_misclassification_is_invisible_without_ground_truth():
    trace = _run(["misclassify"])
    assert diagnose(trace).verdict == "healthy"
    with_truth = diagnose(trace, GOLDEN)
    assert with_truth.root_cause_step is StepName.CLASSIFICATION
    assert with_truth.category is FailureCategory.MISCLASSIFICATION
    assert with_truth.method == "ground_truth"
    assert "confidently wrong" in with_truth.narrative


def test_truncation_is_blamed_on_intake_not_on_the_extractor(doc_of, backend):
    diagnosis = diagnose(run_traced(doc_of("contract_long_truncated.md"), backend))
    assert diagnosis.root_cause_step is StepName.INTAKE
    assert diagnosis.category is FailureCategory.INPUT_TRUNCATION
    assert "lie past the cut" in diagnosis.chain[0].finding
    assert StepName.EXTRACTION in diagnosis.affected_steps


def test_a_failed_call_names_the_error_and_the_skipped_steps():
    diagnosis = diagnose(_run(["call_timeout"]))
    assert diagnosis.category is FailureCategory.LLM_CALL_FAILURE
    assert "TimeoutError" in diagnosis.chain[0].finding
    skipped = [s for s in diagnosis.steps if s.health is StepHealth.SKIPPED]
    assert [s.step for s in skipped] == [StepName.SUMMARIZATION]


def test_advisory_warnings_never_make_a_step_the_root_cause(doc_of, backend):
    """A contract with no dates makes no_dates_found true - and correct."""
    diagnosis = diagnose(run_traced(doc_of("contract_no_dates.md"), backend))
    assert diagnosis.verdict == "healthy"


def test_step_diffs_prefer_a_human_correction_over_ground_truth():
    trace = _run(["misclassify"])
    rows = step_diffs(trace, GOLDEN, {StepName.CLASSIFICATION: {"doc_type": "report"}})
    classification = next(r for r in rows if r["step"] == "classification")
    assert classification["expected_source"] == "correction"
    assert classification["divergences"] == []


# -- guards ---------------------------------------------------------------------


def test_grounding_guard_stops_the_hallucinated_party_at_step_two():
    trace = _run(["hallucinate_party"], config=get_config("v3"))
    extraction = trace.span_for(StepName.EXTRACTION)
    assert "ungrounded_span_dropped" in extraction.warning_codes()
    assert diagnose(trace, GOLDEN).verdict == "healthy"


INVOICE = """INVOICE

Invoice No: INV-1
Invoice Date: March 3, 2025
From: Acme Robotics Inc
Bill To: Beta Labs LLC

Consulting services (10 hours) ......... $1,000.00
Subtotal ............................... $1,000.00
Sales tax at 8 percent ................... $80.00
Total due .............................. $1,080.00
"""


@pytest.mark.parametrize("version, expected_total", [("v1", "$9,999.00"), ("v3", "$1,080.00")])
def test_summary_guard_replaces_an_invented_total_with_the_grounded_one(version, expected_total):
    trace = _run(["invent_total"], config=get_config(version), content=INVOICE, name="invoice.txt")
    assert trace.result.summary.type_specific["total_stated"] == expected_total


def test_summary_guard_withholds_a_figure_it_cannot_repair():
    no_total = INVOICE.replace("Total due .............................. $1,080.00\n", "")
    trace = _run(["invent_total"], config=get_config("v3"), content=no_total, name="invoice.txt")
    summary = trace.result.summary
    assert summary.type_specific["total_stated"] is None
    assert any("$9,999.00" in q for q in summary.unresolved_questions)


def test_swap_date_format_only_touches_ambiguous_numeric_dates():
    doc = INVOICE.replace("March 3, 2025", "03/04/2025")
    trace = _run(["swap_date_format"], content=doc, name="invoice.txt")
    assert "date_normalization_mismatch" in trace.span_for(StepName.EXTRACTION).warning_codes()
    clean = _run(["swap_date_format"], content=INVOICE, name="invoice.txt")
    assert "date_normalization_mismatch" not in clean.span_for(StepName.EXTRACTION).warning_codes()


def test_unknown_pipeline_version_is_rejected():
    with pytest.raises(KeyError):
        get_config("v99")


def test_traces_record_their_pipeline_version_and_faults():
    trace = _run(["hallucinate_party"], config=get_config("v2"))
    assert trace.tags["pipeline_version"] == "v2"
    assert trace.tags["faults"] == ["hallucinate_party"]
    assert trace.tags["max_chars"] == 48_000


def test_existing_corpus_entry_loads_without_simulated_faults(corpus):
    assert all(entry.expectation is None or entry.expectation.simulated_faults == [] for entry in corpus)
