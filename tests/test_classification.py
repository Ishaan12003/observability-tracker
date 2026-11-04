from __future__ import annotations

from obspipe.llm.backends import FakeBackend
from obspipe.models.classification import AMBIGUITY_MARGIN, ClassificationOutput, TypeScore
from obspipe.models.document import DocType
from obspipe.steps.classification import run_classification
from obspipe.steps.extraction import run_extraction
from obspipe.steps.intake import run_intake


def classify(document, ctx, backend):
    intake = run_intake(document, ctx)
    extraction = run_extraction(intake, ctx, backend)
    return run_classification(intake, extraction, ctx, backend)


def codes(output) -> set[str]:
    return set(output.warning_codes())


def test_margin_and_runner_up_are_derived_not_asked_for():
    output = ClassificationOutput(
        doc_id="d",
        doc_type=DocType.REPORT,
        scores=[
            TypeScore(doc_type=DocType.REPORT, score=0.8),
            TypeScore(doc_type=DocType.CORRESPONDENCE, score=0.74),
            TypeScore(doc_type=DocType.CONTRACT, score=0.1),
            TypeScore(doc_type=DocType.INVOICE, score=0.05),
        ],
    )
    assert output.confidence == 0.8
    assert output.runner_up is DocType.CORRESPONDENCE
    assert output.margin == 0.06
    assert output.ambiguous is True


def test_confident_classification_is_not_marked_ambiguous(ctx, backend, doc_of):
    output = classify(doc_of("invoice_mixed_currency.txt"), ctx, backend)
    assert output.doc_type is DocType.INVOICE
    assert output.margin >= AMBIGUITY_MARGIN
    assert "ambiguous_classification" not in codes(output)


def test_genuinely_ambiguous_document_is_flagged_rather_than_forced(ctx, backend, doc_of):
    output = classify(doc_of("memo_ambiguous.md"), ctx, backend)
    assert {output.doc_type, output.runner_up} == {DocType.REPORT, DocType.CORRESPONDENCE}
    assert output.ambiguous is True
    assert "ambiguous_classification" in codes(output)


def test_flat_scores_fault_produces_an_ambiguity_warning(ctx, doc_of):
    output = classify(doc_of("contract_no_dates.md"), ctx, FakeBackend(faults={"flat_scores"}))
    assert output.margin == 0.0
    assert "ambiguous_classification" in codes(output)


def test_no_type_fits_falls_back_to_unknown(ctx, backend):
    from obspipe.models.document import RawDocument

    document = RawDocument.from_text(
        "The quick brown fox jumped over the lazy dog. " * 6, source_name="nothing.txt"
    )
    output = classify(document, ctx, backend)
    assert output.doc_type is DocType.UNKNOWN
    assert "no_type_fits" in codes(output)


def test_evidence_contradiction_between_steps_is_surfaced(ctx, doc_of):
    """Classified invoice with no amounts means one of the two steps is wrong."""
    output = classify(
        doc_of("invoice_mixed_currency.txt"), ctx, FakeBackend(faults={"drop_amounts"})
    )
    assert output.doc_type is DocType.INVOICE
    assert "classification_evidence_missing" in codes(output)


def test_upstream_degradation_is_recorded_on_the_label(ctx, doc_of):
    output = classify(doc_of("contract_no_dates.md"), ctx, FakeBackend(faults={"hallucinate_date"}))
    assert "classified_on_degraded_entities" in codes(output)
