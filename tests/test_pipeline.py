from __future__ import annotations

from obspipe.llm.backends import FakeBackend
from obspipe.models.common import StepName
from obspipe.models.document import DocType
from obspipe.models.pipeline import StepStatus
from obspipe.pipeline import STEP_ORDER, run_pipeline
from tests.helpers import text_doc


def test_every_step_is_recorded_in_order(backend, doc_of):
    result = run_pipeline(doc_of("invoice_mixed_currency.txt"), backend)
    assert [span.step for span in result.spans] == list(STEP_ORDER)
    assert result.duration_ms > 0
    assert all(record.duration_ms >= 0 for record in result.spans)


def test_all_four_artifacts_are_produced_on_a_healthy_run(backend, doc_of):
    result = run_pipeline(doc_of("invoice_mixed_currency.txt"), backend)
    assert result.intake and result.extraction and result.classification and result.summary
    assert result.doc_type is DocType.INVOICE


def test_a_failing_step_stops_the_run_and_marks_the_rest_skipped(backend, doc_of):
    result = run_pipeline(doc_of("empty.txt"), backend)
    assert result.status is StepStatus.FAILED
    assert result.failed_step is StepName.INTAKE
    assert [s.status for s in result.spans[1:]] == [StepStatus.SKIPPED] * 3
    assert result.extraction is None and result.summary is None


def test_pipeline_never_raises_for_a_bad_document(backend):
    result = run_pipeline(text_doc(""), backend)
    assert result.status is StepStatus.FAILED
    assert result.record_for(StepName.INTAKE).error_type == "empty_document"


def test_unexpected_step_crash_is_recorded_not_propagated(doc_of):
    from langchain_core.runnables import RunnableLambda

    def boom(_):
        raise ZeroDivisionError("synthetic crash inside the model call")

    class Exploding(FakeBackend):
        def structured(self, schema, include_raw=False):
            return RunnableLambda(boom)

    result = run_pipeline(doc_of("contract_no_dates.md"), Exploding())
    record = result.record_for(StepName.EXTRACTION)
    assert result.failed_step is StepName.EXTRACTION
    assert record.error_type == "extraction_call_failed"
    assert "ZeroDivisionError" in record.error_message


def test_first_degraded_step_points_at_the_root_cause(backend, doc_of):
    """Truncation happens in step 1; the missing dates in step 2 are a consequence."""
    result = run_pipeline(doc_of("contract_long_truncated.md"), backend)
    assert result.first_degraded_step is StepName.INTAKE
    assert "document_truncated" in {w.code for _, w in result.all_warnings()}


def test_warnings_are_attributed_to_the_step_that_raised_them(backend, doc_of):
    result = run_pipeline(doc_of("memo_ambiguous.md"), backend)
    attribution = {code: step for step, warning in result.all_warnings() for code in [warning.code]}
    assert attribution["ambiguous_classification"] is StepName.CLASSIFICATION
    assert attribution["ambiguity_not_disclosed"] is StepName.SUMMARIZATION


def test_run_is_reproducible_with_the_fake_backend(doc_of):
    document = doc_of("invoice_mixed_currency.txt")
    first = run_pipeline(document, FakeBackend())
    second = run_pipeline(document, FakeBackend())
    drop = {"run_id", "trace_id", "started_at", "duration_ms", "spans"}
    assert first.model_dump(exclude=drop) == second.model_dump(exclude=drop)


def test_result_serializes_to_json_round_trip(backend, doc_of):
    from obspipe.models.pipeline import PipelineResult

    result = run_pipeline(doc_of("invoice_mixed_currency.txt"), backend)
    restored = PipelineResult.model_validate_json(result.model_dump_json())
    assert restored.doc_id == result.doc_id
    assert len(restored.spans) == len(result.spans)
    assert restored.summary.type_specific == result.summary.type_specific
