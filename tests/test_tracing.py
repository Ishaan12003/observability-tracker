"""Tracing layer: spans, payloads, confidence, suspicion."""

from __future__ import annotations

import pytest

from obspipe.llm.backends import FakeBackend
from obspipe.models.common import StepName
from obspipe.pipeline import STEP_ORDER, run_traced
from obspipe.tracing.models import Span, SpanKind, SpanStatus
from obspipe.tracing.payloads import INLINE_LIMIT, PayloadStore, is_ref
from obspipe.tracing.suspicion import score_span
from obspipe.tracing.trace import Trace


@pytest.fixture
def invoice_trace(backend, doc_of):
    return run_traced(doc_of("invoice_mixed_currency.txt"), backend)


# -- payload store -------------------------------------------------------


def test_small_payloads_stay_inline_so_a_trace_stays_readable():
    store = PayloadStore()
    assert store.put({"a": 1}) == {"a": 1}
    assert not store.blobs


def test_large_payloads_are_stored_once_and_referenced():
    store = PayloadStore()
    first = store.put({"text": "x" * (INLINE_LIMIT + 100)})
    second = store.put({"text": "x" * (INLINE_LIMIT + 100)})
    assert is_ref(first) and first == second
    assert len(store.blobs) == 1, "identical payloads must deduplicate"
    assert store.get(first)["text"].startswith("xxx")


def test_missing_payload_raises_rather_than_returning_none():
    with pytest.raises(KeyError):
        PayloadStore().get("@sha256:deadbeef")


def test_the_intake_artifact_is_stored_once_for_all_three_consumers(backend, doc_of):
    """Extraction, classification and summarization all read the same object."""
    trace = run_traced(doc_of("contract_long_truncated.md"), backend)
    refs = {
        trace.span_for(step).inputs["intake"]
        for step in (StepName.EXTRACTION, StepName.CLASSIFICATION, StepName.SUMMARIZATION)
    }
    assert len(refs) == 1, "the same artifact must resolve to one payload"
    assert is_ref(next(iter(refs)))


def test_the_document_body_is_not_copied_into_every_prompt(backend, doc_of):
    """Each prompt embeds the document; stored verbatim that is three copies."""
    trace = run_traced(doc_of("contract_long_truncated.md"), backend)
    prompts = [
        trace.payloads.get(span.llm_calls[0].prompt)
        for span in trace.step_spans()
        if span.llm_calls
    ]
    assert len(prompts) == 3
    body = "MASTER SUPPLY AND DISTRIBUTION AGREEMENT"
    stored_verbatim = sum(1 for p in prompts if body in str(p))
    assert stored_verbatim == 0, "prompt should reference the document, not inline it"
    # ...and it still rehydrates to the real prompt.
    call = trace.span_for(StepName.EXTRACTION).llm_calls[0]
    assert body in trace.resolve(call.prompt)[1]["content"]


def test_spans_are_not_written_twice(backend, doc_of):
    """result.spans and trace.spans are the same objects; only one gets stored."""
    import json

    trace = run_traced(doc_of("invoice_mixed_currency.txt"), backend)
    stored = json.loads(trace.model_dump_json())
    assert "spans" not in stored["result"]
    assert len(stored["spans"]) == 5


# -- span structure ------------------------------------------------------


def test_every_step_produces_a_span_in_order(invoice_trace):
    assert [s.step for s in invoice_trace.step_spans()] == list(STEP_ORDER)
    assert all(s.kind is SpanKind.STEP for s in invoice_trace.step_spans())


def test_a_pipeline_span_parents_the_step_spans(invoice_trace):
    root = next(s for s in invoice_trace.spans if s.kind is SpanKind.PIPELINE)
    children = invoice_trace.children_of(root.span_id)
    assert {c.step for c in children} == set(STEP_ORDER)


def test_spans_capture_inputs_and_outputs(invoice_trace):
    span = invoice_trace.span_for(StepName.EXTRACTION)
    assert "intake" in span.inputs
    assert span.output is not None
    resolved = invoice_trace.resolve(span.output)
    assert resolved["doc_id"] == invoice_trace.doc_id


def test_configuration_arguments_are_not_stored_as_span_inputs(invoice_trace):
    """The backend has no useful serialization and ctx is on the trace header."""
    for span in invoice_trace.step_spans():
        assert "backend" not in span.inputs
        assert "ctx" not in span.inputs


def test_llm_calls_are_attached_to_the_step_that_made_them(invoice_trace):
    assert not invoice_trace.span_for(StepName.INTAKE).llm_calls, "intake uses no model"
    for step in (StepName.EXTRACTION, StepName.CLASSIFICATION, StepName.SUMMARIZATION):
        calls = invoice_trace.span_for(step).llm_calls
        assert len(calls) == 1
        assert calls[0].prompt is not None
        assert calls[0].raw_response is not None
        assert calls[0].duration_ms >= 0


def test_a_failed_step_stops_the_trace_and_marks_the_rest_skipped(backend, doc_of):
    trace = run_traced(doc_of("empty.txt"), backend)
    assert trace.status is SpanStatus.FAILED
    assert trace.failed_step is StepName.INTAKE
    later = [s.status for s in trace.step_spans()[1:]]
    assert later == [SpanStatus.SKIPPED] * 3
    assert trace.span_for(StepName.INTAKE).error_type == "whitespace_only_document"


def test_steps_run_untraced_when_no_tracer_is_active(ctx, backend, doc_of):
    """The decorator must not make steps un-callable outside a pipeline run."""
    from obspipe.steps.intake import run_intake

    output = run_intake(doc_of("contract_no_dates.md"), ctx)
    assert output.normalized_text


# -- confidence ----------------------------------------------------------


def test_self_reported_and_computed_confidence_are_kept_apart(invoice_trace):
    span = invoice_trace.span_for(StepName.EXTRACTION)
    assert 1 <= span.self_confidence <= 5
    assert 0.0 <= span.computed_confidence <= 1.0


def test_intake_has_no_self_confidence_because_it_uses_no_model(invoice_trace):
    span = invoice_trace.span_for(StepName.INTAKE)
    assert span.self_confidence is None
    assert span.computed_confidence is not None


def test_a_step_takes_the_lowest_confidence_of_its_calls():
    from obspipe.tracing.models import LLMCall
    from obspipe.tracing.tracer import Tracer

    tracer = Tracer(Trace(doc_id="d"))
    with tracer.span("step", step=StepName.EXTRACTION) as span:
        tracer.record_llm_call(LLMCall(schema_name="A", self_confidence=5))
        tracer.record_llm_call(LLMCall(schema_name="B", self_confidence=2, confidence_reason="thin"))
    assert span.self_confidence == 2
    assert span.confidence_reason == "thin"


def test_low_confidence_is_reported_by_the_extractor(ctx, doc_of):
    trace = run_traced(doc_of("contract_no_dates.md"), FakeBackend(faults={"empty_extraction"}))
    span = trace.span_for(StepName.EXTRACTION)
    assert span.self_confidence == 1
    assert "low_extraction_confidence" in span.warning_codes()


# -- suspicion -----------------------------------------------------------


def _span(**kwargs) -> Span:
    return Span(trace_id="t", name="x", **kwargs)


def test_a_failed_span_outranks_a_merely_degraded_one():
    from obspipe.models.common import Severity, StepWarning

    failed = _span(status=SpanStatus.FAILED)
    degraded = _span(
        status=SpanStatus.DEGRADED,
        warnings=[StepWarning(code="c", message="m", severity=Severity.WARN)],
    )
    assert score_span(failed, False) > score_span(degraded, False)


def test_skipped_spans_are_never_suspects():
    assert score_span(_span(status=SpanStatus.SKIPPED), True) == 0.0


def test_cascade_warnings_do_not_incriminate_the_victim():
    """'the input was already truncated' is an alibi, not evidence."""
    from obspipe.models.common import Severity, StepWarning

    victim = _span(
        status=SpanStatus.DEGRADED,
        warnings=[
            StepWarning(
                code="extracted_from_truncated_text", message="m", severity=Severity.INFO
            )
        ],
    )
    assert score_span(victim, False) == 0.0


def test_the_earliest_failing_step_is_ranked_first(backend, doc_of):
    """Truncation in step 1 causes the missing dates in step 2, not the reverse."""
    trace = run_traced(doc_of("contract_long_truncated.md"), backend)
    assert trace.first_degraded_step is StepName.INTAKE
    assert trace.ranked_suspects()[0].step is StepName.INTAKE


def test_ranked_suspects_covers_every_step(invoice_trace):
    assert {s.step for s in invoice_trace.ranked_suspects()} == set(STEP_ORDER)
