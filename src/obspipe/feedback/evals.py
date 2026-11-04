"""Turning confirmed flags into eval cases, and replaying them.

A flag becomes a case the moment a human confirms the root cause. The case is
self-contained - it carries the document text and the run conditions - so the
dataset keeps working after the original trace files are gone.
"""

from __future__ import annotations

import time

from obspipe.analysis.golden import compare, project
from obspipe.analysis.taxonomy import ADVISORY_CODES, FailureCategory
from obspipe.feedback.models import CaseInput, CaseResult, EvalCase, EvalRun, Flag
from obspipe.llm.backends import LLMBackend, get_backend
from obspipe.models.common import StepName
from obspipe.models.document import ContentType, RawDocument
from obspipe.pipeline import PipelineConfig, run_traced
from obspipe.tracing.models import SpanStatus
from obspipe.tracing.suspicion import CASCADE_CODES
from obspipe.tracing.trace import Trace


def case_from_flag(
    flag: Flag,
    trace: Trace,
    step: StepName,
    category: FailureCategory,
    corrected_output: dict | None,
) -> EvalCase:
    """Freeze one confirmed failure into a regression test."""
    intake = trace.span_for(StepName.INTAKE)
    document = trace.resolve(intake.inputs.get("document")) if intake else {}
    span = trace.span_for(step)
    output = trace.resolve(span.output) if span and span.output is not None else None

    codes: list[str] = []
    if span is not None:
        if span.status is SpanStatus.FAILED and span.error_type:
            codes.append(span.error_type)
        codes.extend(
            w.code
            for w in span.warnings
            if w.severity.value in ("warn", "error")
            and w.code not in CASCADE_CODES
            and w.code not in ADVISORY_CODES
            and w.code not in codes
        )

    return EvalCase(
        source_flag_id=flag.flag_id,
        source_trace_id=trace.trace_id,
        doc_id=trace.doc_id,
        input=CaseInput(
            source_name=document.get("source_name", trace.source_name),
            content=document.get("content", ""),
            content_type=ContentType(document.get("content_type", ContentType.PLAIN_TEXT.value)),
            backend=trace.backend,
            faults=list(trace.tags.get("faults") or []),
        ),
        failing_step=step,
        failure_category=category,
        failure_codes=codes,
        bad_output=project(step, output, span.warning_codes() if span else []) if output else {},
        corrected_output=corrected_output or None,
        notes=flag.notes or flag.reason,
    )


def _backend_for(case: EvalCase, override: LLMBackend | None) -> LLMBackend:
    if override is not None:
        return override
    if case.input.backend == "fake":
        return get_backend("fake", faults=set(case.input.faults))
    return get_backend(case.input.backend)


def evaluate_case(
    case: EvalCase,
    config: PipelineConfig,
    backend: LLMBackend | None = None,
    batch: str | None = None,
) -> tuple[CaseResult, Trace]:
    """Replay one case against ``config`` and decide whether it is fixed."""
    document = RawDocument(
        doc_id=case.doc_id,
        source_name=case.input.source_name,
        content=case.input.content,
        content_type=case.input.content_type,
    )
    tags = {"kind": "eval", "eval_case": case.case_id}
    if batch:
        tags["batch"] = batch
    trace = run_traced(document, _backend_for(case, backend), config=config, tags=tags)
    span = trace.span_for(case.failing_step)

    if span is None or span.status is SpanStatus.SKIPPED:
        return (
            CaseResult(case_id=case.case_id, passed=False, trace_id=trace.trace_id,
                       detail=f"{case.failing_step.value} never ran"),
            trace,
        )

    present = set(span.warning_codes())
    if span.status is SpanStatus.FAILED and span.error_type:
        present.add(span.error_type)
    recurring = [code for code in case.failure_codes if code in present]

    divergences = []
    if case.corrected_output:
        output = trace.resolve(span.output) if span.output is not None else None
        divergences = compare(case.corrected_output, project(case.failing_step, output, list(present)))

    passed = not recurring and not divergences and span.status is not SpanStatus.FAILED
    if passed:
        detail = "fixed: failure no longer reproduces"
    elif recurring:
        detail = f"still failing: {', '.join(recurring)}"
    elif divergences:
        detail = f"still failing: {divergences[0].describe()}"
    else:
        detail = f"still failing: {case.failing_step.value} {span.status.value}"
    return (
        CaseResult(
            case_id=case.case_id,
            passed=passed,
            trace_id=trace.trace_id,
            recurring_codes=recurring,
            divergences=[d.model_dump(mode="json") for d in divergences],
            detail=detail,
        ),
        trace,
    )


def run_evals(
    cases: list[EvalCase],
    config: PipelineConfig,
    label: str = "",
    backend: LLMBackend | None = None,
) -> tuple[EvalRun, list[Trace]]:
    run = EvalRun(pipeline_version=config.version, label=label or config.notes[:80])
    batch = f"eval-{run.run_id}"
    clock = time.perf_counter()
    traces: list[Trace] = []
    for case in cases:
        result, trace = evaluate_case(case, config, backend=backend, batch=batch)
        run.results.append(result)
        traces.append(trace)
    run.duration_ms = round((time.perf_counter() - clock) * 1000, 3)
    return run, traces
