"""The orchestrator.

Phase 1 kept step records by hand here. Phase 2 replaces that with a tracer:
the orchestrator now decides only *what runs next*, while what happened is
captured by the ``@traced`` decorator on each step and by the observed backend
underneath it. The result is that this file knows nothing about prompts, tokens
or payloads, and a new step is instrumented by one decorator line rather than
by editing this loop.

A failed step stops the run and every later step is recorded as skipped, so a
partial trace is still a complete record of what was attempted.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field

from obspipe.llm.backends import LLMBackend, get_backend
from obspipe.llm.observed import ObservedBackend
from obspipe.models.common import StepName
from obspipe.models.document import RawDocument
from obspipe.models.pipeline import PipelineResult, RunContext, StepStatus
from obspipe.steps import run_classification, run_extraction, run_intake, run_summarization
from obspipe.steps.base import KNOWN_GUARDS, StepFailure
from obspipe.steps.intake import MAX_CHARS
from obspipe.tracing.models import SpanKind, SpanStatus, utcnow
from obspipe.tracing.suspicion import score_trace
from obspipe.tracing.trace import Trace
from obspipe.tracing.tracer import Tracer, use_tracer

STEP_ORDER: tuple[StepName, ...] = (
    StepName.INTAKE,
    StepName.EXTRACTION,
    StepName.CLASSIFICATION,
    StepName.SUMMARIZATION,
)

class PipelineConfig(BaseModel):
    """A named, reproducible pipeline configuration.

    Regression tracking needs to know *which pipeline* an eval run measured.
    Each version is a concrete change an engineer would ship in response to a
    diagnosed failure class: a larger intake budget for truncation, grounding
    guards for hallucinated spans and invented figures.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    max_chars: int = MAX_CHARS
    guards: frozenset[str] = Field(default_factory=frozenset)
    notes: str = ""


PIPELINE_VERSIONS: dict[str, PipelineConfig] = {
    config.version: config
    for config in (
        PipelineConfig(version="v1", notes="Baseline: 12k intake budget, warn-only checks."),
        PipelineConfig(
            version="v2",
            max_chars=48_000,
            notes="Intake budget raised to 48k after truncation was diagnosed as a root cause.",
        ),
        PipelineConfig(
            version="v3",
            max_chars=48_000,
            guards=KNOWN_GUARDS,
            notes="v2 plus grounding guards: ungrounded extraction spans are dropped, "
            "unsupported summary figures are withheld.",
        ),
    )
}
CURRENT_VERSION = "v3"


def get_config(version: str | None = None) -> PipelineConfig:
    name = version or CURRENT_VERSION
    if name not in PIPELINE_VERSIONS:
        raise KeyError(f"Unknown pipeline version '{name}'. Known: {sorted(PIPELINE_VERSIONS)}")
    return PIPELINE_VERSIONS[name]


def run_traced(
    document: RawDocument,
    backend: LLMBackend | None = None,
    ctx: RunContext | None = None,
    max_chars: int | None = None,
    config: PipelineConfig | None = None,
    tags: dict | None = None,
) -> Trace:
    """Run all four steps over one document and return the complete trace.

    Never raises for document-level problems: a broken document produces a
    Trace whose status is ``failed`` and whose ``failed_step`` names the step
    that stopped it, which is strictly more useful than a traceback.

    ``config`` selects a pipeline version; without one the Phase 1 defaults
    apply (12k budget, no guards), so existing callers are unaffected.
    """
    backend = backend or get_backend("fake")
    ctx = ctx or RunContext(
        doc_id=document.doc_id,
        backend=backend.name,
        model=getattr(backend, "model", None),
    )
    if config is not None:
        ctx.extra.setdefault("guards", sorted(config.guards))
        max_chars = max_chars or config.max_chars
    max_chars = max_chars or MAX_CHARS

    trace = Trace(
        trace_id=ctx.run_id,
        doc_id=document.doc_id,
        source_name=document.source_name,
        backend=backend.name,
        model=getattr(backend, "model", None),
        started_at=ctx.started_at,
        tags=dict(tags or {}),
    )
    if config is not None:
        trace.tags.setdefault("pipeline_version", config.version)
    faults = sorted(getattr(backend, "faults", ()) or ())
    if faults:
        trace.tags.setdefault("faults", faults)
    if max_chars != MAX_CHARS:
        trace.tags.setdefault("max_chars", max_chars)
    tracer = Tracer(trace)
    observed = ObservedBackend(backend, tracer)

    result = PipelineResult(
        run_id=ctx.run_id,
        trace_id=trace.trace_id,
        doc_id=document.doc_id,
        source_name=document.source_name,
        backend=backend.name,
        model=getattr(backend, "model", None),
        started_at=ctx.started_at,
    )

    clock = time.perf_counter()
    with use_tracer(tracer), tracer.span(
        "pipeline", kind=SpanKind.PIPELINE, inputs={"doc_id": document.doc_id}
    ) as root:
        aborted = False
        for step in STEP_ORDER:
            if aborted:
                tracer.skipped(step)
                continue
            try:
                _dispatch(step, document, result, ctx, observed, max_chars)
            except StepFailure:
                aborted = True
            except Exception:  # noqa: BLE001 - already recorded on the span
                aborted = True

        root.status = SpanStatus.FAILED if aborted else SpanStatus.OK

    trace.duration_ms = round((time.perf_counter() - clock) * 1000, 3)
    result.duration_ms = trace.duration_ms
    result.spans = trace.step_spans()
    trace.result = result
    trace.status = result.status
    score_trace(trace)
    return trace


def _dispatch(
    step: StepName,
    document: RawDocument,
    result: PipelineResult,
    ctx: RunContext,
    backend: LLMBackend,
    max_chars: int,
) -> None:
    """Call one step and hang its output on the result."""
    if step is StepName.INTAKE:
        result.intake = run_intake(document, ctx, max_chars=max_chars)
    elif step is StepName.EXTRACTION:
        result.extraction = run_extraction(result.intake, ctx, backend)
    elif step is StepName.CLASSIFICATION:
        result.classification = run_classification(result.intake, result.extraction, ctx, backend)
    elif step is StepName.SUMMARIZATION:
        result.summary = run_summarization(
            result.intake, result.extraction, result.classification, ctx, backend
        )


def run_pipeline(
    document: RawDocument,
    backend: LLMBackend | None = None,
    ctx: RunContext | None = None,
    max_chars: int | None = None,
) -> PipelineResult:
    """Phase 1 entry point: the result without the surrounding trace."""
    return run_traced(document, backend, ctx, max_chars).result  # type: ignore[return-value]

