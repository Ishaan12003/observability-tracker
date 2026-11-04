"""Ranking which step is most likely responsible for a bad final output.

The scoring deliberately weights the pipeline's own deterministic findings
above the model's opinion of itself. A grounding failure is a fact; a
self-reported 2/5 is a hint. Weighting them the other way round produces a
ranking that follows the model's mood rather than the evidence.

Position matters too: when several steps look bad, the earliest one usually
caused the rest. That is the difference between finding a root cause and
finding a symptom.
"""

from __future__ import annotations

from obspipe.models.common import Severity
from obspipe.tracing.models import MAX_CONFIDENCE, Span, SpanKind, SpanStatus
from obspipe.tracing.trace import Trace

WEIGHT_FAILURE = 1.00
"""A step that raised is the origin unless something upstream degraded it."""

WEIGHT_ERROR_WARNING = 0.55
WEIGHT_WARN_WARNING = 0.22
WEIGHT_SELF_CONFIDENCE = 0.18
WEIGHT_COMPUTED_CONFIDENCE = 0.25
WEIGHT_EARLIEST = 0.15
"""Bonus for being the first step to go wrong - cause beats symptom."""

CASCADE_CODES = frozenset(
    {
        "extracted_from_truncated_text",
        "classified_from_truncated_text",
        "classified_on_degraded_entities",
    }
)
"""Warnings that exist to say 'the problem came from upstream'.

A step carrying only these is a victim, not a suspect, so they must not inflate
its score.
"""


def _warning_pressure(span: Span) -> float:
    score = 0.0
    for warning in span.warnings:
        if warning.code in CASCADE_CODES:
            continue
        if warning.severity is Severity.ERROR:
            score += WEIGHT_ERROR_WARNING
        elif warning.severity is Severity.WARN:
            score += WEIGHT_WARN_WARNING
    return score


def score_span(span: Span, is_earliest_problem: bool) -> float:
    if span.status is SpanStatus.SKIPPED:
        return 0.0

    score = _warning_pressure(span)

    if span.status is SpanStatus.FAILED:
        score += WEIGHT_FAILURE

    if span.self_confidence is not None:
        # 5/5 contributes nothing; 1/5 contributes the full weight.
        shortfall = (MAX_CONFIDENCE - span.self_confidence) / (MAX_CONFIDENCE - 1)
        score += WEIGHT_SELF_CONFIDENCE * shortfall

    if span.computed_confidence is not None:
        score += WEIGHT_COMPUTED_CONFIDENCE * (1.0 - span.computed_confidence)

    if is_earliest_problem:
        score += WEIGHT_EARLIEST

    return round(min(score, 1.0), 4)


def score_trace(trace: Trace) -> Trace:
    """Fill ``span.suspicion`` for every step span in ``trace``."""
    steps = trace.step_spans()
    earliest = next(
        (
            s.span_id
            for s in steps
            if s.status in (SpanStatus.DEGRADED, SpanStatus.FAILED)
            and any(w.code not in CASCADE_CODES for w in s.warnings)
        ),
        None,
    )
    for span in steps:
        span.suspicion = score_span(span, is_earliest_problem=span.span_id == earliest)
    for span in trace.spans:
        if span.kind is not SpanKind.STEP:
            span.suspicion = 0.0
    return trace
