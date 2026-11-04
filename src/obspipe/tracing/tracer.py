"""The tracer: ambient span context plus the ``@traced`` decorator.

The active span lives in a :class:`~contextvars.ContextVar`, which is what lets
instrumentation be one line on a step function rather than a parameter threaded
through every call site. It is also what lets the LLM backend attach a child
span to whichever step happens to be running, without the step passing anything
down to it.
"""

from __future__ import annotations

import inspect
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Iterator, TypeVar

from obspipe.models.common import StepName
from obspipe.tracing.models import LLMCall, Span, SpanKind, SpanStatus, utcnow
from obspipe.tracing.trace import Trace

T = TypeVar("T")

_current_tracer: ContextVar["Tracer | None"] = ContextVar("obspipe_tracer", default=None)
_current_span: ContextVar[Span | None] = ContextVar("obspipe_span", default=None)


class Tracer:
    """Builds one :class:`Trace`.

    Not thread-shared: one tracer per pipeline run. Spans are appended in
    completion order of *entry*, so ``trace.spans`` reads top to bottom the way
    the pipeline ran.
    """

    def __init__(self, trace: Trace, inline_limit: int | None = None) -> None:
        self.trace = trace
        self._inline_limit = inline_limit

    # -- payload helpers -------------------------------------------------

    def store(self, value: Any) -> Any:
        if self._inline_limit is None:
            return self.trace.payloads.put(value)
        return self.trace.payloads.put(value, inline_limit=self._inline_limit)

    def store_text(self, text: str) -> str:
        if self._inline_limit is None:
            return self.trace.payloads.put_text(text)
        return self.trace.payloads.put_text(text, inline_limit=self._inline_limit)

    # -- span lifecycle --------------------------------------------------

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.STEP,
        step: StepName | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        parent = _current_span.get()
        span = Span(
            trace_id=self.trace.trace_id,
            parent_span_id=parent.span_id if parent else None,
            name=name,
            kind=kind,
            step=step,
            inputs=inputs or {},
            started_at=utcnow(),
        )
        self.trace.spans.append(span)
        token = _current_span.set(span)
        clock = time.perf_counter()
        try:
            yield span
        finally:
            span.duration_ms = round((time.perf_counter() - clock) * 1000, 3)
            _current_span.reset(token)

    def record_llm_call(self, call: LLMCall) -> None:
        """Attach a model call to whichever span is currently open."""
        span = _current_span.get()
        if span is None:
            return
        span.llm_calls.append(call)
        # A step's confidence is the lowest its calls reported: one hedged call
        # is enough to make the whole step's output doubtful.
        confidences = [c.self_confidence for c in span.llm_calls if c.self_confidence]
        if confidences:
            worst = min(confidences)
            span.self_confidence = worst
            span.confidence_reason = next(
                (c.confidence_reason for c in span.llm_calls if c.self_confidence == worst),
                span.confidence_reason,
            )

    def skipped(self, step: StepName) -> Span:
        """Record a step that never ran because an earlier one failed."""
        span = Span(
            trace_id=self.trace.trace_id,
            name=step.value,
            kind=SpanKind.STEP,
            step=step,
            status=SpanStatus.SKIPPED,
            started_at=utcnow(),
        )
        self.trace.spans.append(span)
        return span


@contextmanager
def use_tracer(tracer: Tracer) -> Iterator[Tracer]:
    """Make ``tracer`` the ambient tracer for the duration of the block."""
    token = _current_tracer.set(tracer)
    try:
        yield tracer
    finally:
        _current_tracer.reset(token)


def current_tracer() -> Tracer | None:
    return _current_tracer.get()


def current_span() -> Span | None:
    return _current_span.get()


def annotate(computed_confidence: float | None = None, **attributes: Any) -> None:
    """Attach step-specific evidence to the span currently open.

    ``computed_confidence`` is the pipeline's own measure of how well this step
    went - grounded-span ratio, classification margin, text quality - and is
    deliberately separate from the model's self-rating so the two can be
    compared rather than conflated. No-op when nothing is being traced.
    """
    span = _current_span.get()
    if span is None:
        return
    if computed_confidence is not None:
        span.computed_confidence = max(0.0, min(1.0, round(computed_confidence, 4)))
    span.attributes.update(attributes)


def traced(step: StepName) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Instrument a step function. One line per step.

    Captures the step's bound arguments as span inputs and its return value as
    span output, both through the payload store so large documents are written
    once per trace rather than once per step. Status is derived from the
    returned artifact's own warnings, so a step that reports problems is
    degraded without having to say so twice.

    Runs unchanged when no tracer is active, which keeps the steps directly
    callable from tests.
    """

    def decorate(func: Callable[..., T]) -> Callable[..., T]:
        signature = inspect.signature(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            tracer = _current_tracer.get()
            if tracer is None:
                return func(*args, **kwargs)

            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            inputs = {
                name: tracer.store(value)
                for name, value in bound.arguments.items()
                if name not in _UNTRACED_ARGS
            }

            with tracer.span(step.value, kind=SpanKind.STEP, step=step, inputs=inputs) as span:
                try:
                    output = func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
                    span.status = SpanStatus.FAILED
                    span.error_type = getattr(exc, "code", type(exc).__name__)
                    span.error_message = getattr(exc, "message", str(exc))
                    as_warning = getattr(exc, "as_warning", None)
                    if callable(as_warning):
                        span.warnings.append(as_warning())
                    raise
                span.warnings = list(getattr(output, "warnings", []))
                span.status = (
                    SpanStatus.DEGRADED
                    if any(w.severity.value in ("warn", "error") for w in span.warnings)
                    else SpanStatus.OK
                )
                span.output = tracer.store(output)
                return output

        return wrapper

    return decorate


_UNTRACED_ARGS = frozenset({"backend", "ctx"})
"""Arguments that are configuration, not data.

The backend is an object with no useful serialization and the RunContext is
already recorded on the trace header; storing either per span would be noise.
"""
