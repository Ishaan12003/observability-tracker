"""The Trace: one pipeline execution, whole.

Holds every span, the payload store they share, the final output, and the
status of the run. This is the artifact the storage layer writes and the trace
explorer reads.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer, model_validator

from obspipe.models.common import StepName, StepWarning
from obspipe.models.pipeline import PipelineResult
from obspipe.tracing.models import LLMCall, Span, SpanKind, SpanStatus, new_id, utcnow
from obspipe.tracing.payloads import PayloadStore


class Trace(BaseModel):
    """Everything one document run produced, plus how it got there.

    ``extra="ignore"`` rather than ``forbid``: the computed fields below are
    emitted by ``model_dump`` and must be tolerated when a stored trace is read
    back, or persisted runs cannot be reloaded.
    """

    model_config = ConfigDict(extra="ignore")

    trace_id: str = Field(default_factory=lambda: new_id(16))
    doc_id: str
    source_name: str = ""
    backend: str = "fake"
    model: str | None = None

    started_at: datetime = Field(default_factory=utcnow)
    duration_ms: float = 0.0
    status: SpanStatus = SpanStatus.OK

    spans: list[Span] = Field(default_factory=list)
    payloads: PayloadStore = Field(default_factory=PayloadStore)
    result: PipelineResult | None = None

    tags: dict[str, Any] = Field(
        default_factory=dict,
        description="Run configuration worth filtering on: pipeline_version, batch, faults.",
    )

    final_score: float | None = Field(
        default=None,
        description="Human verdict on the final output. Filled by the Phase 4 feedback API.",
    )
    feedback_notes: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failed_step(self) -> StepName | None:
        for span in self.step_spans():
            if span.status is SpanStatus.FAILED:
                return span.step
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def first_degraded_step(self) -> StepName | None:
        """Earliest step that flagged a problem - the default suspect."""
        for span in self.step_spans():
            if span.status in (SpanStatus.DEGRADED, SpanStatus.FAILED):
                return span.step
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int | None:
        counts = [call.tokens.total_tokens for call in self.llm_calls() if call.tokens.total_tokens]
        return sum(counts) if counts else None

    def step_spans(self) -> list[Span]:
        """Step spans in execution order."""
        return [s for s in self.spans if s.kind is SpanKind.STEP]

    def span_for(self, step: StepName) -> Span | None:
        return next((s for s in self.step_spans() if s.step is step), None)

    def span_by_id(self, span_id: str) -> Span | None:
        return next((s for s in self.spans if s.span_id == span_id), None)

    def children_of(self, span_id: str) -> list[Span]:
        return [s for s in self.spans if s.parent_span_id == span_id]

    def llm_calls(self) -> Iterable[LLMCall]:
        for span in self.spans:
            yield from span.llm_calls

    def all_warnings(self) -> list[tuple[StepName | None, StepWarning]]:
        return [(s.step, w) for s in self.step_spans() for w in s.warnings]

    def ranked_suspects(self) -> list[Span]:
        """Step spans ordered by how likely each is to be the origin of a bad output."""
        return sorted(
            self.step_spans(),
            key=lambda s: (-s.suspicion, self._order(s)),
        )

    def resolve(self, value: Any) -> Any:
        """Expand any payload references inside ``value``."""
        return self.payloads.resolve_deep(value)

    @field_serializer("result")
    def _serialize_result(self, result: PipelineResult | None) -> Any:
        """Write the result without its spans.

        ``result.spans`` and ``trace.spans`` hold the same objects. Serializing
        both writes every span, every warning and every payload reference
        twice; on a long document that was a third of the file. The spans are
        put back on load by :meth:`_rehydrate_result`.
        """
        if result is None:
            return None
        data = result.model_dump(mode="json")
        data.pop("spans", None)
        return data

    @model_validator(mode="after")
    def _rehydrate_result(self) -> "Trace":
        if self.result is not None and not self.result.spans:
            self.result.spans = self.step_spans()
        return self

    def _order(self, span: Span) -> int:
        try:
            return self.spans.index(span)
        except ValueError:  # pragma: no cover - defensive
            return len(self.spans)
