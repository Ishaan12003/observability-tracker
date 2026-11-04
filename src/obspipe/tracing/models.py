"""The trace data model.

Deliberately our own rather than OpenTelemetry's. An OTel attribute must be a
primitive, which makes the one thing this project exists to store - the full
input and output of every step - the one thing it cannot hold. The span below
keeps the payloads, the prompt, the raw model response, the self-reported
confidence and the deterministic warnings side by side, because failure
attribution needs all four in the same place.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from obspipe.models.common import Severity, StepName, StepWarning
from obspipe.tracing.payloads import PayloadStore


def new_id(size: int = 16) -> str:
    return uuid.uuid4().hex[:size]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SpanStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    """Produced output, but flagged problems with it."""
    FAILED = "failed"
    SKIPPED = "skipped"


class SpanKind(str, Enum):
    PIPELINE = "pipeline"
    STEP = "step"
    LLM = "llm"


MIN_CONFIDENCE = 1
MAX_CONFIDENCE = 5
LOW_CONFIDENCE = 2
"""At or below this, the model is telling you it is guessing."""


class TokenUsage(BaseModel):
    """Token counts, or ``None`` when the backend does not report them.

    The fake backend reports nothing rather than zero: a zero would silently
    average into cost dashboards as though the call were free.
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None

    @classmethod
    def from_usage_metadata(cls, usage: dict[str, Any] | None) -> TokenUsage:
        if not usage:
            return cls()
        details = usage.get("output_token_details") or {}
        return cls(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            reasoning_tokens=details.get("reasoning"),
        )


class LLMCall(BaseModel):
    """One model invocation, captured whole.

    ``prompt`` and ``raw_response`` are what make a failed run reproducible:
    with both, the exact call can be replayed without re-running the pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(default_factory=lambda: new_id(12))
    schema_name: str
    model: str | None = None
    backend: str = "fake"
    started_at: datetime = Field(default_factory=utcnow)
    duration_ms: float = 0.0

    prompt: Any = None
    """Messages sent, as [{role, content}]. May be a payload reference."""

    raw_response: Any = None
    """The model's unparsed reply. May be a payload reference."""

    parsed: Any = None
    """The validated object the step received. May be a payload reference."""

    parsing_error: str | None = None
    finish_reason: str | None = None
    tokens: TokenUsage = Field(default_factory=TokenUsage)

    self_confidence: int | None = Field(default=None, ge=1, le=5)
    confidence_reason: str | None = None

    @property
    def is_low_confidence(self) -> bool:
        return self.self_confidence is not None and self.self_confidence <= LOW_CONFIDENCE


class Span(BaseModel):
    """One unit of traced work. Supersedes Phase 1's StepRecord.

    A step span holds the deterministic evidence (warnings, status, timing) and
    the model evidence (nested LLM calls, self-reported confidence). Suspicion
    is computed from both, because the two disagree often and the deterministic
    half is the more trustworthy of the two.
    """

    # extra="ignore", not "forbid": worst_severity below is a computed field,
    # emitted by model_dump but not accepted as input. A strict model cannot
    # re-validate its own output, and a stored trace must reload.
    model_config = ConfigDict(extra="ignore")

    span_id: str = Field(default_factory=lambda: new_id(12))
    parent_span_id: str | None = None
    trace_id: str
    name: str
    kind: SpanKind = SpanKind.STEP
    step: StepName | None = None

    status: SpanStatus = SpanStatus.OK
    started_at: datetime = Field(default_factory=utcnow)
    duration_ms: float = 0.0

    inputs: dict[str, Any] = Field(default_factory=dict)
    output: Any = None

    warnings: list[StepWarning] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None

    llm_calls: list[LLMCall] = Field(default_factory=list)

    self_confidence: int | None = Field(default=None, ge=1, le=5)
    confidence_reason: str | None = None
    computed_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence derived from the pipeline's own checks, not self-reported.",
    )

    suspicion: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How likely this step is the origin of a bad final output.",
    )

    attributes: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def worst_severity(self) -> Severity | None:
        if self.status is SpanStatus.FAILED:
            return Severity.ERROR
        present = {w.severity for w in self.warnings}
        for level in (Severity.ERROR, Severity.WARN, Severity.INFO):
            if level in present:
                return level
        return None

    @property
    def total_tokens(self) -> int | None:
        counts = [c.tokens.total_tokens for c in self.llm_calls if c.tokens.total_tokens]
        return sum(counts) if counts else None

    def warning_codes(self) -> list[str]:
        return [w.code for w in self.warnings]

    def has_error_warning(self) -> bool:
        return any(w.severity is Severity.ERROR for w in self.warnings)
