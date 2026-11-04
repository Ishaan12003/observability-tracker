"""Records the feedback loop is made of: flags, eval cases, eval runs."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from obspipe.analysis.taxonomy import FailureCategory
from obspipe.models.common import StepName
from obspipe.models.document import ContentType


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class FlagStatus(str, Enum):
    OPEN = "open"
    CONFIRMED = "confirmed"
    """The human agreed with the automated diagnosis."""
    OVERRIDDEN = "overridden"
    """The human named a different step or category."""
    DISMISSED = "dismissed"


class Flag(BaseModel):
    """A human saying "this output is bad", and what they decided about why."""

    model_config = ConfigDict(extra="forbid")

    flag_id: str = Field(default_factory=lambda: _id("flag"))
    trace_id: str
    doc_id: str
    created_at: datetime = Field(default_factory=_now)
    reason: str = ""
    actor: str = "user"

    diagnosed_step: StepName | None = None
    diagnosed_category: FailureCategory | None = None
    diagnosis_confidence: float | None = None
    diagnosis_ms: float | None = None

    status: FlagStatus = FlagStatus.OPEN
    resolved_at: datetime | None = None
    confirmed_step: StepName | None = None
    confirmed_category: FailureCategory | None = None
    corrected_output: dict[str, Any] | None = None
    notes: str = ""
    eval_case_id: str | None = None

    @property
    def seconds_to_root_cause(self) -> float | None:
        if self.resolved_at is None:
            return None
        return (self.resolved_at - self.created_at).total_seconds()


class CaseInput(BaseModel):
    """Everything needed to replay the original run, self-contained."""

    model_config = ConfigDict(extra="forbid")

    source_name: str
    content: str
    content_type: ContentType = ContentType.PLAIN_TEXT
    backend: str = "fake"
    faults: list[str] = Field(default_factory=list)
    """Model behaviours the original run exhibited (fake backend only)."""


class EvalCase(BaseModel):
    """One known failure, frozen into a regression test.

    Created automatically when a human confirms a flag. ``failure_codes`` and
    ``corrected_output`` are the two halves of the pass condition: the warning
    that marked the failure must not recur on the failing step, and the step's
    projected output must match the correction.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(default_factory=lambda: _id("case"))
    created_at: datetime = Field(default_factory=_now)
    source_flag_id: str | None = None
    source_trace_id: str
    doc_id: str
    input: CaseInput
    failing_step: StepName
    failure_category: FailureCategory
    failure_codes: list[str] = Field(default_factory=list)
    bad_output: dict[str, Any] = Field(default_factory=dict)
    corrected_output: dict[str, Any] | None = None
    notes: str = ""


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    passed: bool
    trace_id: str | None = None
    recurring_codes: list[str] = Field(default_factory=list)
    divergences: list[dict[str, Any]] = Field(default_factory=list)
    detail: str = ""


class EvalRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: _id("eval"))
    started_at: datetime = Field(default_factory=_now)
    pipeline_version: str
    label: str = ""
    duration_ms: float = 0.0
    results: list[CaseResult] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)
