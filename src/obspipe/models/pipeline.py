from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from obspipe.models.classification import ClassificationOutput
from obspipe.models.common import StepName, StepWarning
from obspipe.models.document import DocType
from obspipe.models.extraction import ExtractionOutput
from obspipe.models.intake import IntakeOutput
from obspipe.models.summary import SummaryOutput
from obspipe.tracing.models import Span, SpanStatus


StepStatus = SpanStatus
"""Phase 1 name for :class:`~obspipe.tracing.models.SpanStatus`.

Steps and spans are the same thing now, so they share one status enum rather
than two that must be kept in agreement.
"""


class RunContext(BaseModel):
    """Per-run state threaded through every step.

    Phase 1 uses it only for identity and config. It exists now so that Phase 2
    can attach a tracer here without changing a single step signature.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    doc_id: str
    backend: str = "fake"
    model: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    """Everything one document run produced, including partial runs."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    doc_id: str
    source_name: str
    backend: str
    model: str | None = None
    started_at: datetime
    duration_ms: float = 0.0

    intake: IntakeOutput | None = None
    extraction: ExtractionOutput | None = None
    classification: ClassificationOutput | None = None
    summary: SummaryOutput | None = None

    spans: list[Span] = Field(default_factory=list)
    """Step spans, in execution order. Owned by the Trace; mirrored here so a
    PipelineResult still answers "what happened" on its own."""

    trace_id: str | None = None

    @property
    def status(self) -> StepStatus:
        if any(s.status is StepStatus.FAILED for s in self.spans):
            return StepStatus.FAILED
        if any(s.status is StepStatus.DEGRADED for s in self.spans):
            return StepStatus.DEGRADED
        return StepStatus.OK

    @property
    def failed_step(self) -> StepName | None:
        for record in self.spans:
            if record.status is StepStatus.FAILED:
                return record.step
        return None

    @property
    def first_degraded_step(self) -> StepName | None:
        """Earliest step that flagged a problem - the default suspect."""
        for record in self.spans:
            if record.status in (StepStatus.DEGRADED, StepStatus.FAILED):
                return record.step
        return None

    @property
    def doc_type(self) -> DocType:
        return self.classification.doc_type if self.classification else DocType.UNKNOWN

    def all_warnings(self) -> list[tuple[StepName, StepWarning]]:
        return [(r.step, w) for r in self.spans for w in r.warnings]

    def record_for(self, step: StepName) -> Span | None:
        return next((r for r in self.spans if r.step is step), None)


class DocumentExpectation(BaseModel):
    """Sidecar metadata shipped with each fixture document.

    Phase 1 uses it for tests. Phase 4 promotes it to eval-dataset ground truth.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    expected_type: DocType
    expected_failure: str | None = None
    expected_warning_codes: list[str] = Field(default_factory=list)
    simulated_faults: list[str] = Field(
        default_factory=list,
        description="Model misbehaviour the fake backend reproduces on this document.",
    )
    notes: str = ""

    @classmethod
    def load(cls, document_path: Any) -> "DocumentExpectation | None":
        """Read the ``<filename>.meta.json`` sidecar next to a fixture document."""
        import json
        from pathlib import Path

        meta = Path(str(document_path) + ".meta.json")
        if not meta.exists():
            return None
        return cls.model_validate(json.loads(meta.read_text(encoding="utf-8")))
