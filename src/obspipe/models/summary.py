from __future__ import annotations

from typing import Any

from pydantic import Field

from obspipe.models.common import StepArtifact
from obspipe.models.document import DocType


class SummaryOutput(StepArtifact):
    """Step 4 output.

    ``type_specific`` holds the fields that only make sense for the chosen
    document type (parties+term for a contract, totals for an invoice, ...).
    Its shape is therefore downstream of step 3: a misclassification produces a
    structurally wrong summary, which is the propagation case the tracer must
    be able to walk backwards.
    """

    doc_type: DocType
    headline: str
    bullets: list[str] = Field(default_factory=list)
    type_specific: dict[str, Any] = Field(default_factory=dict)
    unresolved_questions: list[str] = Field(
        default_factory=list,
        description="Things the summarizer refused to invent, e.g. a missing effective date.",
    )
    self_confidence: int = Field(
        default=3, ge=1, le=5, description="The summarizer's own 1-5 rating of this summary."
    )
    confidence_reason: str = ""
    model_name: str | None = None
