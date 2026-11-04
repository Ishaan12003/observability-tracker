from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from obspipe.models.common import StepArtifact
from obspipe.models.document import ContentType


class Section(BaseModel):
    """A heading-delimited chunk found during normalization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    heading: str
    start_line: int
    end_line: int


class IntakeOutput(StepArtifact):
    """Step 1 output: deterministic normalization, no LLM involved.

    ``normalized_text`` is what every downstream step reads. If intake mangles
    it, everything after is garbage - which is exactly the attribution case we
    want to be able to prove.
    """

    source_name: str
    content_type: ContentType
    normalized_text: str
    char_count: int
    word_count: int
    line_count: int
    sections: list[Section] = Field(default_factory=list)
    truncated: bool = False
    original_char_count: int = 0
    ocr_artifacts_repaired: int = Field(
        default=0,
        description="Count of hyphen line-breaks and broken ligatures rejoined.",
    )
