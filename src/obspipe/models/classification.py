from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, computed_field

from obspipe.models.common import StepArtifact
from obspipe.models.document import DocType

AMBIGUITY_MARGIN = 0.15
"""Below this score gap, the top two labels are treated as a coin flip."""

LOW_CONFIDENCE = 0.45


class TypeScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_type: DocType
    score: float = Field(ge=0.0, le=1.0)


class ClassificationOutput(StepArtifact):
    """Step 3 output.

    ``extra="ignore"`` overrides the base ``forbid``: the computed fields below
    are emitted by ``model_dump`` and must be tolerated when a stored trace is
    read back, otherwise persisted runs cannot be reloaded.

    Scores for every candidate label are kept, not just the winner. The gap
    between first and second place is the single most useful signal for
    "the classifier guessed", which in turn explains a wrong summary shape.
    """

    model_config = ConfigDict(extra="ignore")

    doc_type: DocType
    scores: list[TypeScore] = Field(default_factory=list)
    rationale: str = ""
    self_confidence: int = Field(
        default=3, ge=1, le=5, description="The classifier's own 1-5 rating of this label."
    )
    confidence_reason: str = ""
    model_name: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> float:
        ranked = self._ranked()
        return ranked[0].score if ranked else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def runner_up(self) -> DocType | None:
        ranked = self._ranked()
        return ranked[1].doc_type if len(ranked) > 1 else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def margin(self) -> float:
        ranked = self._ranked()
        if len(ranked) < 2:
            return 1.0
        return round(ranked[0].score - ranked[1].score, 4)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ambiguous(self) -> bool:
        return self.margin < AMBIGUITY_MARGIN

    def _ranked(self) -> list[TypeScore]:
        return sorted(self.scores, key=lambda s: s.score, reverse=True)
