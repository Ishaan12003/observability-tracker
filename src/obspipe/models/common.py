from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class StepName(str, Enum):
    """Canonical step identifiers. These become span names in Phase 2."""

    INTAKE = "intake"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class StepWarning(BaseModel):
    """A self-reported quality signal from a step.

    Steps emit these when they do something questionable but recoverable
    (truncated input, unparseable amount, coin-flip classification). They are
    the raw material for failure attribution: when a final output is bad, the
    earliest step carrying a high-severity warning is the prime suspect.
    """

    model_config = ConfigDict(frozen=True)

    code: str = Field(description="Stable machine-readable code, e.g. 'no_dates_found'.")
    message: str = Field(description="Human-readable detail.")
    severity: Severity = Severity.WARN
    field_path: str | None = Field(
        default=None,
        description="Dotted path to the offending field, when one exists.",
    )

    def __str__(self) -> str:  # pragma: no cover - display helper
        where = f" [{self.field_path}]" if self.field_path else ""
        return f"{self.severity.value}:{self.code}{where}: {self.message}"


class StepArtifact(BaseModel):
    """Base class for every step output.

    Guarantees the two fields the observability layer relies on: the document
    the artifact belongs to, and whatever the producing step wants to confess.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    warnings: list[StepWarning] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(w.severity is Severity.ERROR for w in self.warnings)

    def warning_codes(self) -> list[str]:
        return [w.code for w in self.warnings]
