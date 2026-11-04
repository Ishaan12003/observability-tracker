"""Schemas the LLM is asked to fill.

Kept separate from the public step outputs on purpose. These describe what a
model is *competent* to produce - spans it can see and labels it can judge -
and nothing more. Numeric parsing, date validation, margin arithmetic and
warning generation all happen afterwards in Python. When something is wrong,
you can tell whether the model or the post-processing produced it.
"""

from __future__ import annotations

from typing import get_origin

from pydantic import BaseModel, Field, model_validator

from obspipe.models.document import DocType


class LLMSchema(BaseModel):
    """Base for every schema the model is asked to fill.

    List fields are declared nullable throughout. A model that emits ``null``
    for a section the document genuinely lacks - a research abstract has no
    recommendations - is answering correctly, but a non-nullable array in the
    JSON schema makes the provider reject the whole tool call with a 400 before
    any Python validation runs. Declaring the field nullable keeps the request
    legal; this validator turns the null straight back into an empty list so
    every consumer downstream still sees a list.
    """

    @model_validator(mode="after")
    def _null_lists_become_empty(self):
        for name, field in type(self).model_fields.items():
            if getattr(self, name) is None and _is_list_field(field.annotation):
                setattr(self, name, [])
        return self


def _is_list_field(annotation: object) -> bool:
    """True for ``list[X]`` and ``list[X] | None``."""
    if get_origin(annotation) is list:
        return True
    return any(get_origin(arg) is list for arg in getattr(annotation, "__args__", ()))


class ConfidenceReported(LLMSchema):
    """Mixin for the three top-level step outputs.

    The model rates its own answer 1-5. Treat it as the weakest signal on the
    span: self-reported confidence is poorly calibrated, and a model will
    happily return 4 on a fabricated figure. Its value is as a cheap prior that
    the pipeline's own deterministic checks - span grounding, recall sweeps,
    classification margin - then confirm or contradict.
    """

    self_confidence: int = Field(
        default=3,
        ge=1,
        le=5,
        description="How confident you are in THIS answer. 1 = guessing, the document barely "
        "supports it. 3 = plausible with gaps. 5 = the document states it explicitly. "
        "Rate honestly; a low score is more useful than a wrong high one.",
    )
    confidence_reason: str = Field(
        default="",
        description="One clause naming the single thing that most limits your confidence.",
    )


class RawParty(LLMSchema):
    name: str = Field(description="Party name exactly as written in the document.")
    role: str | None = Field(default=None, description="buyer, vendor, licensor, sender, ...")
    organization: bool = Field(default=True, description="False if a natural person.")


class RawDate(LLMSchema):
    raw_text: str = Field(description="The date span copied verbatim from the document.")
    iso_date: str | None = Field(
        default=None, description="YYYY-MM-DD if unambiguous, otherwise null. Never guess a year."
    )
    role: str = Field(
        default="unknown",
        description="What the date is for. Prefer one of: effective, expiry, signed, issued, "
        "due, mentioned, unknown. If none of those fit, use your own short lowercase label.",
    )


class RawAmount(LLMSchema):
    raw_text: str = Field(
        description="A sum of money copied verbatim including its currency symbol, e.g. "
        "'\u20ac1.250,00'. Monetary values only - never a measurement, vital sign, lab "
        "value, percentage or duration."
    )
    label: str | None = Field(default=None, description="What the amount is for.")
    kind: str = Field(
        default="unknown",
        description="Prefer one of: line_item, subtotal, tax, total, penalty, advance, "
        "balance, unknown. If none of those fit, use your own short lowercase label.",
    )


class RawIdentifier(LLMSchema):
    kind: str = Field(description="invoice_number, po_number, contract_ref, case_number, ...")
    value: str


class ExtractionLLMOutput(ConfidenceReported):
    """Step 2 model output. Spans only - no arithmetic."""

    parties: list[RawParty] | None = Field(default_factory=list)
    dates: list[RawDate] | None = Field(default_factory=list)
    amounts: list[RawAmount] | None = Field(default_factory=list)
    key_terms: list[str] | None = Field(
        default_factory=list, description="5-10 salient domain terms or clause names."
    )
    identifiers: list[RawIdentifier] | None = Field(default_factory=list)


class ClassificationLLMOutput(ConfidenceReported):
    """Step 3 model output.

    A score for every candidate label, not a single winner. The pipeline
    derives the winner and the margin itself so that 'the model was torn'
    survives as evidence instead of collapsing into one label.
    """

    contract: float = Field(ge=0.0, le=1.0)
    invoice: float = Field(ge=0.0, le=1.0)
    report: float = Field(ge=0.0, le=1.0)
    correspondence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(description="One or two sentences citing document features.")

    def as_pairs(self) -> list[tuple[DocType, float]]:
        return [
            (DocType.CONTRACT, self.contract),
            (DocType.INVOICE, self.invoice),
            (DocType.REPORT, self.report),
            (DocType.CORRESPONDENCE, self.correspondence),
        ]


class BaseSummaryLLMOutput(ConfidenceReported):
    headline: str = Field(description="One sentence, under 120 characters.")
    bullets: list[str] | None = Field(
        default_factory=list, description="3-6 factual bullets grounded in the document."
    )
    unresolved_questions: list[str] | None = Field(
        default_factory=list,
        description="Facts a reader would expect but the document does not state. "
        "Put a fact here instead of inventing it.",
    )


class ContractSummaryLLMOutput(BaseSummaryLLMOutput):
    parties: list[str] | None = Field(default_factory=list)
    effective_date: str | None = Field(default=None, description="Null if not stated.")
    term: str | None = None
    governing_law: str | None = None
    key_obligations: list[str] | None = Field(default_factory=list)
    termination: str | None = None


class InvoiceSummaryLLMOutput(BaseSummaryLLMOutput):
    vendor: str | None = None
    bill_to: str | None = None
    invoice_number: str | None = None
    issue_date: str | None = None
    due_date: str | None = None
    total_stated: str | None = Field(
        default=None, description="The document's own total, verbatim. Null if absent."
    )
    currencies_seen: list[str] | None = Field(default_factory=list)


class ReportSummaryLLMOutput(BaseSummaryLLMOutput):
    subject: str | None = None
    period_covered: str | None = None
    key_findings: list[str] | None = Field(default_factory=list)
    recommendations: list[str] | None = Field(default_factory=list)


class CorrespondenceSummaryLLMOutput(BaseSummaryLLMOutput):
    sender: str | None = None
    recipient: str | None = None
    sent_date: str | None = None
    request: str | None = Field(default=None, description="What the sender wants.")
    action_items: list[str] | None = Field(default_factory=list)


class GenericSummaryLLMOutput(BaseSummaryLLMOutput):
    """Fallback when classification could not commit to a type."""

    subject: str | None = None


SUMMARY_SCHEMAS: dict[DocType, type[BaseSummaryLLMOutput]] = {
    DocType.CONTRACT: ContractSummaryLLMOutput,
    DocType.INVOICE: InvoiceSummaryLLMOutput,
    DocType.REPORT: ReportSummaryLLMOutput,
    DocType.CORRESPONDENCE: CorrespondenceSummaryLLMOutput,
    DocType.UNKNOWN: GenericSummaryLLMOutput,
}
