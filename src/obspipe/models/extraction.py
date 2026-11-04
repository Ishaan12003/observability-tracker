from __future__ import annotations

from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from obspipe.models.common import StepArtifact


class DateRole(str, Enum):
    EFFECTIVE = "effective"
    EXPIRY = "expiry"
    SIGNED = "signed"
    ISSUED = "issued"
    DUE = "due"
    MENTIONED = "mentioned"
    UNKNOWN = "unknown"


class AmountKind(str, Enum):
    LINE_ITEM = "line_item"
    SUBTOTAL = "subtotal"
    TAX = "tax"
    TOTAL = "total"
    PENALTY = "penalty"
    ADVANCE = "advance"
    """Already paid against the total - not a component of it."""
    BALANCE = "balance"
    """Total minus advance - derived, not a component either."""
    UNKNOWN = "unknown"


class Party(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: str | None = Field(default=None, description="e.g. buyer, vendor, licensor.")
    organization: bool = True


class DateEntity(BaseModel):
    """A date kept in both raw and parsed form.

    ``iso_date`` stays ``None`` when the raw text could not be resolved. Losing
    that distinction is how pipelines silently invent dates.
    """

    model_config = ConfigDict(extra="forbid")

    raw_text: str
    iso_date: str | None = None
    role: DateRole = DateRole.UNKNOWN


class Amount(BaseModel):
    """A monetary amount, raw text preserved.

    ``value``/``currency`` are filled by deterministic parsing, never by the
    LLM. Mixed-currency documents show up as multiple distinct currencies here
    instead of being silently summed.
    """

    model_config = ConfigDict(extra="forbid")

    raw_text: str
    value: Decimal | None = None
    currency: str | None = Field(default=None, description="ISO-4217 code when resolvable.")
    kind: AmountKind = AmountKind.UNKNOWN
    label: str | None = None


class Identifier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(description="e.g. invoice_number, po_number, contract_ref.")
    value: str


class Entities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parties: list[Party] = Field(default_factory=list)
    dates: list[DateEntity] = Field(default_factory=list)
    amounts: list[Amount] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)
    identifiers: list[Identifier] = Field(default_factory=list)

    def currencies(self) -> set[str]:
        return {a.currency for a in self.amounts if a.currency}

    def total_amount(self) -> Amount | None:
        for amount in self.amounts:
            if amount.kind is AmountKind.TOTAL:
                return amount
        return None

    def is_empty(self) -> bool:
        return not (self.parties or self.dates or self.amounts or self.key_terms or self.identifiers)


class ExtractionOutput(StepArtifact):
    """Step 2 output."""

    entities: Entities = Field(default_factory=Entities)
    self_confidence: int = Field(
        default=3, ge=1, le=5, description="The extractor's own 1-5 rating of this result."
    )
    confidence_reason: str = ""
    model_name: str | None = None
