"""Step 2 - Extraction.

The model is asked only for spans it can see. Everything numeric or temporal
is resolved afterwards in Python, and every returned span is checked back
against the source text. Two distinct failure families come out of that:

* **hallucination** - the model returned a span the document does not contain;
* **recall miss** - the document plainly contains an amount the model skipped.

Both are recorded as warnings rather than exceptions, because a partial
extraction is still worth tracing.
"""

from __future__ import annotations

from decimal import Decimal

from obspipe.llm.backends import LLMBackend
from obspipe.llm.prompts import EXTRACTION_PROMPT
from obspipe.llm.schemas import ExtractionLLMOutput
from obspipe.models.common import StepName
from obspipe.models.extraction import (
    Amount,
    AmountKind,
    DateEntity,
    DateRole,
    Entities,
    ExtractionOutput,
    Identifier,
    Party,
)
from obspipe.models.intake import IntakeOutput
from obspipe.models.pipeline import RunContext
from obspipe.parsing import (
    MONEY_SCAN_RE,
    appears_in,
    is_valid_iso_date,
    looks_non_monetary,
    numeric_span,
    parse_date,
    parse_money,
)
from obspipe.steps.base import GUARD_DROP_UNGROUNDED, StepFailure, error, guard_enabled, info, warn
from obspipe.tracing.tracer import annotate, traced

LOW_CONFIDENCE = 2
"""At or below this on the 1-5 scale, the extractor is telling you it guessed."""
RECALL_MISS_TOLERANCE = 2
"""How many document-visible amounts may be skipped before it counts as a miss."""


_DATE_ROLE_SYNONYMS: dict[str, DateRole] = {
    "commencement": DateRole.EFFECTIVE,
    "start": DateRole.EFFECTIVE,
    "start_date": DateRole.EFFECTIVE,
    "termination": DateRole.EXPIRY,
    "end": DateRole.EXPIRY,
    "end_date": DateRole.EXPIRY,
    "renewal": DateRole.EXPIRY,
    "execution": DateRole.SIGNED,
    "executed": DateRole.SIGNED,
    "invoice_date": DateRole.ISSUED,
    "created": DateRole.ISSUED,
    "sent": DateRole.ISSUED,
    "deadline": DateRole.DUE,
    "payment": DateRole.DUE,
    "payment_due": DateRole.DUE,
    "advance_payment": DateRole.DUE,
    "balance_due": DateRole.DUE,
}

_AMOUNT_KIND_SYNONYMS: dict[str, AmountKind] = {
    "grand_total": AmountKind.TOTAL,
    "amount_due": AmountKind.TOTAL,
    "net_total": AmountKind.TOTAL,
    "vat": AmountKind.TAX,
    "gst": AmountKind.TAX,
    "cgst": AmountKind.TAX,
    "sgst": AmountKind.TAX,
    "igst": AmountKind.TAX,
    "deposit": AmountKind.ADVANCE,
    "advance_paid": AmountKind.ADVANCE,
    "prepayment": AmountKind.ADVANCE,
    "balance_due": AmountKind.BALANCE,
    "balance_payable": AmountKind.BALANCE,
    "outstanding": AmountKind.BALANCE,
    "fee": AmountKind.LINE_ITEM,
    "charge": AmountKind.LINE_ITEM,
    "item": AmountKind.LINE_ITEM,
    "unit_price": AmountKind.LINE_ITEM,
    "shipping": AmountKind.LINE_ITEM,
    "late_fee": AmountKind.PENALTY,
    "interest": AmountKind.PENALTY,
}


def _coerce(raw_label: str | None, enum, synonyms, fallback, field: str, warnings: list):
    """Map a free-text label from the model onto a strict internal enum.

    The wire schema deliberately does not constrain these labels. A fixed enum
    on the wire means a model that meets a date the enum never anticipated -
    an event date, a cancellation deadline - has no legal value to emit, and
    the provider rejects the entire tool call. Better to accept the label and
    resolve it here, recording anything unrecognised rather than discarding it
    silently.
    """
    label = (raw_label or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not label:
        return fallback
    try:
        return enum(label)
    except ValueError:
        pass
    if label in synonyms:
        return synonyms[label]
    warnings.append(
        info(
            f"unrecognised_{field}",
            f"Model returned {field} '{raw_label}', which maps to no known value; "
            f"recorded as '{fallback.value}'.",
            field_path=f"entities.{field}s",
        )
    )
    return fallback


def _build_amount(raw, source_text: str, warnings: list) -> Amount | None:
    """Build one amount, or return ``None`` if the span is not money at all.

    A model handed a document with no money in it will still put *something* in
    the amounts field - a temperature, a blood pressure, a lab value. Those are
    discarded here rather than dragged through currency checks that cannot
    apply to them, which is what otherwise fills a clinical report's trace with
    eight identical 'currency unresolved' warnings and a phantom missing total.
    """
    value, currency = parse_money(raw.raw_text)

    if currency is None and looks_non_monetary(raw.raw_text):
        return None

    if not appears_in(raw.raw_text, source_text):
        number = numeric_span(raw.raw_text)
        if number and appears_in(number, source_text):
            # The figure is in the document; only the currency marker was added,
            # which is what a model does when a table header carries the unit.
            warnings.append(
                info(
                    "currency_annotated_by_model",
                    f"'{raw.raw_text}' is not literally in the document, but '{number}' is - "
                    "the model resolved a currency stated elsewhere, such as a column header.",
                    field_path="entities.amounts",
                )
            )
        else:
            warnings.append(
                error(
                    "hallucinated_amount_span",
                    f"Amount '{raw.raw_text}' does not appear in the document.",
                    field_path="entities.amounts",
                )
            )
    if value is None:
        warnings.append(
            warn(
                "unparseable_amount",
                f"Could not read a number out of '{raw.raw_text}'.",
                field_path="entities.amounts",
            )
        )
    if currency is None:
        warnings.append(
            warn(
                "currency_unresolved",
                f"No currency identifiable in '{raw.raw_text}'; downstream totals "
                "cannot be trusted.",
                field_path="entities.amounts",
            )
        )
    return Amount(
        raw_text=raw.raw_text,
        value=value,
        currency=currency,
        kind=_coerce(raw.kind, AmountKind, _AMOUNT_KIND_SYNONYMS, AmountKind.UNKNOWN,
                     "amount_kind", warnings),
        label=raw.label,
    )


def _build_date(raw, source_text: str, warnings: list) -> DateEntity:
    if not appears_in(raw.raw_text, source_text):
        warnings.append(
            error(
                "hallucinated_date_span",
                f"Date '{raw.raw_text}' does not appear in the document.",
                field_path="entities.dates",
            )
        )

    iso = raw.iso_date
    if iso and not is_valid_iso_date(iso):
        warnings.append(
            warn(
                "invalid_iso_date",
                f"Model returned '{iso}' for '{raw.raw_text}', which is not a valid date. Dropped.",
                field_path="entities.dates",
            )
        )
        iso = None

    independent = parse_date(raw.raw_text)
    if iso and independent and iso != independent:
        warnings.append(
            warn(
                "date_normalization_mismatch",
                f"'{raw.raw_text}' normalized to {independent} locally but {iso} by the model.",
                field_path="entities.dates",
            )
        )
    if iso is None and independent:
        iso = independent
    if iso is None:
        warnings.append(
            info(
                "date_unresolved",
                f"'{raw.raw_text}' could not be resolved to a calendar date.",
                field_path="entities.dates",
            )
        )
    return DateEntity(
        raw_text=raw.raw_text,
        iso_date=iso,
        role=_coerce(raw.role, DateRole, _DATE_ROLE_SYNONYMS, DateRole.MENTIONED,
                     "date_role", warnings),
    )


def _check_totals(entities: Entities, warnings: list) -> None:
    currencies = entities.currencies()
    if len(currencies) > 1:
        warnings.append(
            warn(
                "mixed_currencies",
                f"Amounts span {len(currencies)} currencies ({', '.join(sorted(currencies))}); "
                "they must not be combined into one figure.",
                field_path="entities.amounts",
            )
        )

    total = entities.total_amount()
    if total is None:
        if any(a.kind is AmountKind.LINE_ITEM for a in entities.amounts):
            warnings.append(
                warn(
                    "no_total_found",
                    "Line items are present but the document states no total.",
                    field_path="entities.amounts",
                )
            )
        return

    if total.value is None or total.currency is None:
        return

    same_currency = [
        a
        for a in entities.amounts
        if a is not total and a.currency == total.currency and a.value is not None
    ]
    # advance and balance are derived from the total, never components of it
    same_currency = [
        a for a in same_currency if a.kind not in (AmountKind.ADVANCE, AmountKind.BALANCE)
    ]
    unknown_kind = [a for a in same_currency if a.kind is AmountKind.UNKNOWN]
    if unknown_kind:
        # Reconciling against amounts whose role is unknown invents arithmetic the
        # document never asserted. Say so instead of raising a false mismatch.
        warnings.append(
            info(
                "total_reconciliation_skipped",
                f"{len(unknown_kind)} {total.currency} amount(s) have no identified role, "
                "so the stated total could not be checked.",
                field_path="entities.amounts",
            )
        )
        return

    line_items = [a for a in same_currency if a.kind is AmountKind.LINE_ITEM]
    taxes = [a for a in same_currency if a.kind is AmountKind.TAX]
    subtotals = [a for a in same_currency if a.kind is AmountKind.SUBTOTAL]
    # Line items and subtotals describe the same money; adding both double-counts.
    components = (line_items or subtotals) + taxes
    if not components:
        return

    summed = sum((a.value for a in components), Decimal("0"))
    if abs(summed - total.value) > Decimal("0.01"):
        warnings.append(
            warn(
                "total_mismatch",
                f"Stated total {total.value} {total.currency} does not equal the sum of "
                f"its {total.currency} components ({summed}).",
                field_path="entities.amounts",
            )
        )


def _amount_grounded(amount: Amount, source_text: str) -> bool:
    if appears_in(amount.raw_text, source_text):
        return True
    number = numeric_span(amount.raw_text)
    return bool(number and appears_in(number, source_text))


def _drop_ungrounded(amounts, dates, parties, source_text: str, warnings: list):
    """The grounding guard: an ungrounded span never leaves this step.

    Without it a hallucinated party is reported *and* forwarded, so step 4
    dutifully writes it into the summary. With it the span is discarded here
    and the error becomes an informational record of what was removed.
    """
    for index, warning in enumerate(warnings):
        if warning.code.startswith("hallucinated_"):
            warnings[index] = info(
                "ungrounded_span_dropped",
                f"{warning.message} Dropped by the grounding guard before reaching later steps.",
                field_path=warning.field_path,
            )
    return (
        [a for a in amounts if _amount_grounded(a, source_text)],
        [d for d in dates if appears_in(d.raw_text, source_text)],
        [p for p in parties if appears_in(p.name, source_text)],
    )


@traced(StepName.EXTRACTION)
def run_extraction(
    intake: IntakeOutput,
    ctx: RunContext,
    backend: LLMBackend,
) -> ExtractionOutput:
    """Extract entities from normalized text and validate them against it."""
    source_text = intake.normalized_text
    try:
        # Built inside the try: constructing the chain touches the backend, and a
        # backend that cannot be built is a step failure like any other.
        chain = EXTRACTION_PROMPT | backend.structured(ExtractionLLMOutput)
        raw: ExtractionLLMOutput = chain.invoke(
            {
                "document": source_text,
                "content_type": intake.content_type.value,
                "word_count": intake.word_count,
                "source_name": intake.source_name,
            }
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a step failure, not swallowed
        raise StepFailure(
            StepName.EXTRACTION,
            "extraction_call_failed",
            f"{type(exc).__name__}: {exc}",
        ) from exc

    warnings: list = []
    built = [_build_amount(a, source_text, warnings) for a in raw.amounts]
    amounts = [amount for amount in built if amount is not None]
    discarded = len(built) - len(amounts)
    if discarded:
        warnings.append(
            info(
                "non_monetary_amounts_discarded",
                f"{discarded} extracted 'amount(s)' were measurements rather than money "
                "(vitals, lab values, rates) and were dropped.",
                field_path="entities.amounts",
            )
        )
    dates = [_build_date(d, source_text, warnings) for d in raw.dates]
    parties = [
        Party(name=p.name, role=p.role, organization=p.organization)
        for p in raw.parties
        if p.name.strip()
    ]
    for party in parties:
        if not appears_in(party.name, source_text):
            warnings.append(
                error(
                    "hallucinated_party_span",
                    f"Party '{party.name}' does not appear in the document.",
                    field_path="entities.parties",
                )
            )

    if guard_enabled(ctx, GUARD_DROP_UNGROUNDED):
        amounts, dates, parties = _drop_ungrounded(amounts, dates, parties, source_text, warnings)

    entities = Entities(
        parties=parties,
        dates=dates,
        amounts=amounts,
        key_terms=[t.strip() for t in raw.key_terms if t.strip()],
        identifiers=[Identifier(kind=i.kind, value=i.value) for i in raw.identifiers],
    )

    if entities.is_empty():
        warnings.append(
            error(
                "no_entities_extracted",
                "The extractor returned nothing at all; every later step is now guessing.",
            )
        )
    if not entities.dates:
        warnings.append(
            warn("no_dates_found", "No dates were extracted.", field_path="entities.dates")
        )
    if not entities.parties:
        warnings.append(
            warn("no_parties_found", "No parties were extracted.", field_path="entities.parties")
        )

    visible_amounts = len(MONEY_SCAN_RE.findall(source_text))
    if visible_amounts - len(amounts) > RECALL_MISS_TOLERANCE:
        warnings.append(
            warn(
                "amount_recall_miss",
                f"The document shows {visible_amounts} currency-marked amounts but only "
                f"{len(amounts)} were extracted.",
                field_path="entities.amounts",
            )
        )

    _check_totals(entities, warnings)

    if raw.self_confidence <= LOW_CONFIDENCE:
        warnings.append(
            warn(
                "low_extraction_confidence",
                f"Extractor rated its own result {raw.self_confidence}/5"
                + (f": {raw.confidence_reason}" if raw.confidence_reason else "."),
            )
        )
    if intake.truncated:
        warnings.append(
            info(
                "extracted_from_truncated_text",
                "Extraction ran on truncated input; entities past the cut are missing by "
                "construction, not by model error.",
            )
        )

    # Share of extracted spans that were verified present in the source. This
    # is evidence, not opinion, so it is kept apart from self_confidence.
    span_count = len(entities.amounts) + len(entities.dates) + len(entities.parties)
    hallucinated = sum(1 for w in warnings if w.code.startswith("hallucinated_"))
    annotate(
        computed_confidence=1.0 if not span_count else 1.0 - hallucinated / span_count,
        entities=span_count,
        hallucinated_spans=hallucinated,
    )

    return ExtractionOutput(
        doc_id=intake.doc_id,
        entities=entities,
        self_confidence=raw.self_confidence,
        confidence_reason=raw.confidence_reason,
        model_name=getattr(backend, "model", None),
        warnings=warnings,
    )
