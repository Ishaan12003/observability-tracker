"""Step 4 - Summarization.

The prompt and the output schema are both chosen by step 3's verdict, so this
step is where an upstream misclassification becomes visible as a structurally
wrong artifact. After generation, every date- and money-shaped span in the
summary is checked back against the source document: a summary that states a
figure the document never contained is the canonical "final output is garbage"
case, and it is attributed here only if the underlying entity was sound.
"""

from __future__ import annotations

from typing import Any

from obspipe.digest import ambiguity_note, entity_digest
from obspipe.llm.backends import LLMBackend
from obspipe.llm.prompts import summary_prompt
from obspipe.llm.schemas import SUMMARY_SCHEMAS, BaseSummaryLLMOutput
from obspipe.models.classification import ClassificationOutput
from obspipe.models.common import StepName
from obspipe.models.document import DocType
from obspipe.models.extraction import DateRole, ExtractionOutput
from obspipe.models.intake import IntakeOutput
from obspipe.models.pipeline import RunContext
from obspipe.models.summary import SummaryOutput
from obspipe.parsing import MONEY_SCAN_RE, appears_in, numeric_span, parse_money
from obspipe.steps.base import (
    GUARD_STRIP_UNSUPPORTED,
    StepFailure,
    error,
    guard_enabled,
    info,
    warn,
)
from obspipe.tracing.tracer import annotate, traced

HEADLINE_MAX = 120
MIN_BULLETS = 2

_BASE_FIELDS = frozenset({"headline", "bullets", "unresolved_questions"})


def _summary_text(raw: BaseSummaryLLMOutput) -> str:
    parts: list[str] = [raw.headline, *raw.bullets, *raw.unresolved_questions]
    for name, value in raw.model_dump().items():
        if name in _BASE_FIELDS:
            continue
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
    # Joined with a hard separator so a span check cannot match across two fields.
    return "\n|\n".join(p for p in parts if p)


def _check_grounding(
    raw: BaseSummaryLLMOutput,
    source_text: str,
    extraction: ExtractionOutput,
    warnings: list,
) -> None:
    """Every money figure the summary asserts must trace back to the document.

    Matching is on parsed value and currency, not on the literal span. A model
    that rewrites the German ``EUR 96.000,00`` as ``EUR 96,000.00`` has
    reformatted a real figure, not invented one; only a figure that matches
    nothing in the document is a fabrication.
    """
    known: set[tuple[object, object]] = {
        (amount.value, amount.currency)
        for amount in extraction.entities.amounts
        if amount.value is not None
    }

    seen: set[str] = set()
    for match in MONEY_SCAN_RE.finditer(_summary_text(raw)):
        span = match.group(0).strip()
        if span in seen:
            continue
        seen.add(span)
        if appears_in(span, source_text):
            continue

        value, currency = parse_money(span)
        if value is not None and (value, currency) in known:
            warnings.append(
                info(
                    "amount_reformatted_in_summary",
                    f"Summary writes '{span}' where the document uses a different notation "
                    "for the same figure.",
                    field_path="type_specific",
                )
            )
            continue
        if value is not None and any(value == known_value for known_value, _ in known):
            warnings.append(
                warn(
                    "amount_currency_changed_in_summary",
                    f"Summary states '{span}'; the document has that value under a different "
                    "currency.",
                    field_path="type_specific",
                )
            )
            continue

        number = numeric_span(span)
        if number and appears_in(number, source_text):
            # Same case extraction handles: the figure is in the document, the
            # model supplied a currency stated in a header or nearby line.
            warnings.append(
                info(
                    "currency_annotated_by_model",
                    f"Summary writes '{span}'; '{number}' is in the document and the currency "
                    "was resolved from elsewhere.",
                    field_path="type_specific",
                )
            )
            continue

        warnings.append(
            error(
                "unsupported_amount_in_summary",
                f"Summary states '{span}', which matches no amount in the document.",
                field_path="type_specific",
            )
        )


def _strip_unsupported(
    raw: BaseSummaryLLMOutput, warnings: list, extraction: ExtractionOutput
) -> BaseSummaryLLMOutput:
    """The summary guard: withhold any figure the document does not support.

    A field holding an unsupported figure is cleared, a bullet carrying one is
    removed, and the gap is disclosed as an unresolved question - a summary that
    says "total not stated" is correct, one that states an invented total is not.
    A stated total is the exception: when step 2 found a grounded total, that
    figure replaces the invented one.
    """
    unsupported = [
        w.message.split("'")[1]
        for w in warnings
        if w.code == "unsupported_amount_in_summary" and w.message.count("'") >= 2
    ]
    if not unsupported:
        return raw

    def tainted(value: str) -> bool:
        return any(span in value for span in unsupported)

    grounded_total = extraction.entities.total_amount()
    data = raw.model_dump()
    repaired: list[str] = []
    for name, value in list(data.items()):
        if name in ("headline", "self_confidence", "confidence_reason"):
            continue
        if isinstance(value, str) and tainted(value):
            if name == "total_stated" and grounded_total is not None:
                data[name] = grounded_total.raw_text
                repaired += [span for span in unsupported if span in value]
            else:
                data[name] = None
        elif isinstance(value, list) and name != "unresolved_questions":
            data[name] = [v for v in value if not (isinstance(v, str) and tainted(v))]
    if tainted(data.get("headline") or ""):
        for span in unsupported:
            data["headline"] = data["headline"].replace(span, "[figure withheld]")
    questions = list(data.get("unresolved_questions") or [])
    for span in unsupported:
        if span in repaired:
            continue
        questions.append(f"A figure of {span} was generated but is not supported by the document.")
    data["unresolved_questions"] = questions

    for index, warning in enumerate(warnings):
        if warning.code == "unsupported_amount_in_summary":
            replaced = any(f"'{span}'" in warning.message for span in repaired)
            warnings[index] = info(
                "unsupported_figure_withheld",
                f"{warning.message} "
                + (
                    f"Replaced by the summary guard with step 2's grounded total "
                    f"'{grounded_total.raw_text}'."
                    if replaced and grounded_total is not None
                    else "Withheld by the summary guard."
                ),
                field_path=warning.field_path,
            )
    return type(raw).model_validate(data)


def _check_contract(raw: Any, extraction: ExtractionOutput, warnings: list) -> None:
    stated_dates = extraction.entities.dates
    if raw.effective_date and not stated_dates:
        warnings.append(
            error(
                "invented_effective_date",
                f"Summary gives effective date '{raw.effective_date}' but step 2 found no "
                "dates at all in the document.",
                field_path="type_specific.effective_date",
            )
        )
    elif raw.effective_date and not any(
        appears_in(raw.effective_date, d.raw_text) or raw.effective_date == d.iso_date
        for d in stated_dates
    ):
        warnings.append(
            warn(
                "effective_date_not_in_entities",
                f"Effective date '{raw.effective_date}' matches none of the extracted dates.",
                field_path="type_specific.effective_date",
            )
        )
    if not stated_dates and not raw.unresolved_questions:
        warnings.append(
            warn(
                "missing_gap_disclosure",
                "The contract states no dates, yet the summary raises no unresolved question "
                "about it.",
                field_path="unresolved_questions",
            )
        )
    if not raw.parties and not extraction.entities.parties:
        warnings.append(
            warn(
                "contract_without_parties",
                "A contract summary with no parties is not usable.",
                field_path="type_specific.parties",
            )
        )


def _check_invoice(raw: Any, extraction: ExtractionOutput, warnings: list) -> None:
    currencies = extraction.entities.currencies()
    if len(currencies) > 1:
        if len(raw.currencies_seen) < len(currencies):
            warnings.append(
                warn(
                    "currency_mix_not_reported",
                    f"Document mixes {sorted(currencies)} but the summary lists "
                    f"{raw.currencies_seen or 'none'}.",
                    field_path="type_specific.currencies_seen",
                )
            )
        if raw.total_stated:
            warnings.append(
                warn(
                    "single_total_over_mixed_currencies",
                    f"A single total '{raw.total_stated}' is presented although amounts span "
                    f"{sorted(currencies)}.",
                    field_path="type_specific.total_stated",
                )
            )
    if raw.total_stated is None and not raw.unresolved_questions:
        warnings.append(
            warn(
                "missing_total_not_disclosed",
                "No total was reported and no unresolved question was raised about it.",
                field_path="unresolved_questions",
            )
        )
    if raw.due_date and not any(d.role is DateRole.DUE for d in extraction.entities.dates):
        warnings.append(
            info(
                "due_date_role_unconfirmed",
                f"Summary calls '{raw.due_date}' a due date; step 2 did not label any date "
                "with that role.",
                field_path="type_specific.due_date",
            )
        )


@traced(StepName.SUMMARIZATION)
def run_summarization(
    intake: IntakeOutput,
    extraction: ExtractionOutput,
    classification: ClassificationOutput,
    ctx: RunContext,
    backend: LLMBackend,
) -> SummaryOutput:
    """Produce a summary whose shape is dictated by the classified type."""
    doc_type = classification.doc_type
    schema = SUMMARY_SCHEMAS[doc_type]
    try:
        chain = summary_prompt(doc_type) | backend.structured(schema)
        raw: BaseSummaryLLMOutput = chain.invoke(
            {
                "document": intake.normalized_text,
                "source_name": intake.source_name,
                "entity_digest": entity_digest(extraction.entities),
                "doc_type": doc_type.value,
                "confidence": classification.confidence,
                "runner_up": classification.runner_up.value if classification.runner_up else "none",
                "margin": classification.margin,
                "ambiguity_note": ambiguity_note(classification),
            }
        )
    except Exception as exc:  # noqa: BLE001
        raise StepFailure(
            StepName.SUMMARIZATION,
            "summarization_call_failed",
            f"{type(exc).__name__}: {exc}",
        ) from exc

    warnings: list = []
    _check_grounding(raw, intake.normalized_text, extraction, warnings)
    if guard_enabled(ctx, GUARD_STRIP_UNSUPPORTED):
        raw = _strip_unsupported(raw, warnings, extraction)

    if doc_type is DocType.CONTRACT:
        _check_contract(raw, extraction, warnings)
    elif doc_type is DocType.INVOICE:
        _check_invoice(raw, extraction, warnings)

    if classification.ambiguous and not raw.unresolved_questions:
        warnings.append(
            warn(
                "ambiguity_not_disclosed",
                f"Step 3 was torn between '{doc_type.value}' and "
                f"'{classification.runner_up.value if classification.runner_up else 'unknown'}' "
                "but the summary presents one reading without qualification.",
                field_path="unresolved_questions",
            )
        )
    if doc_type is DocType.UNKNOWN:
        warnings.append(
            warn(
                "summarized_without_type",
                "Summary was generated from the generic template because no type was assigned.",
            )
        )
    type_list_items = sum(
        len(value)
        for name, value in raw.model_dump().items()
        if name not in _BASE_FIELDS and isinstance(value, list)
    )
    if len(raw.bullets) + type_list_items < MIN_BULLETS:
        warnings.append(
            warn(
                "thin_summary",
                f"Only {len(raw.bullets) + type_list_items} summary item(s) produced "
                "across bullets and type-specific fields.",
                field_path="bullets",
            )
        )
    elif not raw.bullets:
        # Content exists, it just all went into the type-specific lists. Worth
        # recording - the summary is usable - but not a defect in the output.
        warnings.append(
            info(
                "bullets_omitted",
                f"No bullets returned; all {type_list_items} summary items went into "
                "type-specific fields instead.",
                field_path="bullets",
            )
        )
    if len(raw.headline) > HEADLINE_MAX:
        warnings.append(
            info(
                "headline_too_long",
                f"Headline is {len(raw.headline)} chars (limit {HEADLINE_MAX}).",
                field_path="headline",
            )
        )
    if not raw.headline.strip():
        raise StepFailure(
            StepName.SUMMARIZATION, "empty_summary", "Summarizer returned an empty headline."
        )

    # Share of money figures in the summary that trace back to the document.
    figures = len({m.group(0).strip() for m in MONEY_SCAN_RE.finditer(_summary_text(raw))})
    unsupported = sum(1 for w in warnings if w.code == "unsupported_amount_in_summary")
    annotate(
        computed_confidence=1.0 if not figures else 1.0 - unsupported / figures,
        figures_checked=figures,
        unsupported_figures=unsupported,
    )

    type_specific = {
        name: value
        for name, value in raw.model_dump().items()
        if name not in _BASE_FIELDS
    }

    return SummaryOutput(
        doc_id=intake.doc_id,
        doc_type=doc_type,
        headline=raw.headline.strip(),
        bullets=[b.strip() for b in raw.bullets if b.strip()],
        type_specific=type_specific,
        unresolved_questions=[q.strip() for q in raw.unresolved_questions if q.strip()],
        self_confidence=raw.self_confidence,
        confidence_reason=raw.confidence_reason,
        model_name=getattr(backend, "model", None),
        warnings=warnings,
    )
