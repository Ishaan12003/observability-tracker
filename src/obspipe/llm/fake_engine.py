"""Deterministic stand-in for a real model.

Purpose is not to be good. It is to be *reproducible and steerable*: the same
document always yields the same output, and named faults can be switched on to
manufacture the exact failure the observability layer is supposed to catch.
Tests therefore run offline, free, and without flake.
"""

from __future__ import annotations

import re
from typing import Iterable

from obspipe.llm.schemas import (
    BaseSummaryLLMOutput,
    ClassificationLLMOutput,
    ContractSummaryLLMOutput,
    CorrespondenceSummaryLLMOutput,
    ExtractionLLMOutput,
    GenericSummaryLLMOutput,
    InvoiceSummaryLLMOutput,
    RawAmount,
    RawDate,
    RawIdentifier,
    RawParty,
    ReportSummaryLLMOutput,
)
from obspipe.models.extraction import AmountKind, DateRole
from obspipe.parsing import MONEY_SCAN_RE as _MONEY_RE, parse_date, parse_money

KNOWN_FAULTS = frozenset(
    {
        "hallucinate_date",   # invent an effective date that is not in the document
        "drop_amounts",       # return no amounts at all
        "flat_scores",        # refuse to differentiate document types
        "overconfident",      # force a near-1.0 score on the top type
        "invent_total",       # state a total the document never gives
        "empty_extraction",   # return nothing
        "hallucinate_party",  # invent a party ("John Smith") the document never names
        "swap_date_format",   # normalize dd/mm as mm/dd - the classic locale bug
        "misclassify",        # confidently pick the wrong document type
        "call_timeout",       # the classification call times out
    }
)

HALLUCINATED_PARTY = "John Smith"
HALLUCINATED_DATE = "January 1, 2024"

_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}"
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October"
    r"|November|December)\s+\d{4})\b"
)

_LABELLED_PARTY_RE = re.compile(
    r"^\s*(Vendor|Supplier|Bill\s*To|Sold\s*To|From|To|Client|Customer|Landlord|Tenant"
    r"|Licensor|Licensee|Buyer|Seller|Contractor|Sender|Recipient)\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_BETWEEN_RE = re.compile(
    r"between\s+(.{3,80}?)\s+\(.*?\)\s*and\s+(.{3,80}?)\s*\(", re.IGNORECASE | re.DOTALL
)

_ID_PATTERNS = (
    ("invoice_number", re.compile(r"Invoice\s*(?:No\.?|Number|#)\s*:?\s*([A-Za-z0-9\-/]+)", re.I)),
    ("po_number", re.compile(r"(?:PO|Purchase Order)\s*(?:No\.?|#)?\s*:?\s*([A-Za-z0-9\-/]+)", re.I)),
    ("contract_ref", re.compile(r"(?:Agreement|Contract)\s*(?:No\.?|Ref\.?|#)\s*:?\s*([A-Za-z0-9\-/]+)", re.I)),
    ("case_number", re.compile(r"Case\s*(?:No\.?|#)\s*:?\s*([A-Za-z0-9\-/]+)", re.I)),
)

_TYPE_KEYWORDS: dict[str, tuple[tuple[str, float], ...]] = {
    "contract": (
        ("agreement", 0.30), ("hereby agree", 0.25), ("shall", 0.12), ("party", 0.10),
        ("parties", 0.15), ("governing law", 0.25), ("term of this", 0.20),
        ("termination", 0.18), ("in witness whereof", 0.30), ("obligations", 0.15),
        ("confidential", 0.10), ("indemnif", 0.15),
    ),
    "invoice": (
        ("invoice", 0.35), ("bill to", 0.25), ("amount due", 0.30), ("payment terms", 0.22),
        ("subtotal", 0.25), ("qty", 0.15), ("unit price", 0.22), ("net 30", 0.20),
        ("remit", 0.18), ("line item", 0.15), ("tax", 0.08),
    ),
    "report": (
        ("report", 0.30), ("findings", 0.25), ("methodology", 0.25), ("analysis", 0.20),
        ("recommendation", 0.25), ("executive summary", 0.28), ("quarter", 0.15),
        ("appendix", 0.15), ("data shows", 0.18), ("kpi", 0.15),
    ),
    "correspondence": (
        ("dear", 0.30), ("sincerely", 0.28), ("regards", 0.22), ("subject:", 0.20),
        ("memorandum", 0.25), ("please find", 0.20), ("writing to", 0.25),
        ("follow up", 0.15), ("thank you", 0.12), ("attn", 0.15),
    ),
}

_TERM_VOCAB = (
    "payment terms", "governing law", "termination", "confidentiality", "indemnification",
    "liability", "warranty", "renewal", "late fee", "purchase order", "milestone",
    "deliverable", "scope of work", "net 30", "net 45", "vat", "gst", "sla",
    "force majeure", "arbitration", "notice period", "penalty", "discount",
)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, round(value, 3)))


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        key = item.strip()
        if key and key.casefold() not in {k.casefold() for k in seen}:
            seen[key] = None
    return list(seen)


def _label_for(line: str, amount_text: str) -> str | None:
    label = line.replace(amount_text, " ").strip(" \t|:-")
    label = re.sub(r"\s{2,}", " ", label)
    return label[:60] or None


def _amount_kind(context: str) -> AmountKind:
    lowered = context.casefold()
    if "total" in lowered and "subtotal" not in lowered:
        return AmountKind.TOTAL
    if "subtotal" in lowered:
        return AmountKind.SUBTOTAL
    if "tax" in lowered or "vat" in lowered or "gst" in lowered:
        return AmountKind.TAX
    if "penalt" in lowered or "late fee" in lowered:
        return AmountKind.PENALTY
    if re.search(r"\bqty\b|\bx\s*\d|\bunit\b|\bhours?\b", lowered):
        return AmountKind.LINE_ITEM
    return AmountKind.UNKNOWN


def _date_role(context: str) -> DateRole:
    lowered = context.casefold()
    for needle, role in (
        ("effective", DateRole.EFFECTIVE),
        ("commenc", DateRole.EFFECTIVE),
        ("expir", DateRole.EXPIRY),
        ("terminat", DateRole.EXPIRY),
        ("due", DateRole.DUE),
        ("payable by", DateRole.DUE),
        ("signed", DateRole.SIGNED),
        ("executed", DateRole.SIGNED),
        ("issue", DateRole.ISSUED),
        ("dated", DateRole.ISSUED),
        ("invoice date", DateRole.ISSUED),
    ):
        if needle in lowered:
            return role
    return DateRole.MENTIONED


def _line_of(text: str, span: tuple[int, int]) -> str:
    """The physical line a match sits on.

    Deliberately not a sliding window: in tabular documents a window bleeds
    into the next row and mislabels a line item as the total.
    """
    start = text.rfind("\n", 0, span[0]) + 1
    end = text.find("\n", span[1])
    return text[start : end if end != -1 else len(text)]


def _context_of(text: str, span: tuple[int, int]) -> str:
    """The match's own line, plus the preceding line for table headers."""
    line = _line_of(text, span)
    prev_start = text.rfind("\n", 0, max(0, text.rfind("\n", 0, span[0])))
    prev_line = text[prev_start : text.rfind("\n", 0, span[0])] if prev_start != -1 else ""
    return f"{prev_line}\n{line}" if len(line.strip()) < 25 else line


def fake_extraction(text: str, faults: frozenset[str]) -> ExtractionLLMOutput:
    if "empty_extraction" in faults:
        return ExtractionLLMOutput(
            self_confidence=1, confidence_reason="Nothing matched any extraction rule."
        )

    parties: list[RawParty] = []
    for match in _BETWEEN_RE.finditer(text):
        parties.append(RawParty(name=match.group(1).strip(" ,\n"), role="party"))
        parties.append(RawParty(name=match.group(2).strip(" ,\n"), role="party"))
    for match in _LABELLED_PARTY_RE.finditer(text):
        label, value = match.group(1).lower().replace(" ", "_"), match.group(2).strip()
        if value and len(value) < 90:
            parties.append(RawParty(name=value, role=label))

    dates: list[RawDate] = []
    for match in _DATE_RE.finditer(text):
        raw = match.group(0)
        dates.append(
            RawDate(
                raw_text=raw,
                iso_date=parse_date(raw),
                role=_date_role(_context_of(text, match.span())).value,
            )
        )

    amounts: list[RawAmount] = []
    if "drop_amounts" not in faults:
        for match in _MONEY_RE.finditer(text):
            context = _context_of(text, match.span())
            amounts.append(
                RawAmount(
                    raw_text=match.group(0).strip(),
                    label=_label_for(_line_of(text, match.span()), match.group(0)),
                    kind=_amount_kind(context).value,
                )
            )

    identifiers = [
        RawIdentifier(kind=kind, value=match.group(1))
        for kind, pattern in _ID_PATTERNS
        if (match := pattern.search(text))
    ]

    lowered = text.casefold()
    key_terms = [term for term in _TERM_VOCAB if term in lowered][:10]

    if "swap_date_format" in faults:
        dates = [_swapped(d) for d in dates]
    if "hallucinate_date" in faults:
        dates.insert(0, RawDate(raw_text=HALLUCINATED_DATE, iso_date="2024-01-01", role=DateRole.EFFECTIVE.value))
    if "hallucinate_party" in faults:
        parties.insert(0, RawParty(name=HALLUCINATED_PARTY, role="signatory", organization=False))

    seen_parties = _dedupe(p.name for p in parties)
    parties = [next(p for p in parties if p.name == name) for name in seen_parties][:8]

    filled = sum(bool(x) for x in (parties, dates, amounts, identifiers, key_terms))
    return ExtractionLLMOutput(
        parties=parties,
        dates=dates[:12],
        amounts=amounts[:25],
        key_terms=key_terms,
        identifiers=identifiers,
        self_confidence=max(1, min(5, filled + 1)),
        confidence_reason=f"{filled} of 5 entity families matched a rule.",
    )


def _swapped(date: RawDate) -> RawDate:
    """Read the month as the day and vice versa, when both could be either."""
    iso = date.iso_date
    if not iso or len(iso) != 10 or not _NUMERIC_DATE_RE.fullmatch(date.raw_text.strip()):
        return date
    year, month, day = iso.split("-")
    if int(day) > 12 or day == month:
        return date
    return RawDate(raw_text=date.raw_text, iso_date=f"{year}-{day}-{month}", role=date.role)


_NUMERIC_DATE_RE = re.compile(r"\d{1,2}[/.]\d{1,2}[/.]\d{4}")

_WRONG_TYPE = {"contract": "report", "invoice": "report", "report": "correspondence",
               "correspondence": "contract"}


def fake_classification(text: str, faults: frozenset[str]) -> ClassificationLLMOutput:
    if "call_timeout" in faults:
        raise TimeoutError("Request timed out after 60.0s (simulated upstream timeout).")
    lowered = text.casefold()
    scores: dict[str, float] = {}
    hits: dict[str, list[str]] = {}
    for doc_type, keywords in _TYPE_KEYWORDS.items():
        score = 0.0
        matched: list[str] = []
        for needle, weight in keywords:
            occurrences = lowered.count(needle)
            if occurrences:
                score += weight * min(occurrences, 3) / 2
                matched.append(needle)
        scores[doc_type] = _clip(score)
        hits[doc_type] = matched

    if "flat_scores" in faults:
        scores = {k: 0.5 for k in scores}
    if "overconfident" in faults:
        top = max(scores, key=lambda k: scores[k])
        scores = {k: (0.97 if k == top else 0.05) for k in scores}

    if "misclassify" in faults:
        # Confidently wrong: no margin problem, no low score - nothing for a
        # warning to catch. Only ground truth or a human can see this one.
        real = max(scores, key=lambda k: scores[k])
        wrong = _WRONG_TYPE[real]
        scores = {k: (0.88 if k == wrong else min(scores[k], 0.4)) for k in scores}
        hits[wrong] = hits[wrong] or ["overall structure"]

    top = max(scores, key=lambda k: scores[k])
    rationale = (
        f"Matched {top} cues: {', '.join(hits[top][:4]) or 'none'}."
        if not faults & {"flat_scores", "overconfident"}
        else "Injected fault: scores are synthetic."
    )
    top_score = max(scores.values())
    return ClassificationLLMOutput(
        rationale=rationale,
        self_confidence=max(1, min(5, int(round(top_score * 4)) + 1)),
        confidence_reason=f"Top keyword score {top_score:.2f}.",
        **scores,
    )


def _first(pattern: str, text: str, group: int = 1) -> str | None:
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(group).strip() if match else None


_DIGEST_DATE_RE = re.compile(r"(?:^|\], )(.+?) \[(\w+)")
_ROLE_SUFFIX_RE = re.compile(r"\s*\([^)]*\)$")


def _digest_line(prompt_text: str, label: str) -> str:
    for line in prompt_text.splitlines():
        if line.startswith(f"{label}: "):
            return line[len(label) + 2 :]
    return ""


def _digest_parties(prompt_text: str) -> list[str]:
    line = _digest_line(prompt_text, "Parties")
    return [_ROLE_SUFFIX_RE.sub("", p).strip() for p in line.split(", ") if p.strip()]


def _digest_effective_date(prompt_text: str) -> str | None:
    line = _digest_line(prompt_text, "Dates")
    for raw, role in _DIGEST_DATE_RE.findall(line):
        if role == "effective":
            return raw.strip()
    return None


def fake_summary(
    text: str,
    schema: type[BaseSummaryLLMOutput],
    faults: frozenset[str],
    prompt_text: str = "",
) -> BaseSummaryLLMOutput:
    """Summarize from the document - except where a fault says to trust step 2.

    Under ``hallucinate_date`` and ``hallucinate_party`` the summary takes those
    entities from the entity digest it was handed, the way a real model leans on
    upstream output. That is what makes an extraction error *propagate*, and
    what lets the grounding guard in step 2 cure it.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    headline = (lines[0] if lines else "Empty document")[:120]
    bullets = [line[:160] for line in lines[1:6]]
    if "hallucinate_party" in faults and (named := _digest_parties(prompt_text)):
        bullets.append(f"Parties involved: {', '.join(named[:4])}.")
    dates = [m.group(0) for m in _DATE_RE.finditer(text)]
    money = [m.group(0).strip() for m in _MONEY_RE.finditer(text)]
    unresolved: list[str] = []

    if schema is ContractSummaryLLMOutput:
        effective = None if not dates else dates[0]
        if "hallucinate_date" in faults:
            effective = _digest_effective_date(prompt_text) or effective
        if not effective:
            unresolved.append("No effective date is stated anywhere in the agreement.")
        parties = _dedupe(m.group(2) for m in _LABELLED_PARTY_RE.finditer(text))
        if "hallucinate_party" in faults:
            parties = _dedupe([*_digest_parties(prompt_text), *parties])
        return ContractSummaryLLMOutput(
            headline=headline,
            bullets=bullets,
            unresolved_questions=unresolved,
            parties=parties[:4],
            effective_date=effective,
            term=_first(r"term of (?:this agreement )?(?:is |shall be )?([^.\n]{3,60})", text),
            governing_law=_first(r"laws of ([^.\n,]{3,60})", text),
            key_obligations=bullets[:3],
            termination=_first(r"(terminat[^.\n]{3,120})", text),
        )

    if schema is InvoiceSummaryLLMOutput:
        total_match = re.search(
            r"^[ \t]*(?:grand\s+)?total[^\n]*?(" + _MONEY_RE.pattern + r")",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        total = total_match.group(1).strip() if total_match else None
        if "invent_total" in faults:
            total = "$9,999.00"
        elif total is None:
            unresolved.append("The invoice states line items but no total.")
        currencies = _dedupe(c for c in (parse_money(m)[1] for m in money) if c)
        return InvoiceSummaryLLMOutput(
            headline=headline,
            bullets=bullets,
            unresolved_questions=unresolved,
            vendor=_first(r"(?:Vendor|From|Supplier)\s*:\s*(.+)", text),
            bill_to=_first(r"Bill\s*To\s*:\s*(.+)", text),
            invoice_number=_first(r"Invoice\s*(?:No\.?|Number|#)\s*:?\s*([A-Za-z0-9\-/]+)", text),
            issue_date=dates[0] if dates else None,
            due_date=dates[1] if len(dates) > 1 else None,
            total_stated=total,
            currencies_seen=currencies[:6],
        )

    if schema is ReportSummaryLLMOutput:
        return ReportSummaryLLMOutput(
            headline=headline,
            bullets=bullets,
            unresolved_questions=unresolved,
            subject=headline,
            period_covered=_first(r"(Q[1-4]\s*(?:FY)?\s*\d{2,4}|\d{4})", text),
            key_findings=bullets[:3],
            recommendations=[line for line in lines if line.lower().startswith(("recommend", "we recommend"))][:3],
        )

    if schema is CorrespondenceSummaryLLMOutput:
        return CorrespondenceSummaryLLMOutput(
            headline=headline,
            bullets=bullets,
            unresolved_questions=unresolved,
            sender=_first(r"From\s*:\s*(.+)", text),
            recipient=_first(r"To\s*:\s*(.+)", text),
            sent_date=dates[0] if dates else None,
            request=_first(r"(?:please|request(?:ing)?|kindly)\s+([^.\n]{5,120})", text),
            action_items=[line for line in lines if line.lstrip().startswith(("-", "*", "1.", "2."))][:4],
        )

    return GenericSummaryLLMOutput(
        headline=headline,
        bullets=bullets,
        unresolved_questions=["Document type could not be determined."],
        subject=headline,
    )
