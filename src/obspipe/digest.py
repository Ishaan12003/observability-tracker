"""Compact renderings of intermediate artifacts for downstream prompts.

Steps 3 and 4 are given step 2's *output*, not a re-read of the document
alone. Passing a rendered digest rather than the object keeps the prompt
readable and, more importantly, makes the inter-step contract explicit: if the
digest is wrong, the failure belongs to step 2, and the stored trace shows the
exact text the later step was handed.
"""

from __future__ import annotations

from obspipe.models.classification import ClassificationOutput
from obspipe.models.extraction import Entities


def entity_digest(entities: Entities, limit: int = 8) -> str:
    if entities.is_empty():
        return "(no entities extracted)"

    lines: list[str] = []
    if entities.parties:
        rendered = ", ".join(
            f"{p.name}" + (f" ({p.role})" if p.role else "") for p in entities.parties[:limit]
        )
        lines.append(f"Parties: {rendered}")
    if entities.dates:
        rendered = ", ".join(
            f"{d.raw_text} [{d.role.value}"
            + (f" -> {d.iso_date}]" if d.iso_date else ", unparsed]")
            for d in entities.dates[:limit]
        )
        lines.append(f"Dates: {rendered}")
    if entities.amounts:
        rendered = ", ".join(
            f"{a.raw_text} [{a.kind.value}"
            + (f", {a.currency}" if a.currency else ", currency unknown")
            + "]"
            for a in entities.amounts[:limit]
        )
        lines.append(f"Amounts: {rendered}")
        currencies = entities.currencies()
        if len(currencies) > 1:
            lines.append(f"Distinct currencies present: {', '.join(sorted(currencies))}")
    if entities.identifiers:
        lines.append(
            "Identifiers: "
            + ", ".join(f"{i.kind}={i.value}" for i in entities.identifiers[:limit])
        )
    if entities.key_terms:
        lines.append("Key terms: " + ", ".join(entities.key_terms[:limit]))
    return "\n".join(lines)


def ambiguity_note(classification: ClassificationOutput) -> str:
    """Tell step 4 how much to trust step 3."""
    if classification.ambiguous:
        return (
            f"WARNING: this document sits between '{classification.doc_type.value}' and "
            f"'{classification.runner_up.value if classification.runner_up else 'unknown'}' "
            f"(margin {classification.margin:.2f}). Hedge accordingly and record the competing "
            "reading in unresolved_questions."
        )
    return "The classification is confident."
