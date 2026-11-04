"""Prompt templates, one per LLM-backed step.

Every template takes ``document`` plus whatever upstream artifacts that step is
entitled to see. Nothing reads global state, so a prompt can be replayed from a
stored trace verbatim.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from obspipe.models.document import DocType

_GROUNDING_RULE = (
    "Copy spans verbatim from the document. If a fact is not present, omit it or use null. "
    "Never infer, complete, or invent a value that the document does not state."
)

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured entities from business documents.\n"
            f"{_GROUNDING_RULE}\n"
            "Amounts means sums of money only. Copy the currency symbol or code together "
            "with the digits. Never record a measurement, vital sign, lab value, percentage, "
            "count or duration as an amount - if a document contains no money, return no "
            "amounts. Do not convert, round, or total anything - another system does the "
            "arithmetic. "
            "For dates, only fill iso_date when day, month and year are all unambiguous."
            "\nRate your own answer 1-5 in self_confidence and say in confidence_reason what most limits it. A low rating on a hard document is more useful than a confident wrong one.",
        ),
        (
            "human",
            "Document ({content_type}, {word_count} words), source '{source_name}':\n"
            "<document>\n{document}\n</document>\n\n"
            "Extract the entities.",
        ),
    ]
)

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You classify a business document into exactly these four types:\n"
            "- contract: creates mutual obligations between parties (agreement, MSA, NDA, lease)\n"
            "- invoice: demands payment for goods or services rendered\n"
            "- report: analyses a subject over a period, with findings\n"
            "- correspondence: a message from a sender to a recipient (letter, email, memo)\n\n"
            "Score each type independently from 0.0 to 1.0 for how well it fits. "
            "Scores need not sum to 1. If the document genuinely sits between two types, "
            "give both a high score - do not artificially separate them. "
            "A forced confident answer on an ambiguous document is worse than an honest tie."
            "\nRate your own answer 1-5 in self_confidence and say in confidence_reason what most limits it. A low rating on a hard document is more useful than a confident wrong one.",
        ),
        (
            "human",
            "Document ({word_count} words), source '{source_name}':\n"
            "<document>\n{document}\n</document>\n\n"
            "Entities already extracted:\n{entity_digest}\n\n"
            "Score each candidate type and justify briefly.",
        ),
    ]
)

_SUMMARY_SYSTEM = (
    "You write structured summaries of {doc_type} documents for a business reader.\n"
    f"{_GROUNDING_RULE}\n"
    "{type_guidance}\n"
    "Always fill bullets with 3-6 sentences summarising the document as a whole, even "
    "when the type-specific fields below repeat some of the same facts - bullets is the "
    "top-level read, the type-specific fields are the structured detail. "
    "Anything a reader would expect but the document omits goes in unresolved_questions. "
    "Leaving a field null and naming the gap is always correct; filling it by inference is not."
    "\nRate your own answer 1-5 in self_confidence and say in confidence_reason what most limits it. A low rating on a hard document is more useful than a confident wrong one."
)

_TYPE_GUIDANCE: dict[DocType, str] = {
    DocType.CONTRACT: (
        "Lead with who is bound to what. Cover parties, effective date, term, governing law, "
        "key obligations and termination. If no effective date is stated, say so explicitly - "
        "an unstated commencement date is a material gap, not a detail to smooth over."
    ),
    DocType.INVOICE: (
        "Lead with who owes whom how much by when. Quote the document's own stated total "
        "verbatim; do not compute one. List every distinct currency that appears. If line "
        "items use more than one currency, do not present a single combined figure."
    ),
    DocType.REPORT: (
        "Lead with the subject and period. Separate findings (what the data shows) from "
        "recommendations (what the author proposes)."
    ),
    DocType.CORRESPONDENCE: (
        "Lead with sender, recipient and the ask. Extract concrete action items with owners "
        "and deadlines where stated."
    ),
    DocType.UNKNOWN: (
        "The document type could not be determined. Summarise conservatively, describe what "
        "the document appears to be, and list the competing interpretations in "
        "unresolved_questions."
    ),
}


def summary_prompt(doc_type: DocType) -> ChatPromptTemplate:
    """Build the step-4 template for one document type.

    The template is selected by step 3's verdict, so a misclassification shows
    up as a structurally wrong summary - the propagation the tracer must be
    able to walk back.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                _SUMMARY_SYSTEM.format(
                    doc_type=doc_type.value,
                    type_guidance=_TYPE_GUIDANCE[doc_type],
                ),
            ),
            (
                "human",
                "Document, source '{source_name}':\n"
                "<document>\n{document}\n</document>\n\n"
                "Extracted entities:\n{entity_digest}\n\n"
                "Classification: {doc_type} (confidence {confidence:.2f}, "
                "runner-up {runner_up}, margin {margin:.2f})\n"
                "{ambiguity_note}\n\n"
                "Write the summary.",
            ),
        ]
    )
