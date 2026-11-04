"""Failure categories, and how warning codes map onto them.

A warning code says what one check saw. A category says what kind of thing went
wrong, at a level a product team can count and prioritise: "hallucination" is
an investment decision, ``hallucinated_party_span`` is a log line.
"""

from __future__ import annotations

from enum import Enum

from obspipe.models.common import StepName


class FailureCategory(str, Enum):
    INVALID_INPUT = "invalid_input"
    INPUT_TRUNCATION = "input_truncation"
    INPUT_CORRUPTION = "input_corruption"
    HALLUCINATION = "hallucination"
    RECALL_MISS = "recall_miss"
    NORMALIZATION_ERROR = "normalization_error"
    ARITHMETIC_INCONSISTENCY = "arithmetic_inconsistency"
    MISCLASSIFICATION = "misclassification"
    CLASSIFICATION_UNCERTAINTY = "classification_uncertainty"
    CROSS_STEP_INCONSISTENCY = "cross_step_inconsistency"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    INVALID_AGGREGATION = "invalid_aggregation"
    MISSING_DISCLOSURE = "missing_disclosure"
    LOW_QUALITY_OUTPUT = "low_quality_output"
    LLM_CALL_FAILURE = "llm_call_failure"
    OTHER = "other"


CATEGORY_LABELS: dict[FailureCategory, str] = {
    FailureCategory.INVALID_INPUT: "Invalid input",
    FailureCategory.INPUT_TRUNCATION: "Input truncation",
    FailureCategory.INPUT_CORRUPTION: "Input corruption (OCR / encoding)",
    FailureCategory.HALLUCINATION: "Hallucinated entity",
    FailureCategory.RECALL_MISS: "Recall miss",
    FailureCategory.NORMALIZATION_ERROR: "Normalization error",
    FailureCategory.ARITHMETIC_INCONSISTENCY: "Arithmetic inconsistency",
    FailureCategory.MISCLASSIFICATION: "Misclassification",
    FailureCategory.CLASSIFICATION_UNCERTAINTY: "Ambiguous classification",
    FailureCategory.CROSS_STEP_INCONSISTENCY: "Cross-step inconsistency",
    FailureCategory.UNSUPPORTED_CLAIM: "Unsupported claim in output",
    FailureCategory.INVALID_AGGREGATION: "Invalid aggregation",
    FailureCategory.MISSING_DISCLOSURE: "Missing disclosure",
    FailureCategory.LOW_QUALITY_OUTPUT: "Low-quality output",
    FailureCategory.LLM_CALL_FAILURE: "LLM call failure",
    FailureCategory.OTHER: "Other",
}


CODE_CATEGORIES: dict[str, FailureCategory] = {
    # intake
    "empty_document": FailureCategory.INVALID_INPUT,
    "whitespace_only_document": FailureCategory.INVALID_INPUT,
    "binary_content": FailureCategory.INVALID_INPUT,
    "very_short_document": FailureCategory.INVALID_INPUT,
    "document_truncated": FailureCategory.INPUT_TRUNCATION,
    "high_noise_ratio": FailureCategory.INPUT_CORRUPTION,
    "replacement_characters_present": FailureCategory.INPUT_CORRUPTION,
    # extraction
    "hallucinated_party_span": FailureCategory.HALLUCINATION,
    "hallucinated_date_span": FailureCategory.HALLUCINATION,
    "hallucinated_amount_span": FailureCategory.HALLUCINATION,
    "amount_recall_miss": FailureCategory.RECALL_MISS,
    "no_entities_extracted": FailureCategory.RECALL_MISS,
    "invalid_iso_date": FailureCategory.NORMALIZATION_ERROR,
    "date_normalization_mismatch": FailureCategory.NORMALIZATION_ERROR,
    "unparseable_amount": FailureCategory.NORMALIZATION_ERROR,
    "currency_unresolved": FailureCategory.NORMALIZATION_ERROR,
    "total_mismatch": FailureCategory.ARITHMETIC_INCONSISTENCY,
    "extraction_call_failed": FailureCategory.LLM_CALL_FAILURE,
    # classification
    "no_type_fits": FailureCategory.CLASSIFICATION_UNCERTAINTY,
    "ambiguous_classification": FailureCategory.CLASSIFICATION_UNCERTAINTY,
    "low_classification_confidence": FailureCategory.CLASSIFICATION_UNCERTAINTY,
    "classification_evidence_missing": FailureCategory.CROSS_STEP_INCONSISTENCY,
    "classification_call_failed": FailureCategory.LLM_CALL_FAILURE,
    # summarization
    "unsupported_amount_in_summary": FailureCategory.UNSUPPORTED_CLAIM,
    "invented_effective_date": FailureCategory.UNSUPPORTED_CLAIM,
    "effective_date_not_in_entities": FailureCategory.UNSUPPORTED_CLAIM,
    "amount_currency_changed_in_summary": FailureCategory.UNSUPPORTED_CLAIM,
    "single_total_over_mixed_currencies": FailureCategory.INVALID_AGGREGATION,
    "missing_gap_disclosure": FailureCategory.MISSING_DISCLOSURE,
    "missing_total_not_disclosed": FailureCategory.MISSING_DISCLOSURE,
    "ambiguity_not_disclosed": FailureCategory.MISSING_DISCLOSURE,
    "currency_mix_not_reported": FailureCategory.MISSING_DISCLOSURE,
    "contract_without_parties": FailureCategory.LOW_QUALITY_OUTPUT,
    "thin_summary": FailureCategory.LOW_QUALITY_OUTPUT,
    "summarized_without_type": FailureCategory.LOW_QUALITY_OUTPUT,
    "summarization_call_failed": FailureCategory.LLM_CALL_FAILURE,
    "empty_summary": FailureCategory.LOW_QUALITY_OUTPUT,
}

ADVISORY_CODES = frozenset(
    {
        "mixed_currencies",
        "no_total_found",
        "no_dates_found",
        "no_parties_found",
    }
)
"""Warnings that describe the *document*, not a mistake by the step.

A contract that states no dates makes ``no_dates_found`` true and correct. These
never make a step the root cause on their own; if ground truth says the dates
were there, the divergence from ground truth is what convicts the step.
"""

SOFT_CODES = frozenset({"low_extraction_confidence", "low_classification_confidence"})
"""Self-reported doubt. Considered only when no harder evidence exists anywhere."""

CATEGORY_PRIORITY: tuple[FailureCategory, ...] = (
    FailureCategory.LLM_CALL_FAILURE,
    FailureCategory.INVALID_INPUT,
    FailureCategory.INPUT_TRUNCATION,
    FailureCategory.INPUT_CORRUPTION,
    FailureCategory.HALLUCINATION,
    FailureCategory.MISCLASSIFICATION,
    FailureCategory.UNSUPPORTED_CLAIM,
    FailureCategory.INVALID_AGGREGATION,
    FailureCategory.NORMALIZATION_ERROR,
    FailureCategory.RECALL_MISS,
    FailureCategory.ARITHMETIC_INCONSISTENCY,
    FailureCategory.CLASSIFICATION_UNCERTAINTY,
    FailureCategory.CROSS_STEP_INCONSISTENCY,
    FailureCategory.MISSING_DISCLOSURE,
    FailureCategory.LOW_QUALITY_OUTPUT,
    FailureCategory.OTHER,
)
"""When one step carries several kinds of problem, the one listed first names it."""

STEP_NUMBER: dict[StepName, int] = {
    StepName.INTAKE: 1,
    StepName.EXTRACTION: 2,
    StepName.CLASSIFICATION: 3,
    StepName.SUMMARIZATION: 4,
}


def step_label(step: StepName) -> str:
    """'Step 2 (Extraction)' - the form every narrative uses."""
    return f"Step {STEP_NUMBER[step]} ({step.value.capitalize()})"


def category_of(code: str) -> FailureCategory:
    return CODE_CATEGORIES.get(code, FailureCategory.OTHER)


def dominant_category(codes: list[str]) -> FailureCategory | None:
    present = {category_of(code) for code in codes}
    for category in CATEGORY_PRIORITY:
        if category in present:
            return category
    return None
