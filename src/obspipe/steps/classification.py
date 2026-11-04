"""Step 3 - Classification.

The model scores all four candidate types; the winner, the margin and the
ambiguity verdict are computed here. Keeping the arithmetic out of the prompt
means "the model was torn between contract and correspondence" survives as
evidence instead of collapsing into a single confident-looking label - which
is the thing that later explains a structurally wrong summary.
"""

from __future__ import annotations

from obspipe.digest import entity_digest
from obspipe.llm.backends import LLMBackend
from obspipe.llm.prompts import CLASSIFICATION_PROMPT
from obspipe.llm.schemas import ClassificationLLMOutput
from obspipe.models.classification import (
    AMBIGUITY_MARGIN,
    LOW_CONFIDENCE,
    ClassificationOutput,
    TypeScore,
)
from obspipe.models.common import StepName
from obspipe.models.document import DocType
from obspipe.models.extraction import ExtractionOutput
from obspipe.models.intake import IntakeOutput
from obspipe.models.pipeline import RunContext
from obspipe.steps.base import StepFailure, error, info, warn
from obspipe.tracing.tracer import annotate, traced

UNKNOWN_FLOOR = 0.2
"""Top score below this means no candidate type actually fits."""

_EXPECTED_EVIDENCE: dict[DocType, tuple[str, ...]] = {
    DocType.INVOICE: ("amounts",),
    DocType.CONTRACT: ("parties",),
    DocType.CORRESPONDENCE: ("parties",),
}


@traced(StepName.CLASSIFICATION)
def run_classification(
    intake: IntakeOutput,
    extraction: ExtractionOutput,
    ctx: RunContext,
    backend: LLMBackend,
) -> ClassificationOutput:
    """Assign a document type, preserving how close the call was."""
    try:
        chain = CLASSIFICATION_PROMPT | backend.structured(ClassificationLLMOutput)
        raw: ClassificationLLMOutput = chain.invoke(
            {
                "document": intake.normalized_text,
                "word_count": intake.word_count,
                "source_name": intake.source_name,
                "entity_digest": entity_digest(extraction.entities),
            }
        )
    except Exception as exc:  # noqa: BLE001
        raise StepFailure(
            StepName.CLASSIFICATION,
            "classification_call_failed",
            f"{type(exc).__name__}: {exc}",
        ) from exc

    pairs = raw.as_pairs()
    scores = [TypeScore(doc_type=doc_type, score=score) for doc_type, score in pairs]
    ranked = sorted(pairs, key=lambda p: p[1], reverse=True)
    top_type, top_score = ranked[0]
    runner_up_type, runner_up_score = ranked[1]
    margin = round(top_score - runner_up_score, 4)

    warnings: list = []
    chosen = top_type

    if top_score < UNKNOWN_FLOOR:
        chosen = DocType.UNKNOWN
        warnings.append(
            error(
                "no_type_fits",
                f"Best candidate '{top_type.value}' scored only {top_score:.2f}; "
                "falling back to 'unknown' rather than forcing a label.",
                field_path="doc_type",
            )
        )
    elif margin < AMBIGUITY_MARGIN:
        warnings.append(
            warn(
                "ambiguous_classification",
                f"'{top_type.value}' ({top_score:.2f}) and '{runner_up_type.value}' "
                f"({runner_up_score:.2f}) are {margin:.2f} apart. The summary shape below "
                "is effectively a coin flip.",
                field_path="doc_type",
            )
        )

    if top_score < LOW_CONFIDENCE and chosen is not DocType.UNKNOWN:
        warnings.append(
            warn(
                "low_classification_confidence",
                f"Top score {top_score:.2f} is below {LOW_CONFIDENCE}.",
                field_path="doc_type",
            )
        )

    for required in _EXPECTED_EVIDENCE.get(chosen, ()):
        if not getattr(extraction.entities, required):
            warnings.append(
                warn(
                    "classification_evidence_missing",
                    f"Classified as '{chosen.value}' but step 2 found no {required}. "
                    "Either the label or the extraction is wrong.",
                    field_path="doc_type",
                )
            )

    if intake.truncated:
        warnings.append(
            info(
                "classified_from_truncated_text",
                "Classification saw only the truncated head of the document.",
            )
        )
    if extraction.has_errors:
        warnings.append(
            info(
                "classified_on_degraded_entities",
                "Upstream extraction reported errors; this label inherits that risk.",
            )
        )

    # The gap to the runner-up is the honest measure of how decided this was.
    annotate(
        computed_confidence=margin,
        top_score=top_score,
        runner_up=runner_up_type.value,
        margin=margin,
    )

    return ClassificationOutput(
        doc_id=intake.doc_id,
        doc_type=chosen,
        scores=scores,
        rationale=raw.rationale,
        self_confidence=raw.self_confidence,
        confidence_reason=raw.confidence_reason,
        model_name=getattr(backend, "model", None),
        warnings=warnings,
    )
