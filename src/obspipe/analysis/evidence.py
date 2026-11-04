"""Backward trace analysis: from a bad final output to the step that caused it.

The tracer records what each step saw and did. This module reads that record
backwards and produces a :class:`Diagnosis` - the root-cause step, a failure
category, and an *evidence chain*: one link per affected step, each carrying
the specific input/output pair that supports it, plus a plain-English narrative
a person can act on without opening the trace JSON.

Root-cause selection is deliberately simple and explainable: the earliest step
that either carries a hard problem or diverges from ground truth. Earlier steps
cannot be blamed for later ones, so the first deviation is the cause and
everything after it is propagation. Warnings that describe the document rather
than a mistake (``no_dates_found`` on a contract that has none), warnings that
say "this came from upstream", and the model's own self-doubt are excluded from
selection, so a victim is never convicted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from obspipe.analysis.golden import Divergence, GoldenRecord, compare, project
from obspipe.analysis.taxonomy import (
    ADVISORY_CODES,
    CATEGORY_LABELS,
    SOFT_CODES,
    FailureCategory,
    category_of,
    dominant_category,
    step_label,
)
from obspipe.llm.fake_engine import _DATE_RE as DATE_SCAN_RE
from obspipe.models.classification import AMBIGUITY_MARGIN
from obspipe.models.common import Severity, StepName, StepWarning
from obspipe.parsing import MONEY_SCAN_RE, appears_in
from obspipe.tracing.models import LOW_CONFIDENCE, Span, SpanStatus
from obspipe.tracing.suspicion import CASCADE_CODES
from obspipe.tracing.trace import Trace

EXCERPT = 360


class StepHealth(str, Enum):
    """What the trace view colours a node by."""

    HEALTHY = "healthy"
    LOW_CONFIDENCE = "low_confidence"
    ROOT_CAUSE = "root_cause"
    AFFECTED = "affected"
    SKIPPED = "skipped"


class EvidenceLink(BaseModel):
    """One step's part in the failure, with the data that proves it."""

    model_config = ConfigDict(extra="forbid")

    step: StepName
    role: str
    """``root_cause`` or ``propagation``."""
    finding: str
    warning_codes: list[str] = Field(default_factory=list)
    traced_values: list[str] = Field(default_factory=list)
    """Concrete bad values followed downstream from the root cause."""
    received: Any = None
    produced: Any = None
    expected: Any = None
    divergences: list[Divergence] = Field(default_factory=list)


class StepAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: StepName
    health: StepHealth
    status: SpanStatus
    suspicion: float = 0.0
    self_confidence: int | None = None
    computed_confidence: float | None = None
    problem_codes: list[str] = Field(default_factory=list)
    divergence_count: int = 0


class Diagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    doc_id: str
    verdict: str
    """``healthy`` or ``failure``."""
    root_cause_step: StepName | None = None
    category: FailureCategory | None = None
    category_label: str | None = None
    confidence: float = 0.0
    method: str = "heuristic"
    """``ground_truth`` when a golden record decided it, else ``heuristic``."""
    headline: str
    narrative: str
    chain: list[EvidenceLink] = Field(default_factory=list)
    steps: list[StepAssessment] = Field(default_factory=list)
    affected_steps: list[StepName] = Field(default_factory=list)
    computed_ms: float = 0.0


# -- per-step view -----------------------------------------------------------


@dataclass
class _View:
    span: Span
    step: StepName
    inputs: dict
    output: dict | None
    projection: dict
    expected: dict | None
    divergences: list[Divergence]
    problems: list[StepWarning] = field(default_factory=list)
    soft: list[StepWarning] = field(default_factory=list)
    symptoms: list[StepWarning] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.span.status is SpanStatus.FAILED

    @property
    def implicated(self) -> bool:
        return self.failed or bool(self.problems) or bool(self.divergences)

    def problem_codes(self) -> list[str]:
        codes = [w.code for w in self.problems]
        if self.failed and self.span.error_type and self.span.error_type not in codes:
            codes.insert(0, self.span.error_type)
        return codes


def _views(trace: Trace, golden: GoldenRecord | None) -> list[_View]:
    views: list[_View] = []
    for span in trace.step_spans():
        output = trace.resolve(span.output) if span.output is not None else None
        inputs = trace.resolve(span.inputs) if span.inputs else {}
        projection = project(span.step, output, span.warning_codes()) if output else {}
        expected = golden.for_step(span.step) if golden else None
        divergences = (
            compare(expected, projection)
            if expected and span.status is not SpanStatus.SKIPPED and output is not None
            else []
        )
        view = _View(span, span.step, inputs, output, projection, expected, divergences)
        for warning in span.warnings:
            if warning.code in CASCADE_CODES or warning.code in ADVISORY_CODES:
                view.symptoms.append(warning)
            elif warning.code in SOFT_CODES:
                view.soft.append(warning)
            elif warning.severity in (Severity.WARN, Severity.ERROR):
                view.problems.append(warning)
        views.append(view)
    return views


# -- helpers -------------------------------------------------------------------


def _clip(text: Any, limit: int = EXCERPT) -> Any:
    if isinstance(text, str) and len(text) > limit:
        return text[:limit] + f"... [{len(text) - limit} more chars]"
    return text


def _quote_list(values: list[str], noun: str) -> str:
    quoted = [f"'{v}'" for v in values]
    if len(quoted) == 1:
        return f"the {noun} {quoted[0]}"
    return f"the {noun}s {', '.join(quoted[:-1])} and {quoted[-1]}"


def _find_paths(value: Any, needle: str, path: str = "") -> list[tuple[str, str]]:
    """Every string leaf inside ``value`` that contains ``needle``."""
    hits: list[tuple[str, str]] = []
    if isinstance(value, str):
        if needle.casefold() in value.casefold():
            hits.append((path or "$", value))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in ("warnings", "doc_id", "confidence_reason", "model_name"):
                continue
            hits.extend(_find_paths(item, needle, f"{path}.{key}" if path else key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hits.extend(_find_paths(item, needle, f"{path}[{index}]"))
    return hits


def _source_text(views: list[_View]) -> str:
    intake = next((v for v in views if v.step is StepName.INTAKE), None)
    if intake and intake.output:
        return intake.output.get("normalized_text", "")
    return ""


def _original_text(views: list[_View]) -> str:
    intake = next((v for v in views if v.step is StepName.INTAKE), None)
    if intake:
        document = intake.inputs.get("document") or {}
        if isinstance(document, dict):
            return document.get("content", "")
    return ""


def _received(view: _View, views: list[_View]) -> dict:
    """The part of a step's input that bears on its correctness."""
    by_step = {v.step: v for v in views}
    if view.step is StepName.INTAKE:
        document = view.inputs.get("document") or {}
        content = document.get("content", "") if isinstance(document, dict) else ""
        return {
            "source_name": document.get("source_name") if isinstance(document, dict) else None,
            "original_chars": len(content),
            "content_head": _clip(content, 240),
        }
    if view.step is StepName.EXTRACTION:
        intake = by_step.get(StepName.INTAKE)
        text = intake.output.get("normalized_text", "") if intake and intake.output else ""
        return {"normalized_chars": len(text), "text_head": _clip(text, 240)}
    if view.step is StepName.CLASSIFICATION:
        extraction = by_step.get(StepName.EXTRACTION)
        return {"entities_from_step_2": extraction.projection if extraction else None}
    if view.step is StepName.SUMMARIZATION:
        extraction = by_step.get(StepName.EXTRACTION)
        classification = by_step.get(StepName.CLASSIFICATION)
        return {
            "doc_type_from_step_3": classification.projection.get("doc_type") if classification else None,
            "margin_from_step_3": classification.projection.get("margin") if classification else None,
            "entities_from_step_2": extraction.projection if extraction else None,
        }
    return {}


# -- category ------------------------------------------------------------------


def _categorize(view: _View) -> FailureCategory:
    if view.step is StepName.CLASSIFICATION and any(d.path == "doc_type" for d in view.divergences):
        return FailureCategory.MISCLASSIFICATION
    category = dominant_category(view.problem_codes())
    if category is not None and category is not FailureCategory.OTHER:
        return category
    kinds = {d.kind for d in view.divergences}
    if view.step is StepName.INTAKE:
        return FailureCategory.INPUT_TRUNCATION if view.projection.get("truncated") else FailureCategory.INPUT_CORRUPTION
    if view.step is StepName.EXTRACTION:
        if "unexpected" in kinds:
            return FailureCategory.HALLUCINATION
        if "missing" in kinds:
            return FailureCategory.RECALL_MISS
        return FailureCategory.NORMALIZATION_ERROR
    if view.step is StepName.CLASSIFICATION:
        return FailureCategory.CLASSIFICATION_UNCERTAINTY
    if view.step is StepName.SUMMARIZATION and view.divergences:
        return FailureCategory.UNSUPPORTED_CLAIM
    soft = dominant_category([w.code for w in view.soft])
    return soft or category or FailureCategory.OTHER


# -- root-cause findings -------------------------------------------------------


def _ungrounded_entities(view: _View, source: str) -> dict[str, list[str]]:
    entities = (view.output or {}).get("entities") or {}
    found: dict[str, list[str]] = {}
    for party in entities.get("parties") or []:
        if not appears_in(party.get("name", ""), source):
            found.setdefault("party", []).append(party["name"])
    for date in entities.get("dates") or []:
        if not appears_in(date.get("raw_text", ""), source):
            found.setdefault("date", []).append(date["raw_text"])
    for amount in entities.get("amounts") or []:
        if not appears_in(amount.get("raw_text", ""), source):
            found.setdefault("amount", []).append(amount["raw_text"])
    for warning in view.span.warnings:
        if warning.code.startswith("hallucinated_") and "'" in warning.message:
            noun = warning.code.removeprefix("hallucinated_").removesuffix("_span")
            value = warning.message.split("'")[1]
            if value not in found.get(noun, []):
                found.setdefault(noun, []).append(value)
    return found


def _root_finding(
    view: _View, category: FailureCategory, views: list[_View]
) -> tuple[str, Any, list[str]]:
    """Return (finding sentence, produced-evidence, values to trace downstream)."""
    label = step_label(view.step)
    output = view.output or {}
    messages = {w.code: w.message for w in view.span.warnings}

    if view.failed:
        reason = view.span.error_message or "no message"
        calls = [c for c in view.span.llm_calls if c.parsing_error]
        produced = {
            "error_type": view.span.error_type,
            "error_message": _clip(reason),
            "llm_call_error": calls[0].parsing_error if calls else None,
        }
        if category is FailureCategory.LLM_CALL_FAILURE:
            return (f"{label} failed because its model call errored: {reason}", produced, [])
        return (f"{label} rejected the document: {reason}", produced, [])

    if category is FailureCategory.INPUT_TRUNCATION:
        original = _original_text(views)
        kept = output.get("normalized_text", "")
        lost_dates = [m.group(0) for m in DATE_SCAN_RE.finditer(original) if not appears_in(m.group(0), kept)]
        lost_money = [m.group(0).strip() for m in MONEY_SCAN_RE.finditer(original) if not appears_in(m.group(0), kept)]
        sentence = (
            f"{label} truncated the document from {len(original):,} to "
            f"{output.get('char_count', len(kept)):,} characters."
        )
        if lost_dates or lost_money:
            parts = []
            if lost_dates:
                parts.append(f"{len(lost_dates)} date(s)")
            if lost_money:
                parts.append(f"{len(lost_money)} amount(s)")
            example = (lost_dates or lost_money)[0]
            sentence += (
                f" {' and '.join(parts)} lie past the cut and could never be extracted, "
                f"e.g. '{example}'."
            )
        produced = {
            "truncated": True,
            "char_count": output.get("char_count"),
            "lost_dates": lost_dates[:6],
            "lost_amounts": lost_money[:6],
            "text_tail": _clip(kept[-200:], 200),
        }
        return sentence, produced, []

    if category is FailureCategory.INPUT_CORRUPTION:
        message = messages.get("high_noise_ratio") or messages.get("replacement_characters_present", "")
        return (
            f"{label} passed on damaged text: {message}",
            {"normalized_text_head": _clip(output.get("normalized_text", ""), 240)},
            [],
        )

    if category is FailureCategory.INVALID_INPUT:
        message = messages.get("very_short_document", "the document is unusable")
        return f"{label} received an unusable document: {message}", view.projection, []

    if category is FailureCategory.HALLUCINATION:
        source = _source_text(views)
        ungrounded = _ungrounded_entities(view, source)
        values = [v for items in ungrounded.values() for v in items]
        if not values:
            values = [str(d.actual) for d in view.divergences if d.kind == "unexpected"]
        phrases = [_quote_list(items, noun) for noun, items in ungrounded.items()]
        what = " and ".join(phrases) if phrases else "entities"
        verb = "does" if len(values) == 1 else "do"
        sentence = f"{label} hallucinated {what}, which {verb} not appear in the source document."
        produced = {
            "ungrounded": ungrounded,
            "occurrences_in_source": {v: source.casefold().count(v.casefold()) for v in values},
        }
        return sentence, produced, values

    if category is FailureCategory.RECALL_MISS:
        missing = [str(d.expected) for d in view.divergences if d.kind == "missing"]
        source = _source_text(views)
        visible = [m.group(0).strip() for m in MONEY_SCAN_RE.finditer(source)]
        extracted = view.projection.get("amounts") or []
        detail = messages.get("amount_recall_miss") or messages.get("no_entities_extracted")
        sentence = f"{label} missed entities the document plainly contains"
        if detail:
            sentence += f": {detail}"
        elif missing:
            sentence += f": {', '.join(missing[:4])} were never extracted."
        else:
            sentence += "."
        produced = {"visible_in_source": visible[:8], "extracted_amounts": extracted, "missing": missing[:8]}
        return sentence, produced, []

    if category is FailureCategory.MISCLASSIFICATION:
        scores = view.projection.get("scores") or {}
        got = view.projection.get("doc_type")
        want = next((d.expected for d in view.divergences if d.path == "doc_type"), None)
        want_label = " or ".join(want) if isinstance(want, list) else want
        right_score = scores.get(want if isinstance(want, str) else (want or [None])[0])
        sentence = (
            f"{label} labelled the document '{got}' ({scores.get(got, 0):.2f}) but it is "
            f"{'an' if str(want_label)[:1] in 'aeiou' else 'a'} '{want_label}'"
        )
        sentence += f"; the correct type scored only {right_score:.2f}." if right_score is not None else "."
        sentence += " No check fired: the label was confidently wrong."
        return sentence, {"doc_type": got, "scores": scores, "margin": view.projection.get("margin")}, [got] if got else []

    if category is FailureCategory.CLASSIFICATION_UNCERTAINTY:
        scores = view.projection.get("scores") or {}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        margin = view.projection.get("margin")
        if len(ranked) >= 2 and margin is not None:
            sentence = (
                f"{label} could not separate '{ranked[0][0]}' ({ranked[0][1]:.2f}) from "
                f"'{ranked[1][0]}' ({ranked[1][1]:.2f}): a margin of {margin:.2f} is below the "
                f"{AMBIGUITY_MARGIN:.2f} ambiguity threshold, so the summary shape chosen from it "
                "is a coin flip."
            )
        else:
            sentence = f"{label} was uncertain: {next(iter(messages.values()), '')}"
        return sentence, {"scores": scores, "margin": margin}, []

    if category is FailureCategory.UNSUPPORTED_CLAIM:
        codes = [w for w in view.problems if category_of(w.code) is category]
        figures = [w.message.split("'")[1] for w in codes if w.message.count("'") >= 2]
        if figures:
            hits = [p for f in figures for p, _ in _find_paths(output, f)]
            where = f" in {', '.join(sorted(set(hits))[:3])}" if hits else ""
            sentence = (
                f"{label} stated {_quote_list(figures, 'figure')}{where}, which matches nothing "
                "in the source document or in Step 2's extracted entities."
            )
        elif view.divergences:
            sentence = f"{label} produced claims that contradict ground truth: " + "; ".join(
                d.describe() for d in view.divergences[:3]
            ) + "."
            figures = [str(d.actual) for d in view.divergences if d.actual]
        else:
            sentence = f"{label} made an unsupported claim: {codes[0].message if codes else ''}"
        return sentence, view.projection, figures

    first = view.problems[0] if view.problems else (view.soft[0] if view.soft else None)
    if first is not None:
        sentence = f"{label} raised {first.code}: {first.message}"
    elif view.divergences:
        sentence = f"{label} diverged from ground truth: " + "; ".join(d.describe() for d in view.divergences[:3]) + "."
    else:
        sentence = f"{label} is the most likely origin."
    return sentence, view.projection, []


# -- propagation ---------------------------------------------------------------


def _propagation(
    view: _View, root: _View, category: FailureCategory, values: list[str]
) -> EvidenceLink | None:
    label = step_label(view.step)
    if view.span.status is SpanStatus.SKIPPED:
        return EvidenceLink(
            step=view.step,
            role="propagation",
            finding=f"{label} never ran: it was skipped because {step_label(root.step)} failed.",
        )

    details: list[str] = []
    traced: list[str] = []
    for value in values:
        hits = _find_paths(view.output or {}, value)
        if hits:
            traced.append(value)
            path, text = hits[0]
            extra = f" (and {len(hits) - 1} more place(s))" if len(hits) > 1 else ""
            snippet = f" - \"{_clip(text, 90)}\"" if text.casefold() != value.casefold() else ""
            details.append(f"included '{value}' in {path}{snippet}{extra}")

    for divergence in view.divergences:
        if divergence.kind == "unexpected" and any(
            str(divergence.actual).casefold() == v.casefold() for v in traced
        ):
            continue
        details.append(divergence.describe())

    codes: list[str] = []
    for warning in [*view.symptoms, *view.problems]:
        if warning.code in CASCADE_CODES:
            continue
        codes.append(warning.code)
        details.append(f"raised {warning.code} ({warning.message.rstrip('.')})")

    if view.step is StepName.SUMMARIZATION and category in (
        FailureCategory.MISCLASSIFICATION,
        FailureCategory.CLASSIFICATION_UNCERTAINTY,
    ):
        details.insert(0, f"chose its prompt and output schema from that label ('{view.projection.get('doc_type')}')")

    if not details:
        return None

    return EvidenceLink(
        step=view.step,
        role="propagation",
        finding=f"This propagated to {label}, which " + "; ".join(details[:4]) + ".",
        warning_codes=codes,
        traced_values=traced,
        produced=view.projection,
        expected=view.expected,
        divergences=view.divergences,
    )


# -- entry point ---------------------------------------------------------------


def _health(view: _View) -> StepHealth:
    span = view.span
    if span.status is SpanStatus.SKIPPED:
        return StepHealth.SKIPPED
    low = (span.self_confidence is not None and span.self_confidence <= LOW_CONFIDENCE) or (
        span.computed_confidence is not None and span.computed_confidence < 0.5
    )
    if low or view.implicated or view.soft:
        return StepHealth.LOW_CONFIDENCE
    return StepHealth.HEALTHY


def diagnose(trace: Trace, golden: GoldenRecord | None = None) -> Diagnosis:
    """Walk ``trace`` backwards and explain what broke, where, and how it spread."""
    clock = time.perf_counter()
    views = _views(trace, golden)

    root = next((v for v in views if v.implicated), None)
    if root is None:
        root = next((v for v in views if v.soft), None)

    assessments = [
        StepAssessment(
            step=v.step,
            health=_health(v),
            status=v.span.status,
            suspicion=v.span.suspicion,
            self_confidence=v.span.self_confidence,
            computed_confidence=v.span.computed_confidence,
            problem_codes=v.problem_codes(),
            divergence_count=len(v.divergences),
        )
        for v in views
    ]

    if root is None:
        return Diagnosis(
            trace_id=trace.trace_id,
            doc_id=trace.doc_id,
            verdict="healthy",
            method="ground_truth" if golden else "heuristic",
            confidence=0.9 if golden else 0.6,
            headline="No failure found",
            narrative=(
                "Every step passed its checks"
                + (" and every output matches ground truth." if golden else ".")
                + " If the output is still wrong, flag it and name the step - that is a "
                "failure the checks cannot see yet."
            ),
            steps=assessments,
            computed_ms=round((time.perf_counter() - clock) * 1000, 3),
        )

    category = _categorize(root)
    finding, produced, values = _root_finding(root, category, views)
    chain = [
        EvidenceLink(
            step=root.step,
            role="root_cause",
            finding=finding,
            warning_codes=root.problem_codes() or [w.code for w in root.soft],
            traced_values=values,
            received=_received(root, views),
            produced=produced,
            expected=root.expected,
            divergences=root.divergences,
        )
    ]
    downstream = views[views.index(root) + 1 :]
    for view in downstream:
        link = _propagation(view, root, category, values)
        if link is not None:
            link.received = _received(view, views)
            chain.append(link)

    affected = [link.step for link in chain[1:]]
    for assessment in assessments:
        if assessment.step is root.step:
            assessment.health = StepHealth.ROOT_CAUSE
        elif assessment.step in affected and assessment.health is not StepHealth.SKIPPED:
            assessment.health = StepHealth.AFFECTED

    used_golden = bool(golden) and bool(root.divergences or any(v.divergences for v in downstream))
    if used_golden:
        confidence = 0.95 if (root.problems or root.failed) else 0.85
    elif root.failed:
        confidence = 0.95
    elif root.problems:
        confidence = round(min(0.9, 0.55 + 0.4 * root.span.suspicion), 2)
    else:
        confidence = 0.35

    if not affected and len(chain) == 1:
        tail = " No downstream step shows a symptom of it."
    else:
        tail = ""
    narrative = " ".join(link.finding for link in chain) + tail

    return Diagnosis(
        trace_id=trace.trace_id,
        doc_id=trace.doc_id,
        verdict="failure",
        root_cause_step=root.step,
        category=category,
        category_label=CATEGORY_LABELS[category],
        confidence=confidence,
        method="ground_truth" if used_golden else "heuristic",
        headline=f"{CATEGORY_LABELS[category]} at {step_label(root.step)}",
        narrative=narrative,
        chain=chain,
        steps=assessments,
        affected_steps=affected,
        computed_ms=round((time.perf_counter() - clock) * 1000, 3),
    )


def step_diffs(
    trace: Trace,
    golden: GoldenRecord | None = None,
    corrections: dict[StepName, dict] | None = None,
) -> list[dict]:
    """Received vs produced vs expected, per step, for the diff view.

    ``corrections`` (a human's corrected output) take precedence over ground
    truth for the steps they cover: the person who flagged the trace is the
    final authority on what it should have said.
    """
    views = _views(trace, golden)
    rows = []
    for view in views:
        expected = (corrections or {}).get(view.step, view.expected)
        source = "correction" if corrections and view.step in corrections else ("golden" if view.expected else None)
        divergences = compare(expected, view.projection) if expected and view.output is not None else []
        rows.append(
            {
                "step": view.step.value,
                "status": view.span.status.value,
                "received": _received(view, views),
                "produced": view.projection,
                "expected": expected,
                "expected_source": source,
                "divergences": [d.model_dump(mode="json") for d in divergences],
            }
        )
    return rows
