"""One object that ties traces, ground truth and feedback together.

The CLI and the web explorer both go through a :class:`Workspace`, so a flag
raised in the browser and one raised from the terminal produce the same records
and the same eval cases.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from obspipe.analysis.evidence import Diagnosis, diagnose, step_diffs
from obspipe.analysis.golden import GoldenIndex, GoldenRecord
from obspipe.analysis.taxonomy import FailureCategory
from obspipe.corpus import DEFAULT_CORPUS_DIR, DEMO_CORPUS_DIR
from obspipe.feedback import analytics
from obspipe.feedback.evals import case_from_flag, run_evals
from obspipe.feedback.models import EvalCase, EvalRun, Flag, FlagStatus
from obspipe.feedback.store import DEFAULT_DATASET, FeedbackStore
from obspipe.models.common import StepName
from obspipe.pipeline import get_config
from obspipe.storage import DEFAULT_ROOT, TraceStore
from obspipe.tracing.trace import Trace

DEFAULT_GOLDEN_DIRS = (DEFAULT_CORPUS_DIR, DEMO_CORPUS_DIR)
"""Where golden sidecars are looked up.

The two corpora live in different places on purpose: the committed fixtures
under ``tests/``, the generated demo set under untracked ``data/``. Deriving
one path from the other silently broke demo golden lookup when the fixtures
moved, so both are now named explicitly.
"""


class FlagError(ValueError):
    """A flag operation that cannot be applied as asked."""


class Workspace:
    def __init__(
        self,
        traces_dir: Path | str = DEFAULT_ROOT,
        dataset_path: Path | str = DEFAULT_DATASET,
        golden_dirs: Iterable[Path | str] = DEFAULT_GOLDEN_DIRS,
    ) -> None:
        self.traces = TraceStore(traces_dir)
        self.feedback = FeedbackStore(Path(traces_dir) / "feedback.db", dataset_path)
        self.golden = GoldenIndex(golden_dirs)

    # -- traces & diagnosis -------------------------------------------------

    def resolve_trace_id(self, trace_id: str) -> str:
        matches = self.traces.index.resolve_id(trace_id)
        if not matches:
            raise KeyError(f"No trace matching '{trace_id}'.")
        if len(matches) > 1:
            raise KeyError(f"'{trace_id}' is ambiguous: {', '.join(m[:12] for m in matches)}")
        return matches[0]

    def load(self, trace_id: str) -> Trace:
        return self.traces.load(self.resolve_trace_id(trace_id))

    def add_golden_dir(self, directory: Path | str) -> int:
        """Register another directory of golden sidecars.

        Returns the total number of records known afterwards. Used when running
        against a corpus that is not one of the defaults - the corpus owns its
        ground truth, so it has to be consulted for it.
        """
        self.golden.extend(directory)
        return len(self.golden)

    def golden_for(self, trace: Trace) -> GoldenRecord | None:
        return self.golden.get(trace.doc_id)

    def save(self, trace: Trace) -> Diagnosis:
        """Store a trace and its diagnosis together, the way every run should."""
        self.traces.save(trace)
        return self.diagnose(trace, refresh=True)

    def diagnose(self, trace: Trace, refresh: bool = False) -> Diagnosis:
        if not refresh:
            cached = self.feedback.get_diagnosis(trace.trace_id)
            if cached is not None:
                return cached
        diagnosis = diagnose(trace, self.golden_for(trace))
        self.feedback.save_diagnosis(diagnosis, trace.started_at)
        return diagnosis

    def corrections_for(self, trace_id: str) -> dict[StepName, dict]:
        for flag in self.feedback.flags(trace_id):
            if flag.corrected_output and flag.confirmed_step:
                return {flag.confirmed_step: flag.corrected_output}
        return {}

    def diffs(self, trace: Trace) -> list[dict]:
        return step_diffs(trace, self.golden_for(trace), self.corrections_for(trace.trace_id))

    def diagnose_all(self) -> int:
        count = 0
        for row in self.traces.index.all_rows():
            try:
                trace = self.traces.load(row["trace_id"])
            except KeyError:
                continue
            self.diagnose(trace, refresh=True)
            count += 1
        return count

    # -- flagging -----------------------------------------------------------

    def flag(self, trace_id: str, reason: str = "", actor: str = "user") -> tuple[Flag, Diagnosis]:
        """Mark a trace's output as bad and run the backward analysis on it."""
        trace = self.load(trace_id)
        diagnosis = self.diagnose(trace, refresh=True)
        flag = Flag(
            trace_id=trace.trace_id,
            doc_id=trace.doc_id,
            reason=reason,
            actor=actor,
            diagnosed_step=diagnosis.root_cause_step,
            diagnosed_category=diagnosis.category,
            diagnosis_confidence=diagnosis.confidence,
            diagnosis_ms=diagnosis.computed_ms,
        )
        return self.feedback.save_flag(flag), diagnosis

    def resolve_flag(
        self,
        flag_id: str,
        step: StepName | str | None = None,
        category: FailureCategory | str | None = None,
        corrected_output: dict[str, Any] | None = None,
        notes: str = "",
        resolved_at: datetime | None = None,
    ) -> tuple[Flag, EvalCase]:
        """Confirm or override the diagnosis; either way, grow the eval dataset."""
        flag = self.feedback.get_flag(flag_id)
        if flag is None:
            raise FlagError(f"No flag {flag_id}.")
        if flag.status not in (FlagStatus.OPEN,):
            raise FlagError(f"Flag {flag_id} is already {flag.status.value}.")

        chosen_step = StepName(step) if step else flag.diagnosed_step
        chosen_category = FailureCategory(category) if category else flag.diagnosed_category
        if chosen_step is None or chosen_category is None:
            raise FlagError(
                "The automated diagnosis found no root cause; name the failing step and category."
            )

        trace = self.load(flag.trace_id)
        if corrected_output is None:
            golden = self.golden_for(trace)
            corrected_output = golden.for_step(chosen_step) if golden else None

        flag.status = (
            FlagStatus.CONFIRMED
            if chosen_step == flag.diagnosed_step and chosen_category == flag.diagnosed_category
            else FlagStatus.OVERRIDDEN
        )
        flag.confirmed_step = chosen_step
        flag.confirmed_category = chosen_category
        flag.corrected_output = corrected_output
        flag.notes = notes
        flag.resolved_at = resolved_at or datetime.now(timezone.utc)

        case = case_from_flag(flag, trace, chosen_step, chosen_category, corrected_output)
        self.feedback.append_case(case)
        flag.eval_case_id = case.case_id
        self.feedback.save_flag(flag)
        return flag, case

    def dismiss_flag(self, flag_id: str, notes: str = "") -> Flag:
        flag = self.feedback.get_flag(flag_id)
        if flag is None:
            raise FlagError(f"No flag {flag_id}.")
        flag.status = FlagStatus.DISMISSED
        flag.notes = notes
        flag.resolved_at = datetime.now(timezone.utc)
        return self.feedback.save_flag(flag)

    # -- evals & analytics ----------------------------------------------------

    def run_evals(self, version: str | None = None, label: str = "", store_traces: bool = True) -> EvalRun:
        config = get_config(version)
        run, traces = run_evals(self.feedback.cases(), config, label=label)
        if store_traces:
            for trace in traces:
                self.traces.save(trace)
        return self.feedback.save_run(run)

    def analytics(self) -> dict[str, Any]:
        return analytics.compute(self.traces.index, self.feedback)
