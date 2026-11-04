"""Seed a workspace with a realistic history for the demo.

Three production batches of the 50-document corpus, one per pipeline version,
with an eval run after each - which is the history a team would actually have
after shipping two fixes. Most failures from the first batch are flagged and
confirmed (as ``actor="seed"``, so they never count towards human
time-to-root-cause); a few are deliberately left unflagged so the live demo has
something real to flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from obspipe.corpus import load_corpus
from obspipe.llm.backends import get_backend
from obspipe.pipeline import get_config, run_traced
from obspipe.workspace import DEMO_CORPUS_DIR, Workspace

RESERVED_FOR_LIVE_DEMO = frozenset({"009_contract_lease", "017_invoice_wingtip", "021_memo_review"})
"""Left unflagged: the hallucinated 'John Smith', the silent misclassification
(the one only a human override can catch) and the ambiguous memo."""

NOT_FLAGGED = frozenset({"045_upload_blank"})
"""Correctly rejected input. A flag on it would teach the eval set nothing."""

BATCHES = (("batch-01", "v1"), ("batch-02", "v2"), ("batch-03", "v3"))


def run_batch(
    workspace: Workspace, corpus_dir: Path | str, batch: str, version: str
) -> list[tuple[str, str]]:
    """Run every document once; return ``(doc_id, trace_id)`` for each failure."""
    config = get_config(version)
    failures: list[tuple[str, str]] = []
    for entry in load_corpus(corpus_dir):
        faults = set(entry.expectation.simulated_faults) if entry.expectation else set()
        trace = run_traced(
            entry.document,
            get_backend("fake", faults=faults),
            config=config,
            tags={"batch": batch},
        )
        diagnosis = workspace.save(trace)
        if diagnosis.verdict == "failure":
            failures.append((entry.document.doc_id, trace.trace_id))
    return failures


def seed(
    workspace: Workspace,
    corpus_dir: Path | str = DEMO_CORPUS_DIR,
    log: Callable[[str], None] = print,
) -> dict:
    if workspace.feedback.cases():
        raise RuntimeError(
            "The eval dataset is not empty; seeding twice would duplicate every case. "
            "Run with --reset, or point --dir/--dataset at a fresh workspace."
        )

    summary: dict = {"batches": [], "flagged": [], "eval_runs": []}
    for position, (batch, version) in enumerate(BATCHES):
        failures = run_batch(workspace, corpus_dir, batch, version)
        log(f"{batch} ({version}): {len(failures)} failure(s) of 50")
        summary["batches"].append({"batch": batch, "version": version, "failures": len(failures)})

        if position == 0:
            for doc_id, trace_id in failures:
                if doc_id in RESERVED_FOR_LIVE_DEMO or doc_id in NOT_FLAGGED:
                    continue
                flag, _ = workspace.flag(trace_id, reason="Output looked wrong in review.", actor="seed")
                workspace.resolve_flag(flag.flag_id, notes="Confirmed during triage.")
                summary["flagged"].append(doc_id)
            log(f"  flagged and confirmed {len(summary['flagged'])}; eval dataset now "
                f"{len(workspace.feedback.cases())} case(s)")

        run = workspace.run_evals(version, label=f"after {batch}")
        log(f"  eval {version}: {run.passed}/{run.total} known issues resolved")
        summary["eval_runs"].append({"version": version, "passed": run.passed, "total": run.total})
    return summary


def reset(traces_dir: Path | str, dataset_path: Path | str) -> list[Path]:
    """Delete what a seed creates: trace files, both databases, the dataset."""
    root = Path(traces_dir)
    removed: list[Path] = []
    for path in sorted(root.glob("*/*.json")):
        path.unlink()
        removed.append(path)
    for name in ("index.db", "feedback.db"):
        if (root / name).exists():
            (root / name).unlink()
            removed.append(root / name)
    dataset = Path(dataset_path)
    if dataset.exists():
        dataset.unlink()
        removed.append(dataset)
    for day in root.glob("*"):
        if day.is_dir() and not any(day.iterdir()):
            day.rmdir()
    return removed
