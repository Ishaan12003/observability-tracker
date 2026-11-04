"""Failure analytics: where a product team should invest engineering effort.

Everything here is an aggregate over three sources - the trace index, the
cached diagnoses and the feedback records - and excludes traces produced by
eval replays, which would otherwise count every known failure twice.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from obspipe.analysis.taxonomy import CATEGORY_LABELS, FailureCategory
from obspipe.feedback.models import FlagStatus
from obspipe.feedback.store import FeedbackStore
from obspipe.storage.index import TraceIndex

STEP_ORDER = ("intake", "extraction", "classification", "summarization")


def _is_eval(row: Any) -> bool:
    return bool(row["batch"]) and str(row["batch"]).startswith("eval-")


def compute(index: TraceIndex, feedback: FeedbackStore) -> dict[str, Any]:
    traces = [row for row in index.all_rows() if not _is_eval(row)]
    trace_ids = {row["trace_id"] for row in traces}
    diagnoses = {
        row["trace_id"]: row for row in feedback.diagnosis_rows() if row["trace_id"] in trace_ids
    }
    failures = [d for d in diagnoses.values() if d["verdict"] == "failure"]

    categories = Counter(d["category"] for d in failures if d["category"])
    root_steps = Counter(d["root_step"] for d in failures if d["root_step"])

    # failure rate per batch, in the order batches were first run
    batches: dict[str, dict[str, Any]] = {}
    for row in traces:
        key = row["batch"] or row["created_at"][:10]
        bucket = batches.setdefault(
            key,
            {"batch": key, "first_seen": row["created_at"], "pipeline_version": row["pipeline_version"],
             "traces": 0, "failures": 0},
        )
        bucket["traces"] += 1
        diagnosis = diagnoses.get(row["trace_id"])
        if diagnosis is not None and diagnosis["verdict"] == "failure":
            bucket["failures"] += 1
    over_time = sorted(batches.values(), key=lambda b: b["first_seen"])
    for bucket in over_time:
        bucket["failure_rate"] = round(bucket["failures"] / bucket["traces"], 4) if bucket["traces"] else 0.0

    by_step_category: dict[str, Counter] = defaultdict(Counter)
    for d in failures:
        if d["root_step"] and d["category"]:
            by_step_category[d["root_step"]][d["category"]] += 1

    automated = [d["computed_ms"] for d in diagnoses.values() if d["computed_ms"] is not None]
    flags = feedback.flags()
    resolved = [f for f in flags if f.status in (FlagStatus.CONFIRMED, FlagStatus.OVERRIDDEN)]
    human = [f.seconds_to_root_cause for f in resolved if f.actor != "seed" and f.seconds_to_root_cause is not None]
    agreed = sum(f.status is FlagStatus.CONFIRMED for f in resolved)

    runs = feedback.runs()
    trend = [
        {
            "run_id": r["run_id"],
            "started_at": r["started_at"],
            "pipeline_version": r["pipeline_version"],
            "label": r["label"],
            "total": r["total"],
            "passed": r["passed"],
            "resolved_rate": round(r["passed"] / r["total"], 4) if r["total"] else 0.0,
        }
        for r in runs
    ]

    return {
        "totals": {
            "traces": len(traces),
            "diagnosed": len(diagnoses),
            "failures": len(failures),
            "failure_rate": round(len(failures) / len(diagnoses), 4) if diagnoses else 0.0,
            "flags": len(flags),
            "eval_cases": len(feedback.cases()),
        },
        "by_category": [
            {"category": c, "label": CATEGORY_LABELS[FailureCategory(c)], "count": n}
            for c, n in categories.most_common()
        ],
        "by_root_step": [{"step": s, "count": root_steps.get(s, 0)} for s in STEP_ORDER],
        "step_category_matrix": {
            step: dict(by_step_category.get(step, {})) for step in STEP_ORDER
        },
        "failure_rate_over_time": over_time,
        "time_to_root_cause": {
            "automated_ms_mean": round(mean(automated), 2) if automated else None,
            "automated_ms_median": round(median(automated), 2) if automated else None,
            "automated_samples": len(automated),
            "human_seconds_mean": round(mean(human), 1) if human else None,
            "human_seconds_median": round(median(human), 1) if human else None,
            "human_samples": len(human),
            "diagnosis_agreement": round(agreed / len(resolved), 4) if resolved else None,
            "resolved_flags": len(resolved),
        },
        "eval_trend": trend,
    }
