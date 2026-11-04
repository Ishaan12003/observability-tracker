"""Phases 4-6: the demo corpus, flags, eval cases, regression runs, analytics, the API."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from obspipe.analysis import FailureCategory, GoldenIndex, diagnose
from obspipe.corpus import load_corpus
from obspipe.demo.generator import FAILURE_PLAN, build_corpus, write_corpus
from obspipe.demo.seed import RESERVED_FOR_LIVE_DEMO, seed
from obspipe.feedback.models import FlagStatus
from obspipe.llm.backends import FakeBackend
from obspipe.models.common import StepName
from obspipe.pipeline import get_config, run_traced
from obspipe.web.server import build_server
from obspipe.workspace import FlagError, Workspace

EXPECTED_DIAGNOSES = {
    "004_contract_long_lease": ("intake", "input_truncation"),
    "009_contract_lease": ("extraction", "hallucination"),
    "013_invoice_contoso": ("summarization", "unsupported_claim"),
    "017_invoice_wingtip": ("classification", "misclassification"),
    "021_memo_review": ("classification", "classification_uncertainty"),
    "024_invoice_bellows": ("extraction", "recall_miss"),
    "028_scan_brandt": ("intake", "input_corruption"),
    "031_contract_lease": ("extraction", "hallucination"),
    "035_letter_brandt": ("classification", "llm_call_failure"),
    "038_invoice_coho": ("summarization", "invalid_aggregation"),
    "042_invoice_datum": ("extraction", "normalization_error"),
    "045_upload_blank": ("intake", "invalid_input"),
    "048_contract_long_license": ("intake", "input_truncation"),
}


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("demo")
    write_corpus(path)
    return path


@pytest.fixture
def workspace(tmp_path, demo_dir):
    return Workspace(tmp_path / "traces", tmp_path / "evals" / "dataset.jsonl", golden_dirs=[demo_dir])


def _run_doc(workspace, demo_dir, doc_id, version="v1"):
    entry = next(e for e in load_corpus(demo_dir) if e.document.doc_id == doc_id)
    trace = run_traced(entry.document, FakeBackend(set(entry.expectation.simulated_faults)),
                       config=get_config(version), tags={"batch": "test"})
    workspace.save(trace)
    return trace


# -- corpus -----------------------------------------------------------------------


def test_the_demo_corpus_is_fifty_documents_with_thirteen_designed_failures():
    docs = build_corpus()
    assert len(docs) == 50
    assert sum(1 for d in docs if d.expected_failure) == len(FAILURE_PLAN) == 13
    assert len({d.doc_id for d in docs}) == 50


def test_generation_is_deterministic():
    assert [d.content for d in build_corpus()] == [d.content for d in build_corpus()]


def test_every_document_is_diagnosed_as_designed(demo_dir):
    """Clean documents match independent ground truth; each failure lands on its step."""
    golden = GoldenIndex([demo_dir])
    for entry in load_corpus(demo_dir):
        doc_id = entry.document.doc_id
        trace = run_traced(entry.document, FakeBackend(set(entry.expectation.simulated_faults)),
                           config=get_config("v1"))
        diagnosis = diagnose(trace, golden.get(doc_id))
        if doc_id in EXPECTED_DIAGNOSES:
            step, category = EXPECTED_DIAGNOSES[doc_id]
            assert (diagnosis.root_cause_step.value, diagnosis.category.value) == (step, category), doc_id
        else:
            assert diagnosis.verdict == "healthy", f"{doc_id}: {diagnosis.narrative}"


# -- flagging ---------------------------------------------------------------------


def test_flagging_runs_the_diagnosis_and_confirming_grows_the_dataset(workspace, demo_dir):
    trace = _run_doc(workspace, demo_dir, "009_contract_lease")
    flag, diagnosis = workspace.flag(trace.trace_id, reason="who is John Smith?")
    assert flag.diagnosed_step is StepName.EXTRACTION
    assert diagnosis.category is FailureCategory.HALLUCINATION

    resolved, case = workspace.resolve_flag(flag.flag_id)
    assert resolved.status is FlagStatus.CONFIRMED
    assert case.failing_step is StepName.EXTRACTION
    assert case.failure_codes == ["hallucinated_party_span"]
    assert "John Smith" in case.bad_output["parties"]
    assert "John Smith" not in case.corrected_output["parties"], "defaults to ground truth"
    assert case.input.faults == ["hallucinate_party"]
    assert len(workspace.feedback.cases()) == 1
    line = workspace.feedback.dataset_path.read_text().strip()
    assert json.loads(line)["case_id"] == case.case_id


def test_an_override_is_recorded_as_such(workspace, demo_dir):
    trace = _run_doc(workspace, demo_dir, "021_memo_review")
    flag, _ = workspace.flag(trace.trace_id)
    resolved, case = workspace.resolve_flag(
        flag.flag_id, step="summarization", category="missing_disclosure",
        corrected_output={"doc_type": "report"},
    )
    assert resolved.status is FlagStatus.OVERRIDDEN
    assert case.failure_category is FailureCategory.MISSING_DISCLOSURE
    assert case.corrected_output == {"doc_type": "report"}


def test_a_flag_on_a_healthy_trace_needs_a_human_verdict(workspace, demo_dir):
    entry = next(e for e in load_corpus(demo_dir) if not e.expectation.expected_failure)
    trace = run_traced(entry.document, FakeBackend())
    workspace.save(trace)
    flag, diagnosis = workspace.flag(trace.trace_id)
    assert diagnosis.verdict == "healthy"
    with pytest.raises(FlagError):
        workspace.resolve_flag(flag.flag_id)
    _, case = workspace.resolve_flag(flag.flag_id, step="summarization", category="low_quality_output")
    assert case.failure_codes == []


def test_a_flag_cannot_be_resolved_twice(workspace, demo_dir):
    trace = _run_doc(workspace, demo_dir, "013_invoice_contoso")
    flag, _ = workspace.flag(trace.trace_id)
    workspace.resolve_flag(flag.flag_id)
    with pytest.raises(FlagError):
        workspace.resolve_flag(flag.flag_id)


# -- regression -------------------------------------------------------------------


def test_a_truncation_case_fails_on_v1_and_is_fixed_by_v2(workspace, demo_dir):
    trace = _run_doc(workspace, demo_dir, "004_contract_long_lease")
    flag, _ = workspace.flag(trace.trace_id)
    workspace.resolve_flag(flag.flag_id)

    v1 = workspace.run_evals("v1")
    v2 = workspace.run_evals("v2")
    assert (v1.passed, v2.passed) == (0, 1)
    assert "document_truncated" in v1.results[0].detail


def test_a_silent_misclassification_stays_failing_until_its_correction_holds(workspace, demo_dir):
    trace = _run_doc(workspace, demo_dir, "017_invoice_wingtip")
    flag, _ = workspace.flag(trace.trace_id)
    _, case = workspace.resolve_flag(flag.flag_id)
    assert case.failure_codes == [], "no check fired - only the correction can detect it"
    run = workspace.run_evals("v3")
    assert not run.results[0].passed
    assert run.results[0].divergences[0]["path"] == "doc_type"


def test_seed_builds_the_demo_history_and_analytics_excludes_eval_replays(tmp_path, demo_dir):
    workspace = Workspace(tmp_path / "traces", tmp_path / "dataset.jsonl", golden_dirs=[demo_dir])
    summary = seed(workspace, demo_dir, log=lambda _: None)
    assert [b["failures"] for b in summary["batches"]] == [13, 11, 8]
    assert [(r["passed"], r["total"]) for r in summary["eval_runs"]] == [(0, 9), (2, 9), (4, 9)]
    assert not RESERVED_FOR_LIVE_DEMO & set(summary["flagged"])

    data = workspace.analytics()
    assert data["totals"]["traces"] == 150
    assert data["totals"]["failures"] == 32
    assert [b["failures"] for b in data["failure_rate_over_time"]] == [13, 11, 8]
    assert data["time_to_root_cause"]["human_samples"] == 0, "seeded flags are not human"
    assert [r["passed"] for r in data["eval_trend"]] == [0, 2, 4]
    with pytest.raises(RuntimeError):
        seed(workspace, demo_dir, log=lambda _: None)


# -- web API ------------------------------------------------------------------------


@pytest.fixture
def server(workspace):
    srv = build_server(workspace, port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _call(url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def test_the_api_supports_the_whole_flag_to_eval_loop(server, workspace, demo_dir):
    trace = _run_doc(workspace, demo_dir, "009_contract_lease")

    listing = _call(f"{server}/api/traces")
    assert listing["traces"][0]["verdict"] == "failure"

    detail = _call(f"{server}/api/traces/{trace.trace_id[:8]}")
    assert detail["diagnosis"]["root_cause_step"] == "extraction"
    assert [s["step"] for s in detail["spans"]] == ["intake", "extraction", "classification", "summarization"]
    assert detail["spans"][1]["llm_calls"][0]["prompt"], "prompts are rehydrated"
    extraction_diff = next(d for d in detail["diffs"] if d["step"] == "extraction")
    assert {"path": "parties", "kind": "unexpected", "expected": None, "actual": "John Smith"} in extraction_diff["divergences"]

    flagged = _call(f"{server}/api/traces/{trace.trace_id}/flag", {"reason": "bad"})
    resolved = _call(f"{server}/api/flags/{flagged['flag']['flag_id']}/resolve", {})
    assert resolved["dataset_size"] == 1

    run = _call(f"{server}/api/evals/run", {"version": "v3"})
    assert (run["passed"], run["total"]) == (1, 1)
    assert _call(f"{server}/api/analytics")["totals"]["eval_cases"] == 1


def test_the_api_returns_json_errors(server):
    with pytest.raises(urllib.error.HTTPError) as missing:
        _call(f"{server}/api/traces/deadbeef")
    assert missing.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as empty:
        _call(f"{server}/api/evals/run", {"version": "v3"})
    assert empty.value.code == 409


def test_the_explorer_shell_is_served(server):
    with urllib.request.urlopen(f"{server}/") as response:
        assert b"trace explorer" in response.read()
    with urllib.request.urlopen(f"{server}/static/app.js") as response:
        assert response.headers["Content-Type"].startswith("text/javascript")
