from __future__ import annotations

import json

import pytest
from rich.console import Console
from typer.testing import CliRunner

from obspipe.cli import app

runner = CliRunner()


@pytest.fixture
def wide_console(monkeypatch):
    """Rich falls back to 80 columns off a TTY, which ellipsises table cells."""
    monkeypatch.setattr("obspipe.cli.console", Console(width=200))


def test_docs_lists_the_corpus(wide_console):
    result = runner.invoke(app, ["docs"])
    assert result.exit_code == 0
    assert "invoice_mixed_currency" in result.stdout


def test_run_emits_a_parsable_json_result(tmp_path):
    out = tmp_path / "result.json"
    result = runner.invoke(
        app,
        ["run", "tests/fixtures/documents/invoice_mixed_currency.txt", "--json", "--out", str(out)],
    )
    assert result.exit_code == 0
    payload = json.loads(out.read_text())
    assert payload["doc_id"] == "invoice_mixed_currency"
    assert [span["step"] for span in payload["spans"]] == [
        "intake",
        "extraction",
        "classification",
        "summarization",
    ]


def test_run_exits_nonzero_when_the_pipeline_fails():
    result = runner.invoke(app, ["run", "tests/fixtures/documents/empty.txt"])
    assert result.exit_code == 1
    assert "whitespace_only_document" in result.stdout


def test_run_accepts_injected_faults():
    result = runner.invoke(
        app, ["run", "tests/fixtures/documents/contract_no_dates.md", "--fault", "hallucinate_date"]
    )
    assert "hallucinated_date_span" in result.stdout


def test_unknown_fault_is_rejected():
    result = runner.invoke(app, ["run", "tests/fixtures/documents/contract_no_dates.md", "--fault", "nope"])
    assert result.exit_code == 2
    assert "Unknown fault" in result.stdout


def test_run_all_writes_one_result_per_document(tmp_path):
    from obspipe.corpus import load_corpus

    result = runner.invoke(app, ["run-all", "--out-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert len(list(tmp_path.glob("*.json"))) == len(load_corpus())


def test_groq_backend_without_a_key_fails_cleanly(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("obspipe.cli.load_dotenv", lambda *a, **k: None)
    result = runner.invoke(
        app, ["run", "tests/fixtures/documents/contract_no_dates.md", "--backend", "groq"]
    )
    assert result.exit_code == 2
    assert "GROQ_API_KEY" in result.stdout


# -- Phase 2: trace commands ---------------------------------------------


def _run_and_store(tmp_path, document="tests/fixtures/documents/invoice_mixed_currency.txt"):
    result = runner.invoke(app, ["run", document, "--traces-dir", str(tmp_path)])
    assert result.exit_code in (0, 1), result.stdout
    return result


def test_run_persists_a_trace_by_default(tmp_path):
    _run_and_store(tmp_path)
    from obspipe.storage import TraceStore

    assert TraceStore(tmp_path).index.count() == 1
    assert list(tmp_path.glob("*/*.json"))


def test_no_store_skips_persistence(tmp_path):
    result = runner.invoke(
        app,
        ["run", "tests/fixtures/documents/invoice_no_total.txt", "--traces-dir", str(tmp_path), "--no-store"],
    )
    assert result.exit_code == 0
    assert not list(tmp_path.glob("*/*.json"))


def test_trace_json_flag_emits_the_whole_trace(tmp_path):
    out = tmp_path / "trace.json"
    result = runner.invoke(
        app,
        ["run", "tests/fixtures/documents/invoice_mixed_currency.txt", "--json", "--trace",
         "--out", str(out), "--traces-dir", str(tmp_path), "--no-store"],
    )
    assert result.exit_code == 0
    payload = json.loads(out.read_text())
    assert "spans" in payload and "payloads" in payload
    assert any(span["llm_calls"] for span in payload["spans"])


def test_traces_command_lists_stored_runs(tmp_path, wide_console):
    _run_and_store(tmp_path)
    result = runner.invoke(app, ["traces", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "invoice_mixed_currency" in result.stdout


def test_trace_command_shows_spans_and_prompts(tmp_path):
    _run_and_store(tmp_path)
    from obspipe.storage import TraceStore

    trace_id = TraceStore(tmp_path).index.recent(1)[0]["trace_id"]
    result = runner.invoke(app, ["trace", trace_id[:8], "--dir", str(tmp_path), "--prompts"])
    assert result.exit_code == 0
    assert "extraction" in result.stdout
    assert "system" in result.stdout, "--prompts should rehydrate the messages"


def test_trace_command_rejects_an_unknown_id(tmp_path):
    _run_and_store(tmp_path)
    result = runner.invoke(app, ["trace", "zzzzzzzz", "--dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "No trace matching" in result.stdout


def test_blame_command_names_the_first_bad_step(tmp_path):
    runner.invoke(
        app, ["run", "tests/fixtures/documents/contract_long_truncated.md", "--traces-dir", str(tmp_path)]
    )
    from obspipe.storage import TraceStore

    trace_id = TraceStore(tmp_path).index.recent(1)[0]["trace_id"]
    result = runner.invoke(app, ["blame", trace_id, "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "first bad step: intake" in result.stdout.replace("\n", " ")


def test_stats_aggregates_across_stored_traces(tmp_path, wide_console):
    runner.invoke(app, ["run-all", "--traces-dir", str(tmp_path)])
    result = runner.invoke(app, ["stats", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "failures by step" in result.stdout
    assert "extraction" in result.stdout


def test_reindex_rebuilds_from_disk(tmp_path):
    runner.invoke(app, ["run-all", "--traces-dir", str(tmp_path)])
    (tmp_path / "index.db").unlink()
    result = runner.invoke(app, ["reindex", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "indexed" in result.stdout


# -- Phase 3-6 commands --------------------------------------------------------


def _first_trace_id(traces_dir):
    from obspipe.storage import TraceIndex

    return TraceIndex(traces_dir / "index.db").recent(limit=1)[0]["trace_id"]


def test_diagnose_flag_confirm_and_eval_run(tmp_path, wide_console):
    traces_dir = tmp_path / "traces"
    dataset = tmp_path / "dataset.jsonl"
    runner.invoke(app, ["run", "tests/fixtures/documents/contract_long_truncated.md", "--traces-dir", str(traces_dir)])
    trace_id = _first_trace_id(traces_dir)

    diagnosed = runner.invoke(app, ["diagnose", trace_id[:8], "--dir", str(traces_dir)])
    assert diagnosed.exit_code == 0
    assert "Input truncation at Step 1 (Intake)" in diagnosed.stdout

    flagged = runner.invoke(app, ["flag", trace_id, "--dir", str(traces_dir), "--dataset", str(dataset)])
    flag_id = next(w for w in flagged.stdout.split() if w.startswith("flag_"))
    confirmed = runner.invoke(app, ["confirm", flag_id, "--dir", str(traces_dir), "--dataset", str(dataset),
                                    "--corrected", '{"truncated": false}'])
    assert confirmed.exit_code == 0, confirmed.stdout
    assert "confirmed" in confirmed.stdout and "1 case(s)" in confirmed.stdout

    v1 = runner.invoke(app, ["eval", "run", "--pipeline-version", "v1", "--dir", str(traces_dir), "--dataset", str(dataset)])
    v2 = runner.invoke(app, ["eval", "run", "--pipeline-version", "v2", "--dir", str(traces_dir), "--dataset", str(dataset)])
    assert "0/1" in v1.stdout and "1/1" in v2.stdout


def test_confirm_rejects_an_unknown_flag(tmp_path):
    result = runner.invoke(app, ["confirm", "flag_0000000000", "--dir", str(tmp_path)])
    assert result.exit_code == 1


def test_demo_generate_and_seed(tmp_path):
    corpus = tmp_path / "demo"
    assert runner.invoke(app, ["demo", "generate", "--out", str(corpus)]).exit_code == 0
    result = runner.invoke(app, ["demo", "seed", "--corpus", str(corpus), "--dir", str(tmp_path / "t"),
                                 "--dataset", str(tmp_path / "d.jsonl")])
    assert result.exit_code == 0, result.stdout
    assert "batch-03 (v3): 8 failure(s)" in result.stdout
    assert "4/9 known issues resolved" in result.stdout
