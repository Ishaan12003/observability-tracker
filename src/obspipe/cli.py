"""Command line entry point.

Everything the web explorer does is also reachable from here: run, diagnose,
flag, confirm, replay the eval set, aggregate. The two share one
:class:`~obspipe.workspace.Workspace`, so their records are interchangeable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from obspipe.corpus import DEFAULT_CORPUS_DIR, load_corpus, load_entry
from obspipe.llm.backends import BackendError, get_backend
from obspipe.llm.fake_engine import KNOWN_FAULTS
from obspipe.models.common import Severity
from obspipe.models.pipeline import PipelineResult, StepStatus
from obspipe.pipeline import CURRENT_VERSION, PIPELINE_VERSIONS, get_config, run_traced
from obspipe.storage import DEFAULT_ROOT, TraceStore, trace_path
from obspipe.tracing.models import SpanKind, SpanStatus
from obspipe.tracing.trace import Trace

app = typer.Typer(add_completion=False, help="Traceable multi-step document pipeline.")
eval_app = typer.Typer(add_completion=False, help="The eval dataset grown from human flags.")
demo_app = typer.Typer(add_completion=False, help="The 50-document demo scenario.")
app.add_typer(eval_app, name="eval")
app.add_typer(demo_app, name="demo")
console = Console()

_STATUS_STYLE = {
    StepStatus.OK: "green",
    StepStatus.DEGRADED: "yellow",
    StepStatus.FAILED: "red",
    StepStatus.SKIPPED: "dim",
}
_SEVERITY_STYLE = {Severity.INFO: "dim cyan", Severity.WARN: "yellow", Severity.ERROR: "red"}


def _build_backend(backend: str, model: str | None, fault: list[str] | None):
    try:
        return get_backend(backend, model=model, faults=set(fault or []))
    except BackendError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


def _render(result: PipelineResult) -> None:
    style = _STATUS_STYLE[result.status]
    console.print(
        Panel(
            f"[bold]{result.doc_id}[/bold]  ({result.source_name})\n"
            f"status   [{style}]{result.status.value}[/{style}]\n"
            f"type     {result.doc_type.value}\n"
            f"backend  {result.backend} ({result.model})\n"
            f"run_id   {result.run_id}   {result.duration_ms:.1f} ms",
            title="run",
            expand=False,
        )
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("step")
    table.add_column("status")
    table.add_column("ms", justify="right")
    table.add_column("warnings", justify="right")
    table.add_column("worst")
    table.add_column("conf", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("suspicion", justify="right")
    for record in result.spans:
        record_style = _STATUS_STYLE[record.status]
        worst = record.worst_severity
        table.add_row(
            record.step.value,
            f"[{record_style}]{record.status.value}[/{record_style}]",
            f"{record.duration_ms:.1f}",
            str(len(record.warnings)),
            f"[{_SEVERITY_STYLE[worst]}]{worst.value}[/{_SEVERITY_STYLE[worst]}]" if worst else "-",
            _confidence_cell(record),
            str(record.total_tokens or "-"),
            _suspicion_cell(record.suspicion),
        )
    console.print(table)

    for record in result.spans:
        if record.error_message:
            console.print(
                f"  [red]{record.step.value}: {record.error_type} - {record.error_message}[/red]"
            )
        for warning in record.warnings:
            warn_style = _SEVERITY_STYLE[warning.severity]
            location = f" ({warning.field_path})" if warning.field_path else ""
            console.print(
                f"  [{warn_style}]{record.step.value:15} {warning.severity.value:5} "
                f"{warning.code}{location}[/{warn_style}]\n"
                f"  {' ' * 21}{warning.message}"
            )

    if result.summary:
        console.print(Panel(result.summary.headline, title="headline", expand=False))
        for bullet in result.summary.bullets:
            console.print(f"  - {bullet}")
        for question in result.summary.unresolved_questions:
            console.print(f"  [yellow]? {question}[/yellow]")



def _confidence_cell(span) -> str:
    """Self-reported rating over the pipeline's own measure, when both exist."""
    self_rating = "-" if span.self_confidence is None else f"{span.self_confidence}/5"
    if span.self_confidence is not None and span.self_confidence <= 2:
        self_rating = f"[yellow]{self_rating}[/yellow]"
    computed = "" if span.computed_confidence is None else f" ({span.computed_confidence:.2f})"
    return f"{self_rating}{computed}"


def _suspicion_cell(value: float) -> str:
    if value >= 0.6:
        return f"[red]{value:.2f}[/red]"
    if value >= 0.3:
        return f"[yellow]{value:.2f}[/yellow]"
    return f"{value:.2f}" if value else "-"


def _render_blame(trace: Trace) -> None:
    """Rank the steps by how likely each is to be the origin of a bad output."""
    suspects = [s for s in trace.ranked_suspects() if s.suspicion > 0]
    if not suspects:
        console.print("[green]No step reported a problem.[/green]")
        return

    console.print(
        Panel(
            f"[bold]{trace.doc_id}[/bold]  trace {trace.trace_id}\n"
            f"status {trace.status.value}   "
            f"first bad step: {trace.first_degraded_step.value if trace.first_degraded_step else '-'}",
            title="blame",
            expand=False,
        )
    )
    for rank, span in enumerate(suspects, start=1):
        console.print(
            f"\n[bold]{rank}. {span.name}[/bold]  suspicion "
            f"{_suspicion_cell(span.suspicion)}  status {span.status.value}"
        )
        if span.self_confidence is not None:
            console.print(
                f"   model rated itself {span.self_confidence}/5"
                + (f" - {span.confidence_reason}" if span.confidence_reason else "")
            )
        if span.computed_confidence is not None:
            console.print(f"   pipeline measured {span.computed_confidence:.2f}")
        if span.error_message:
            console.print(f"   [red]{span.error_type}: {span.error_message}[/red]")
        for warning in span.warnings:
            style = _SEVERITY_STYLE[warning.severity]
            console.print(f"   [{style}]{warning.severity.value:5} {warning.code}[/{style}]")


@app.command()
def run(
    path: Annotated[Path, typer.Argument(help="Document to run.")],
    backend: Annotated[str, typer.Option(help="fake or groq.")] = "fake",
    model: Annotated[str | None, typer.Option(help="Override the model id.")] = None,
    fault: Annotated[
        list[str] | None, typer.Option(help="Inject a named fault (fake backend only).")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the result as JSON.")] = False,
    as_trace: Annotated[
        bool, typer.Option("--trace", help="With --json, emit the whole trace instead.")
    ] = False,
    blame: Annotated[
        bool, typer.Option("--blame", help="Rank the steps by likely responsibility.")
    ] = False,
    out: Annotated[Path | None, typer.Option(help="Write the JSON to this file.")] = None,
    traces_dir: Annotated[Path, typer.Option(help="Where traces are stored.")] = DEFAULT_ROOT,
    no_store: Annotated[bool, typer.Option("--no-store", help="Do not persist the trace.")] = False,
    pipeline_version: Annotated[
        str | None, typer.Option("--pipeline-version", help="v1, v2 or v3; default is Phase 1 behaviour.")
    ] = None,
) -> None:
    """Run the four-step pipeline over one document."""
    load_dotenv()
    entry = load_entry(path)
    config = _config_or_exit(pipeline_version) if pipeline_version else None
    trace = run_traced(entry.document, _build_backend(backend, model, fault), config=config)
    result = trace.result

    saved_to = None
    if not no_store:
        _workspace(traces_dir).save(trace)
        saved_to = trace_path(trace, traces_dir)

    payload = trace.model_dump(mode="json") if as_trace else result.model_dump(mode="json")
    if out:
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
    else:
        _render(result)
        if blame:
            _render_blame(trace)
        if out:
            console.print(f"[dim]wrote {out}[/dim]")
        if saved_to:
            console.print(f"[dim]trace {trace.trace_id} -> {saved_to}[/dim]")

    if result.status is StepStatus.FAILED:
        raise typer.Exit(code=1)


@app.command("run-all")
def run_all(
    directory: Annotated[Path, typer.Option("--dir", help="Corpus directory.")] = DEFAULT_CORPUS_DIR,
    backend: Annotated[str, typer.Option(help="fake or groq.")] = "fake",
    model: Annotated[str | None, typer.Option()] = None,
    out_dir: Annotated[Path | None, typer.Option(help="Write one JSON result per document.")] = None,
    traces_dir: Annotated[Path, typer.Option(help="Where traces are stored.")] = DEFAULT_ROOT,
    no_store: Annotated[bool, typer.Option("--no-store", help="Do not persist traces.")] = False,
    pipeline_version: Annotated[str | None, typer.Option("--pipeline-version")] = None,
    batch: Annotated[str | None, typer.Option(help="Tag every trace with this batch name.")] = None,
    simulate: Annotated[
        bool,
        typer.Option(help="Apply each document's simulated_faults (fake backend only)."),
    ] = True,
) -> None:
    """Run every document in the corpus and print a comparison table."""
    load_dotenv()
    llm = _build_backend(backend, model, None)
    workspace = _workspace(traces_dir)
    config = _config_or_exit(pipeline_version) if pipeline_version else None
    tags = {"batch": batch} if batch else None

    table = Table(show_header=True, header_style="bold", title="corpus run")
    table.add_column("document")
    table.add_column("status")
    table.add_column("type")
    table.add_column("expected")
    table.add_column("first bad step")
    table.add_column("warnings")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for entry in load_corpus(directory):
        doc_backend = llm
        faults = entry.expectation.simulated_faults if entry.expectation else []
        if simulate and faults and backend == "fake":
            doc_backend = _build_backend(backend, model, faults)
        trace = run_traced(entry.document, doc_backend, config=config, tags=tags)
        result = trace.result
        if not no_store:
            workspace.save(trace)
        expected = entry.expectation.expected_type.value if entry.expectation else "-"
        matched = expected in ("-", result.doc_type.value)
        style = _STATUS_STYLE[result.status]
        first_bad = result.first_degraded_step
        table.add_row(
            entry.document.doc_id,
            f"[{style}]{result.status.value}[/{style}]",
            result.doc_type.value if matched else f"[red]{result.doc_type.value}[/red]",
            expected,
            first_bad.value if first_bad else "-",
            ", ".join(sorted({w.code for _, w in result.all_warnings()})) or "-",
        )
        if out_dir:
            (out_dir / f"{entry.document.doc_id}.json").write_text(
                json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8"
            )

    console.print(table)
    if out_dir:
        console.print(f"[dim]wrote results to {out_dir}[/dim]")


@app.command()
def docs(
    directory: Annotated[Path, typer.Option("--dir")] = DEFAULT_CORPUS_DIR,
) -> None:
    """List the fixture corpus and the failure each document is designed to cause."""
    table = Table(show_header=True, header_style="bold", title="corpus")
    table.add_column("document")
    table.add_column("expected type")
    table.add_column("designed failure")
    table.add_column("chars", justify="right")
    for entry in load_corpus(directory):
        expectation = entry.expectation
        table.add_row(
            entry.document.doc_id,
            expectation.expected_type.value if expectation else "-",
            expectation.expected_failure if expectation and expectation.expected_failure else "-",
            str(len(entry.document.content)),
        )
    console.print(table)


@app.command()
def faults() -> None:
    """List the faults the fake backend can inject."""
    for name in sorted(KNOWN_FAULTS):
        console.print(f"  {name}")




@app.command()
def traces(
    limit: Annotated[int, typer.Option(help="How many to list.")] = 20,
    status: Annotated[str | None, typer.Option(help="Filter: ok, degraded, failed.")] = None,
    traces_dir: Annotated[Path, typer.Option("--dir", help="Where traces are stored.")] = DEFAULT_ROOT,
) -> None:
    """List stored traces, most recent first."""
    index = TraceStore(traces_dir).index
    rows = index.recent(limit=limit, status=status)
    if not rows:
        console.print("[dim]no traces stored yet - run something first[/dim]")
        raise typer.Exit()

    table = Table(show_header=True, header_style="bold", title=f"traces ({index.count()} total)")
    table.add_column("trace_id")
    table.add_column("when")
    table.add_column("document")
    table.add_column("status")
    table.add_column("first bad step")
    table.add_column("ms", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("score", justify="right")
    for row in rows:
        style = _STATUS_STYLE[StepStatus(row["status"])]
        table.add_row(
            row["trace_id"][:12],
            row["created_at"][:19].replace("T", " "),
            row["doc_id"][:28],
            f"[{style}]{row['status']}[/{style}]",
            row["first_degraded_step"] or "-",
            f"{row['duration_ms']:.0f}" if row["duration_ms"] else "-",
            str(row["total_tokens"] or "-"),
            "-" if row["final_score"] is None else f"{row['final_score']:.1f}",
        )
    console.print(table)


def _load_trace(traces_dir: Path, trace_id: str) -> Trace:
    store = TraceStore(traces_dir)
    matches = store.index.resolve_id(trace_id)
    if not matches:
        console.print(f"[red]No trace matching '{trace_id}'.[/red]")
        raise typer.Exit(code=1)
    if len(matches) > 1:
        console.print(f"[red]'{trace_id}' is ambiguous: {', '.join(m[:12] for m in matches)}[/red]")
        raise typer.Exit(code=1)
    return store.load(matches[0])


@app.command()
def trace(
    trace_id: Annotated[str, typer.Argument(help="Trace id, or an unambiguous prefix.")],
    traces_dir: Annotated[Path, typer.Option("--dir")] = DEFAULT_ROOT,
    show_prompts: Annotated[bool, typer.Option("--prompts", help="Print the prompts sent.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Dump the raw trace.")] = False,
) -> None:
    """Inspect one stored trace, span by span."""
    loaded = _load_trace(traces_dir, trace_id)
    if as_json:
        typer.echo(json.dumps(loaded.model_dump(mode="json"), indent=2))
        raise typer.Exit()

    style = _STATUS_STYLE[loaded.status]
    console.print(
        Panel(
            f"[bold]{loaded.doc_id}[/bold]  ({loaded.source_name})\n"
            f"status   [{style}]{loaded.status.value}[/{style}]\n"
            f"backend  {loaded.backend} ({loaded.model})\n"
            f"trace_id {loaded.trace_id}   {loaded.duration_ms:.0f} ms   "
            f"{loaded.total_tokens or 0} tokens",
            title="trace",
            expand=False,
        )
    )

    for span in loaded.step_spans():
        span_style = _STATUS_STYLE[span.status]
        console.print(
            f"\n[bold]{span.name}[/bold]  [{span_style}]{span.status.value}[/{span_style}]  "
            f"{span.duration_ms:.1f} ms  confidence {_confidence_cell(span)}  "
            f"suspicion {_suspicion_cell(span.suspicion)}"
        )
        if span.attributes:
            rendered = "  ".join(f"{k}={v}" for k, v in span.attributes.items())
            console.print(f"  [dim]{rendered}[/dim]")
        if span.error_message:
            console.print(f"  [red]{span.error_type}: {span.error_message}[/red]")
        for warning in span.warnings:
            warn_style = _SEVERITY_STYLE[warning.severity]
            console.print(f"  [{warn_style}]{warning.severity.value:5} {warning.code}[/{warn_style}]"
                          f" {warning.message}")
        for call in span.llm_calls:
            console.print(
                f"  [dim]llm {call.schema_name} {call.duration_ms:.0f}ms "
                f"in={call.tokens.input_tokens or '-'} out={call.tokens.output_tokens or '-'} "
                f"finish={call.finish_reason or '-'}[/dim]"
            )
            if show_prompts:
                for message in loaded.resolve(call.prompt) or []:
                    console.print(
                        f"    [cyan]{message['role']}[/cyan]: {message['content'][:600]}"
                    )


@app.command()
def blame(
    trace_id: Annotated[str, typer.Argument(help="Trace id, or an unambiguous prefix.")],
    traces_dir: Annotated[Path, typer.Option("--dir")] = DEFAULT_ROOT,
) -> None:
    """Answer 'where did this go wrong?' for one stored trace."""
    _render_blame(_load_trace(traces_dir, trace_id))


@app.command()
def stats(
    traces_dir: Annotated[Path, typer.Option("--dir")] = DEFAULT_ROOT,
) -> None:
    """Aggregate every stored trace: which step breaks, and how often."""
    index = TraceStore(traces_dir).index
    if not index.count():
        console.print("[dim]no traces stored yet[/dim]")
        raise typer.Exit()

    table = Table(show_header=True, header_style="bold", title="failures by step")
    table.add_column("step")
    table.add_column("not ok", justify="right")
    table.add_column("failed", justify="right")
    table.add_column("degraded", justify="right")
    table.add_column("avg suspicion", justify="right")
    table.add_column("avg self-confidence", justify="right")
    for row in index.failure_counts_by_step():
        table.add_row(
            row["step"],
            str(row["occurrences"]),
            str(row["failures"]),
            str(row["degraded"]),
            f"{row['avg_suspicion']:.2f}" if row["avg_suspicion"] is not None else "-",
            f"{row['avg_self_confidence']:.1f}" if row["avg_self_confidence"] is not None else "-",
        )
    console.print(table)

    frequency = index.warning_frequency(limit=12)
    if frequency:
        warn_table = Table(show_header=True, header_style="bold", title="most frequent warnings")
        warn_table.add_column("code")
        warn_table.add_column("spans", justify="right")
        for code, count in frequency:
            warn_table.add_row(code, str(count))
        console.print(warn_table)


@app.command()
def reindex(
    traces_dir: Annotated[Path, typer.Option("--dir")] = DEFAULT_ROOT,
) -> None:
    """Rebuild the SQLite index from the JSON files on disk."""
    count = TraceStore(traces_dir).reindex()
    console.print(f"indexed {count} trace(s) from {traces_dir}")


# -- Phase 3-5: diagnosis, feedback, evals, analytics ---------------------------


def _workspace(traces_dir: Path, dataset: Path | None = None):
    from obspipe.feedback.store import DEFAULT_DATASET
    from obspipe.workspace import Workspace

    return Workspace(traces_dir, dataset or DEFAULT_DATASET)


def _config_or_exit(version: str | None):
    try:
        return get_config(version)
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        raise typer.Exit(code=2) from exc


def _load_or_exit(workspace, trace_id: str):
    try:
        return workspace.load(trace_id)
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        raise typer.Exit(code=1) from exc


_HEALTH_STYLE = {
    "healthy": "green",
    "low_confidence": "yellow",
    "affected": "yellow",
    "root_cause": "bold red",
    "skipped": "dim",
}


def _render_diagnosis(diagnosis) -> None:
    pipeline = "  ->  ".join(
        f"[{_HEALTH_STYLE[a.health.value]}]{a.step.value}[/{_HEALTH_STYLE[a.health.value]}]"
        for a in diagnosis.steps
    )
    if diagnosis.verdict == "healthy":
        console.print(Panel(f"{pipeline}\n\n[green]{diagnosis.narrative}[/green]",
                            title=f"diagnosis {diagnosis.trace_id[:12]}", expand=False))
        return
    console.print(
        Panel(
            f"[bold red]{diagnosis.headline}[/bold red]   confidence {diagnosis.confidence:.0%}  "
            f"({diagnosis.method.replace('_', ' ')}, {diagnosis.computed_ms:.1f} ms)\n\n{pipeline}",
            title=f"diagnosis {diagnosis.trace_id[:12]}  {diagnosis.doc_id}",
            expand=False,
        )
    )
    for position, link in enumerate(diagnosis.chain, start=1):
        marker = "[bold red]root cause[/bold red]" if link.role == "root_cause" else "[yellow]propagation[/yellow]"
        console.print(f"\n{position}. {marker}  [bold]{link.step.value}[/bold]")
        console.print(f"   {link.finding}")
        if link.warning_codes:
            console.print(f"   [dim]codes: {', '.join(link.warning_codes)}[/dim]")
        for divergence in link.divergences[:4]:
            console.print(f"   [magenta]diff[/magenta] {divergence.describe()}")


@app.command()
def diagnose(
    trace_id: Annotated[str, typer.Argument(help="Trace id, or an unambiguous prefix.")],
    traces_dir: Annotated[Path, typer.Option("--dir")] = DEFAULT_ROOT,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Backward analysis: root cause, category and the evidence chain."""
    workspace = _workspace(traces_dir)
    diagnosis = workspace.diagnose(_load_or_exit(workspace, trace_id), refresh=True)
    if as_json:
        typer.echo(diagnosis.model_dump_json(indent=2))
        return
    _render_diagnosis(diagnosis)


@app.command("diagnose-all")
def diagnose_all(traces_dir: Annotated[Path, typer.Option("--dir")] = DEFAULT_ROOT) -> None:
    """Recompute and cache the diagnosis of every stored trace."""
    console.print(f"diagnosed {_workspace(traces_dir).diagnose_all()} trace(s)")


@app.command()
def flag(
    trace_id: Annotated[str, typer.Argument(help="Trace id, or an unambiguous prefix.")],
    reason: Annotated[str, typer.Option(help="What looked wrong.")] = "",
    traces_dir: Annotated[Path, typer.Option("--dir")] = DEFAULT_ROOT,
    dataset: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Mark a trace's output as bad; prints the automated diagnosis to confirm."""
    workspace = _workspace(traces_dir, dataset)
    _load_or_exit(workspace, trace_id)
    created, diagnosis = workspace.flag(trace_id, reason=reason)
    _render_diagnosis(diagnosis)
    console.print(
        f"\nflag [bold]{created.flag_id}[/bold] opened. Confirm with "
        f"`obspipe confirm {created.flag_id}` or override with --step/--category."
    )


@app.command()
def confirm(
    flag_id: Annotated[str, typer.Argument()],
    step: Annotated[str | None, typer.Option(help="Override the root-cause step.")] = None,
    category: Annotated[str | None, typer.Option(help="Override the failure category.")] = None,
    corrected: Annotated[
        str | None, typer.Option(help="Corrected output for the failing step, as JSON.")
    ] = None,
    notes: Annotated[str, typer.Option()] = "",
    traces_dir: Annotated[Path, typer.Option("--dir")] = DEFAULT_ROOT,
    dataset: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Confirm or override a flag's diagnosis; appends an eval case."""
    from obspipe.workspace import FlagError

    workspace = _workspace(traces_dir, dataset)
    try:
        resolved, case = workspace.resolve_flag(
            flag_id,
            step=step,
            category=category,
            corrected_output=json.loads(corrected) if corrected else None,
            notes=notes,
        )
    except (FlagError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"flag {resolved.flag_id} [bold]{resolved.status.value}[/bold]: "
        f"{case.failing_step.value} / {case.failure_category.value}\n"
        f"eval case {case.case_id} appended -> {workspace.feedback.dataset_path} "
        f"({len(workspace.feedback.cases())} case(s))"
    )


@app.command()
def flags(traces_dir: Annotated[Path, typer.Option("--dir")] = DEFAULT_ROOT) -> None:
    """List flags, newest first."""
    rows = _workspace(traces_dir).feedback.flags()
    table = Table(show_header=True, header_style="bold", title=f"flags ({len(rows)})")
    for column in ("flag", "document", "status", "diagnosed", "confirmed", "case"):
        table.add_column(column)
    for item in rows:
        table.add_row(
            item.flag_id,
            item.doc_id[:28],
            item.status.value,
            f"{item.diagnosed_step.value if item.diagnosed_step else '-'} / "
            f"{item.diagnosed_category.value if item.diagnosed_category else '-'}",
            f"{item.confirmed_step.value if item.confirmed_step else '-'} / "
            f"{item.confirmed_category.value if item.confirmed_category else '-'}",
            item.eval_case_id or "-",
        )
    console.print(table)


@eval_app.command("list")
def eval_list(
    traces_dir: Annotated[Path, typer.Option("--dir")] = DEFAULT_ROOT,
    dataset: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """The eval dataset, with each case's latest result."""
    workspace = _workspace(traces_dir, dataset)
    latest = workspace.feedback.latest_results()
    table = Table(show_header=True, header_style="bold", title=f"eval dataset ({workspace.feedback.dataset_path})")
    for column in ("case", "document", "step", "category", "codes", "latest"):
        table.add_column(column)
    for case in workspace.feedback.cases():
        result = latest.get(case.case_id)
        verdict = "-" if result is None else ("[green]fixed[/green]" if result.passed else "[red]failing[/red]")
        table.add_row(case.case_id, case.doc_id[:28], case.failing_step.value,
                      case.failure_category.value, ", ".join(case.failure_codes) or "-", verdict)
    console.print(table)


@eval_app.command("run")
def eval_run(
    version: Annotated[str, typer.Option("--pipeline-version", help="Pipeline version to test.")] = CURRENT_VERSION,
    label: Annotated[str, typer.Option()] = "",
    traces_dir: Annotated[Path, typer.Option("--dir")] = DEFAULT_ROOT,
    dataset: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Replay every eval case against a pipeline version."""
    _config_or_exit(version)
    workspace = _workspace(traces_dir, dataset)
    if not workspace.feedback.cases():
        console.print("[dim]eval dataset is empty - flag and confirm a bad output first[/dim]")
        raise typer.Exit()
    run_record = workspace.run_evals(version, label=label)
    cases = {c.case_id: c for c in workspace.feedback.cases()}
    for result in run_record.results:
        case = cases[result.case_id]
        style = "green" if result.passed else "red"
        console.print(f"  [{style}]{'PASS' if result.passed else 'FAIL'}[/{style}] "
                      f"{case.doc_id[:28]:28} {case.failure_category.value:26} {result.detail}")
    console.print(f"\n{version}: [bold]{run_record.passed}/{run_record.total}[/bold] known issues resolved")


@eval_app.command("trend")
def eval_trend(traces_dir: Annotated[Path, typer.Option("--dir")] = DEFAULT_ROOT) -> None:
    """Known issues resolved, run by run."""
    runs = _workspace(traces_dir).feedback.runs()
    for row in runs:
        rate = row["passed"] / row["total"] if row["total"] else 0
        bar = "#" * round(rate * 30)
        console.print(f"{row['started_at'][:19]}  {row['pipeline_version']:3}  "
                      f"{row['passed']:>3}/{row['total']:<3} {bar:30} {rate:.0%}")


@app.command()
def analytics(
    traces_dir: Annotated[Path, typer.Option("--dir")] = DEFAULT_ROOT,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Failure analytics: categories, root-cause steps, rates, time to root cause."""
    data = _workspace(traces_dir).analytics()
    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return
    totals = data["totals"]
    console.print(
        f"[bold]{totals['traces']}[/bold] traces, {totals['failures']} diagnosed failures "
        f"({totals['failure_rate']:.0%}), {totals['flags']} flags, {totals['eval_cases']} eval cases"
    )
    table = Table(show_header=True, header_style="bold", title="failure categories")
    table.add_column("category")
    table.add_column("count", justify="right")
    for row in data["by_category"]:
        table.add_row(row["label"], str(row["count"]))
    console.print(table)
    steps = Table(show_header=True, header_style="bold", title="root cause by step")
    steps.add_column("step")
    steps.add_column("count", justify="right")
    for row in data["by_root_step"]:
        steps.add_row(row["step"], str(row["count"]))
    console.print(steps)
    ttrc = data["time_to_root_cause"]
    if ttrc["automated_ms_mean"] is not None:
        console.print(f"automated diagnosis: mean {ttrc['automated_ms_mean']:.1f} ms over "
                      f"{ttrc['automated_samples']} traces")
    if ttrc["human_seconds_mean"] is not None:
        console.print(f"human flag-to-confirmation: mean {ttrc['human_seconds_mean']:.0f} s over "
                      f"{ttrc['human_samples']} flags")


@app.command()
def versions() -> None:
    """The pipeline versions regression runs can target."""
    for config in PIPELINE_VERSIONS.values():
        marker = " (current)" if config.version == CURRENT_VERSION else ""
        console.print(f"[bold]{config.version}[/bold]{marker}  max_chars={config.max_chars}  "
                      f"guards={sorted(config.guards) or '-'}\n    {config.notes}")


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8765,
    traces_dir: Annotated[Path, typer.Option("--dir")] = DEFAULT_ROOT,
    dataset: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Start the visual trace explorer."""
    from obspipe.web.server import serve as start

    start(_workspace(traces_dir, dataset), host=host, port=port,
          log=lambda m: console.print(m))


@demo_app.command("generate")
def demo_generate(
    directory: Annotated[Path, typer.Option("--out")] = Path("data/demo"),
) -> None:
    """Write the 50-document corpus with meta and golden sidecars."""
    from obspipe.demo.generator import write_corpus

    docs = write_corpus(directory)
    failing = [d for d in docs if d.expected_failure]
    console.print(f"wrote {len(docs)} documents to {directory} ({len(failing)} designed to fail)")


@demo_app.command("seed")
def demo_seed(
    corpus: Annotated[Path, typer.Option("--corpus")] = Path("data/demo"),
    traces_dir: Annotated[Path, typer.Option("--dir")] = DEFAULT_ROOT,
    dataset: Annotated[Path | None, typer.Option()] = None,
    reset: Annotated[
        bool, typer.Option("--reset", help="Delete stored traces, both databases and the dataset first.")
    ] = False,
) -> None:
    """Run three batches (v1, v2, v3), flag first-batch failures, replay evals."""
    from obspipe.demo.seed import reset as wipe
    from obspipe.demo.seed import seed as run_seed
    from obspipe.feedback.store import DEFAULT_DATASET

    if not corpus.is_dir():
        console.print(f"[red]{corpus} does not exist - run `obspipe demo generate` first[/red]")
        raise typer.Exit(code=1)
    if reset:
        removed = wipe(traces_dir, dataset or DEFAULT_DATASET)
        console.print(f"[dim]reset: removed {len(removed)} file(s)[/dim]")
    try:
        run_seed(_workspace(traces_dir, dataset), corpus, log=lambda m: console.print(m))
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":  # pragma: no cover
    app()
