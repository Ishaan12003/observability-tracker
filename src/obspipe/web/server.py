"""The trace explorer's HTTP server: a JSON API plus three static files.

Standard library only. The explorer is a local debugging tool over local files,
so a framework would add a dependency without adding a capability; every route
below is a thin call into :class:`~obspipe.workspace.Workspace`, the same object
the CLI uses.
"""

from __future__ import annotations

import json
import re
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from obspipe.analysis.taxonomy import CATEGORY_LABELS, FailureCategory
from obspipe.models.common import StepName
from obspipe.pipeline import CURRENT_VERSION, PIPELINE_VERSIONS
from obspipe.tracing.trace import Trace
from obspipe.workspace import FlagError, Workspace

STATIC_DIR = Path(__file__).parent / "static"
MAX_TEXT = 24_000
"""Longest string sent to the browser per field; the trace file keeps the rest."""

_CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                  ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}


def _clip_deep(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + f"\n... [{len(value) - MAX_TEXT} more chars]"
    if isinstance(value, dict):
        return {k: _clip_deep(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip_deep(v) for v in value]
    return value


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


# -- payload builders -----------------------------------------------------------


def trace_detail(workspace: Workspace, trace: Trace) -> dict:
    diagnosis = workspace.diagnose(trace)
    spans = []
    for span in trace.step_spans():
        calls = []
        for call in span.llm_calls:
            calls.append({
                "schema_name": call.schema_name,
                "model": call.model,
                "duration_ms": call.duration_ms,
                "prompt": trace.resolve(call.prompt),
                "raw_response": trace.resolve(call.raw_response),
                "parsed": trace.resolve(call.parsed),
                "parsing_error": call.parsing_error,
                "finish_reason": call.finish_reason,
                "tokens": call.tokens.model_dump(),
                "self_confidence": call.self_confidence,
                "confidence_reason": call.confidence_reason,
            })
        spans.append({
            "span_id": span.span_id,
            "step": span.step.value if span.step else span.name,
            "status": span.status.value,
            "duration_ms": span.duration_ms,
            "inputs": trace.resolve(span.inputs),
            "output": trace.resolve(span.output),
            "warnings": [w.model_dump(mode="json") for w in span.warnings],
            "error_type": span.error_type,
            "error_message": span.error_message,
            "self_confidence": span.self_confidence,
            "confidence_reason": span.confidence_reason,
            "computed_confidence": span.computed_confidence,
            "suspicion": span.suspicion,
            "attributes": span.attributes,
            "llm_calls": calls,
        })
    return _clip_deep(_jsonable({
        "trace": {
            "trace_id": trace.trace_id,
            "doc_id": trace.doc_id,
            "source_name": trace.source_name,
            "backend": trace.backend,
            "model": trace.model,
            "status": trace.status.value,
            "started_at": trace.started_at.isoformat(),
            "duration_ms": trace.duration_ms,
            "total_tokens": trace.total_tokens,
            "tags": trace.tags,
            "has_golden": workspace.golden_for(trace) is not None,
        },
        "final_output": trace.result.summary.model_dump(mode="json") if trace.result and trace.result.summary else None,
        "spans": spans,
        "diagnosis": diagnosis.model_dump(mode="json"),
        "diffs": workspace.diffs(trace),
        "flags": [f.model_dump(mode="json") for f in workspace.feedback.flags(trace.trace_id)],
    }))


def trace_list(workspace: Workspace, query: dict[str, str]) -> dict:
    include_evals = query.get("include_evals") == "1"
    diagnoses = {r["trace_id"]: r for r in workspace.feedback.diagnosis_rows()}
    flagged = {f.trace_id: f.status.value for f in workspace.feedback.flags()}
    rows = []
    for row in reversed(workspace.traces.index.all_rows()):
        batch = row["batch"] or ""
        if batch.startswith("eval-") and not include_evals:
            continue
        if query.get("batch") and batch != query["batch"]:
            continue
        diagnosis = diagnoses.get(row["trace_id"])
        verdict = diagnosis["verdict"] if diagnosis else None
        if query.get("verdict") and verdict != query["verdict"]:
            continue
        needle = (query.get("q") or "").casefold()
        if needle and needle not in row["doc_id"].casefold() and needle not in row["trace_id"]:
            continue
        rows.append({
            "trace_id": row["trace_id"],
            "doc_id": row["doc_id"],
            "created_at": row["created_at"],
            "status": row["status"],
            "batch": batch or None,
            "pipeline_version": row["pipeline_version"],
            "duration_ms": row["duration_ms"],
            "verdict": verdict,
            "root_step": diagnosis["root_step"] if diagnosis else None,
            "category": diagnosis["category"] if diagnosis else None,
            "flag": flagged.get(row["trace_id"]),
        })
    batches = sorted({(r["batch"] or "") for r in workspace.traces.index.all_rows()} - {""})
    return {"traces": rows, "batches": [b for b in batches if include_evals or not b.startswith("eval-")]}


def evals_payload(workspace: Workspace) -> dict:
    latest = workspace.feedback.latest_results()
    cases = []
    for case in workspace.feedback.cases():
        result = latest.get(case.case_id)
        cases.append({
            **case.model_dump(mode="json", exclude={"input"}),
            "input": {**case.input.model_dump(mode="json", exclude={"content"}),
                      "chars": len(case.input.content)},
            "latest": result.model_dump(mode="json") if result else None,
        })
    runs = []
    for row in workspace.feedback.runs():
        runs.append({**dict(row), "results": [
            r.model_dump(mode="json") for r in workspace.feedback.run_results(row["run_id"])]})
    return {"cases": cases, "runs": runs, "dataset_path": str(workspace.feedback.dataset_path)}


def meta_payload() -> dict:
    return {
        "steps": [s.value for s in StepName],
        "categories": [{"value": c.value, "label": CATEGORY_LABELS[c]} for c in FailureCategory],
        "versions": [c.model_dump(mode="json") for c in PIPELINE_VERSIONS.values()],
        "current_version": CURRENT_VERSION,
    }


# -- routing ----------------------------------------------------------------------


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def make_handler(workspace: Workspace) -> type[BaseHTTPRequestHandler]:
    routes: list[tuple[str, re.Pattern, Callable[..., Any]]] = []

    def route(method: str, pattern: str):
        def register(func):
            routes.append((method, re.compile(f"^{pattern}$"), func))
            return func
        return register

    def load(trace_id: str) -> Trace:
        try:
            return workspace.load(trace_id)
        except KeyError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, exc.args[0]) from exc

    @route("GET", r"/api/meta")
    def _meta(query, body):
        return meta_payload()

    @route("GET", r"/api/traces")
    def _traces(query, body):
        return trace_list(workspace, query)

    @route("GET", r"/api/traces/(?P<trace_id>[0-9a-f]+)")
    def _trace(query, body, trace_id):
        return trace_detail(workspace, load(trace_id))

    @route("POST", r"/api/traces/(?P<trace_id>[0-9a-f]+)/flag")
    def _flag(query, body, trace_id):
        load(trace_id)
        flag, diagnosis = workspace.flag(trace_id, reason=str(body.get("reason", "")))
        return {"flag": flag.model_dump(mode="json"), "diagnosis": diagnosis.model_dump(mode="json")}

    @route("POST", r"/api/flags/(?P<flag_id>flag_[0-9a-f]+)/resolve")
    def _resolve(query, body, flag_id):
        corrected = body.get("corrected_output")
        if isinstance(corrected, str):
            corrected = json.loads(corrected) if corrected.strip() else None
        try:
            flag, case = workspace.resolve_flag(
                flag_id,
                step=body.get("step") or None,
                category=body.get("category") or None,
                corrected_output=corrected,
                notes=str(body.get("notes", "")),
            )
        except FlagError as exc:
            raise ApiError(HTTPStatus.CONFLICT, str(exc)) from exc
        return {"flag": flag.model_dump(mode="json"), "case": case.model_dump(mode="json", exclude={"input"}),
                "dataset_size": len(workspace.feedback.cases())}

    @route("POST", r"/api/flags/(?P<flag_id>flag_[0-9a-f]+)/dismiss")
    def _dismiss(query, body, flag_id):
        try:
            return {"flag": workspace.dismiss_flag(flag_id, str(body.get("notes", ""))).model_dump(mode="json")}
        except FlagError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, str(exc)) from exc

    @route("GET", r"/api/flags")
    def _flags(query, body):
        return {"flags": [f.model_dump(mode="json") for f in workspace.feedback.flags()]}

    @route("GET", r"/api/evals")
    def _evals(query, body):
        return evals_payload(workspace)

    @route("POST", r"/api/evals/run")
    def _run_evals(query, body):
        version = body.get("version") or CURRENT_VERSION
        if version not in PIPELINE_VERSIONS:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"unknown pipeline version {version}")
        if not workspace.feedback.cases():
            raise ApiError(HTTPStatus.CONFLICT, "The eval dataset is empty.")
        run = workspace.run_evals(version, label=str(body.get("label", "")))
        return {"run_id": run.run_id, "passed": run.passed, "total": run.total, "version": version}

    @route("GET", r"/api/analytics")
    def _analytics(query, body):
        return workspace.analytics()

    class Handler(BaseHTTPRequestHandler):
        server_version = "obspipe-explorer"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
            pass

        def _send(self, status: HTTPStatus, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, status: HTTPStatus, data: Any) -> None:
            self._send(status, json.dumps(data, default=str).encode("utf-8"), "application/json")

        def _dispatch(self, method: str) -> None:
            parsed = urlparse(self.path)
            if method == "GET" and not parsed.path.startswith("/api/"):
                self._static(parsed.path)
                return
            query = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
            body: dict = {}
            if method == "POST":
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "body is not JSON"})
                    return
            for verb, pattern, func in routes:
                match = pattern.match(parsed.path)
                if verb == method and match:
                    try:
                        self._json(HTTPStatus.OK, func(query, body, **match.groupdict()))
                    except ApiError as exc:
                        self._json(exc.status, {"error": exc.message})
                    except (ValueError, KeyError) as exc:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
                        traceback.print_exc()
                        self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})
                    return
            self._json(HTTPStatus.NOT_FOUND, {"error": f"no route {method} {parsed.path}"})

        def _static(self, path: str) -> None:
            name = "index.html" if path in ("", "/") else path.lstrip("/").removeprefix("static/")
            target = (STATIC_DIR / name).resolve()
            if STATIC_DIR.resolve() not in target.parents or not target.is_file():
                target = STATIC_DIR / "index.html"  # single-page app: unknown paths get the shell
            self._send(HTTPStatus.OK, target.read_bytes(),
                       _CONTENT_TYPES.get(target.suffix, "application/octet-stream"))

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

    return Handler


def build_server(workspace: Workspace, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(workspace))


def serve(workspace: Workspace, host: str = "127.0.0.1", port: int = 8765,
          log: Callable[[str], None] = print) -> None:
    server = build_server(workspace, host, port)
    log(f"trace explorer on http://{host}:{server.server_address[1]}  (ctrl-c to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("stopped")
    finally:
        server.server_close()
