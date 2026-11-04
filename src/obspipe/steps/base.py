"""Shared step plumbing.

A step is a plain function ``(typed input, RunContext, backend) -> typed
output``. It reads nothing global, writes nothing global, and raises
:class:`StepFailure` when it cannot produce a valid output at all. Anything
short of that is reported as a warning on the output instead, so a degraded
run still yields something the tracer can inspect.
"""

from __future__ import annotations

from obspipe.models.common import Severity, StepName, StepWarning
from obspipe.models.pipeline import RunContext


class StepError(RuntimeError):
    """Base class for step-level errors."""


class StepFailure(StepError):
    """A step could not produce output.

    Carries the step identity and a machine code so the orchestrator can
    record it without string-matching an exception message.
    """

    def __init__(self, step: StepName, code: str, message: str) -> None:
        super().__init__(message)
        self.step = step
        self.code = code
        self.message = message

    def as_warning(self) -> StepWarning:
        return StepWarning(code=self.code, message=self.message, severity=Severity.ERROR)


def warn(code: str, message: str, *, field_path: str | None = None) -> StepWarning:
    return StepWarning(code=code, message=message, severity=Severity.WARN, field_path=field_path)


def info(code: str, message: str, *, field_path: str | None = None) -> StepWarning:
    return StepWarning(code=code, message=message, severity=Severity.INFO, field_path=field_path)


def error(code: str, message: str, *, field_path: str | None = None) -> StepWarning:
    return StepWarning(code=code, message=message, severity=Severity.ERROR, field_path=field_path)


GUARD_DROP_UNGROUNDED = "drop_ungrounded_spans"
"""Extraction discards any span it cannot find in the source instead of passing
it downstream with an error attached."""

GUARD_STRIP_UNSUPPORTED = "strip_unsupported_summary_figures"
"""Summarization removes a money figure that matches nothing in the document
and raises an unresolved question in its place."""

KNOWN_GUARDS = frozenset({GUARD_DROP_UNGROUNDED, GUARD_STRIP_UNSUPPORTED})


def guard_enabled(ctx: RunContext, guard: str) -> bool:
    """Whether the run's pipeline version switched ``guard`` on."""
    return guard in (ctx.extra.get("guards") or ())
