"""Backend wrapper that turns every model call into a span.

This is where the prompt, the unparsed reply and the token counts are captured.
They are invisible to a step by construction: ``with_structured_output`` hands
back a validated object and discards the ``AIMessage`` that carried them. So
the capture happens one level down, in a wrapper that presents exactly the same
interface - meaning no step function changes to gain full LLM observability.
"""

from __future__ import annotations

import re
import time
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from obspipe.llm.backends import LLMBackend
from obspipe.tracing.models import LLMCall, TokenUsage, utcnow
from obspipe.tracing.tracer import Tracer


_DOCUMENT_BLOCK_RE = re.compile(r"(<document>\n)(.*?)(\n</document>)", re.DOTALL)


def _dehydrate(content: str, tracer: Tracer) -> str:
    """Replace the document body inside a prompt with a payload reference.

    Every step's prompt embeds the same document. Stored verbatim that is three
    copies per trace on top of the one the intake span already holds. The
    surrounding instructions - the part that actually differs between steps -
    stay inline and readable.
    """
    return _DOCUMENT_BLOCK_RE.sub(
        lambda m: m.group(1) + tracer.store_text(m.group(2)) + m.group(3), content
    )


def _messages_of(prompt: Any, tracer: Tracer) -> Any:
    if isinstance(prompt, PromptValue):
        return [
            {"role": message.type, "content": _dehydrate(str(message.content), tracer)}
            for message in prompt.to_messages()
        ]
    if isinstance(prompt, BaseMessage):
        return [{"role": prompt.type, "content": _dehydrate(str(prompt.content), tracer)}]
    if isinstance(prompt, list):
        return [_messages_of(item, tracer) for item in prompt]
    return str(prompt)


class ObservedBackend:
    """Wraps any :class:`LLMBackend`, recording each call into the active span."""

    def __init__(self, wrapped: LLMBackend, tracer: Tracer) -> None:
        self._wrapped = wrapped
        self._tracer = tracer
        self.name = wrapped.name
        self.model = getattr(wrapped, "model", None)

    @property
    def wrapped(self) -> LLMBackend:
        return self._wrapped

    def structured(
        self, schema: type[BaseModel], include_raw: bool = False
    ) -> Runnable[Any, Any]:
        inner = self._wrapped.structured(schema, include_raw=True)

        def _invoke(prompt: Any) -> Any:
            call = LLMCall(
                schema_name=schema.__name__,
                model=self.model,
                backend=self.name,
                started_at=utcnow(),
                prompt=self._tracer.store(_messages_of(prompt, self._tracer)),
            )
            clock = time.perf_counter()
            try:
                result = inner.invoke(prompt)
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
                call.duration_ms = round((time.perf_counter() - clock) * 1000, 3)
                call.parsing_error = f"{type(exc).__name__}: {exc}"
                self._tracer.record_llm_call(call)
                raise
            call.duration_ms = round((time.perf_counter() - clock) * 1000, 3)

            raw = result.get("raw") if isinstance(result, dict) else None
            parsed = result.get("parsed") if isinstance(result, dict) else result
            parsing_error = result.get("parsing_error") if isinstance(result, dict) else None

            if raw is not None:
                call.raw_response = self._tracer.store(str(raw.content) or _tool_args(raw))
                call.tokens = TokenUsage.from_usage_metadata(getattr(raw, "usage_metadata", None))
                call.finish_reason = (getattr(raw, "response_metadata", {}) or {}).get(
                    "finish_reason"
                )
            if parsing_error is not None:
                call.parsing_error = str(parsing_error)
            if parsed is not None:
                call.parsed = self._tracer.store(parsed)
                call.self_confidence = getattr(parsed, "self_confidence", None)
                call.confidence_reason = getattr(parsed, "confidence_reason", None)

            self._tracer.record_llm_call(call)
            return result if include_raw else parsed

        return RunnableLambda(_invoke, name=f"observed:{schema.__name__}")


def _tool_args(message: Any) -> Any:
    """Recover the reply when the model answered with a tool call, not text."""
    calls = getattr(message, "tool_calls", None)
    if calls:
        return [{"name": c.get("name"), "args": c.get("args")} for c in calls]
    return ""
