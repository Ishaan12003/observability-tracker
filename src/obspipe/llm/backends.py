"""LLM backends exposed as LCEL Runnables.

Both backends return a ``Runnable`` that accepts whatever a ChatPromptTemplate
emits and returns a populated Pydantic model. Steps therefore always compose
the same way::

    chain = PROMPT | backend.structured(SomeSchema)
    result = chain.invoke({...})

Swapping Groq for the deterministic fake changes no step code, which is what
lets the whole pipeline be exercised offline.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from obspipe.llm import fake_engine
from obspipe.llm.schemas import (
    BaseSummaryLLMOutput,
    ClassificationLLMOutput,
    ExtractionLLMOutput,
)

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class BackendError(RuntimeError):
    """Raised when a backend cannot be constructed or a call fails hard."""


@runtime_checkable
class LLMBackend(Protocol):
    name: str
    model: str | None

    def structured(
        self, schema: type[BaseModel], include_raw: bool = False
    ) -> Runnable[Any, Any]:
        """Return a Runnable that fills ``schema``.

        With ``include_raw`` the Runnable yields ``{"raw", "parsed",
        "parsing_error"}`` instead of the bare model, so an observer can record
        the unparsed reply and its token counts. Steps never ask for it.
        """
        ...


class GroqBackend:
    """Groq chat completions via ``langchain-groq``.

    Structured output goes through tool calling, so the model is constrained by
    the schema rather than asked politely for JSON.
    """

    name = "groq"

    def __init__(
        self,
        model: str = DEFAULT_GROQ_MODEL,
        temperature: float = 0.0,
        api_key: str | None = None,
        max_retries: int = 2,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self._api_key:
            raise BackendError(
                "GROQ_API_KEY is not set. Export it, put it in .env, "
                "or run with --backend fake."
            )
        self._chat: Any | None = None

    def _client(self) -> Any:
        if self._chat is None:
            from langchain_groq import ChatGroq

            self._chat = ChatGroq(
                model=self.model,
                temperature=self.temperature,
                api_key=self._api_key,
                max_retries=self.max_retries,
                timeout=self.timeout,
            )
        return self._chat

    def structured(
        self, schema: type[BaseModel], include_raw: bool = False
    ) -> Runnable[Any, Any]:
        return self._client().with_structured_output(schema, include_raw=include_raw)


class FakeBackend:
    """Rule-based backend with injectable faults.

    Deterministic by construction: same document in, same object out. Faults
    are named in :data:`obspipe.llm.fake_engine.KNOWN_FAULTS` and let a test
    manufacture a specific bad output on demand.
    """

    name = "fake"

    def __init__(self, faults: set[str] | frozenset[str] | None = None) -> None:
        requested = frozenset(faults or ())
        unknown = requested - fake_engine.KNOWN_FAULTS
        if unknown:
            raise BackendError(
                f"Unknown fault(s) {sorted(unknown)}; known: {sorted(fake_engine.KNOWN_FAULTS)}"
            )
        self.faults = requested
        self.model = "fake-rules-v1"

    def structured(
        self, schema: type[BaseModel], include_raw: bool = False
    ) -> Runnable[Any, Any]:
        def _invoke(prompt: Any) -> Any:
            document = _document_from_prompt(prompt)
            if schema is ExtractionLLMOutput:
                parsed: BaseModel = fake_engine.fake_extraction(document, self.faults)
            elif schema is ClassificationLLMOutput:
                parsed = fake_engine.fake_classification(document, self.faults)
            elif isinstance(schema, type) and issubclass(schema, BaseSummaryLLMOutput):
                parsed = fake_engine.fake_summary(
                    document, schema, self.faults, prompt_text=_prompt_to_text(prompt)
                )
            else:
                raise BackendError(f"FakeBackend has no rules for schema {schema.__name__}")

            if not include_raw:
                return parsed
            # Shaped exactly like the Groq path so the observer needs no branch.
            # usage_metadata stays absent: the fake spends no tokens, and a
            # zero here would average into cost figures as a real free call.
            return {
                "raw": AIMessage(content=parsed.model_dump_json()),
                "parsed": parsed,
                "parsing_error": None,
            }

        return RunnableLambda(_invoke, name=f"fake:{schema.__name__}")


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, PromptValue):
        return prompt.to_string()
    if isinstance(prompt, BaseMessage):
        return str(prompt.content)
    if isinstance(prompt, list):
        return "\n".join(_prompt_to_text(item) for item in prompt)
    if isinstance(prompt, dict):
        return "\n".join(f"{k}: {v}" for k, v in prompt.items())
    return str(prompt)


def _document_from_prompt(prompt: Any) -> str:
    """Recover the document body the prompt was built around.

    The templates wrap it in ``<document>`` tags precisely so a replayed prompt
    can be reduced back to its input.
    """
    text = _prompt_to_text(prompt)
    start = text.find("<document>")
    end = text.rfind("</document>")
    if start != -1 and end > start:
        return text[start + len("<document>") : end].strip()
    return text


def get_backend(
    name: str = "fake",
    model: str | None = None,
    faults: set[str] | None = None,
) -> LLMBackend:
    """Construct a backend by name.

    ``fake`` never touches the network. ``groq`` requires ``GROQ_API_KEY``.
    """
    normalized = (name or "fake").strip().lower()
    if normalized == "fake":
        return FakeBackend(faults=faults)
    if normalized == "groq":
        return GroqBackend(model=model or DEFAULT_GROQ_MODEL)
    raise BackendError(f"Unknown backend '{name}'. Use 'fake' or 'groq'.")
