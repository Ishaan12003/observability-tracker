"""Content-addressed payload store.

A trace records what went into and out of every step. Done naively the
normalized document text is written three times - once as extraction's input,
once as classification's, once as summarization's - and a 20k-char contract
produces a 60KB trace before anything interesting is in it.

Anything above a size threshold is therefore hashed and stored once in
``Trace.payloads``; spans keep a ``@sha256:...`` reference. Nothing is lost, so
a stored trace can still be replayed exactly, and a human reading the JSON sees
the small fields inline and one copy of the big ones at the bottom.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel

REF_PREFIX = "@sha256:"
INLINE_LIMIT = 2_048
"""Values rendering to more than this many characters are stored by reference."""


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def is_ref(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(REF_PREFIX)


EMBEDDED_REF_RE = re.compile(re.escape(REF_PREFIX) + r"[0-9a-f]{32}")
"""Matches a reference sitting *inside* a larger string.

A rendered prompt is mostly boilerplate wrapped around the document. Replacing
just the document body with a reference keeps one copy per trace instead of one
per LLM call, while the prompt still reads as a prompt.
"""


class PayloadStore(BaseModel):
    """Hash -> serialized payload, shared across every span in one trace."""

    blobs: dict[str, str] = {}

    def put(self, value: Any, *, inline_limit: int = INLINE_LIMIT) -> Any:
        """Store ``value``, returning it inline or as a reference.

        Small values are returned unchanged so the trace stays readable. Large
        ones are serialized once, deduplicated by content hash, and replaced
        with a reference.
        """
        jsonable = _to_jsonable(value)
        rendered = json.dumps(jsonable, ensure_ascii=False, sort_keys=True)
        if len(rendered) <= inline_limit:
            return jsonable
        key = digest(rendered)
        self.blobs.setdefault(key, rendered)
        return f"{REF_PREFIX}{key}"

    def put_text(self, text: str, *, inline_limit: int = INLINE_LIMIT) -> str:
        """Store a bare string, returning it verbatim or as a reference."""
        if len(text) <= inline_limit:
            return text
        key = digest(text)
        self.blobs.setdefault(key, json.dumps(text, ensure_ascii=False))
        return f"{REF_PREFIX}{key}"

    def expand_text(self, text: str) -> str:
        """Put every embedded reference in ``text`` back."""
        return EMBEDDED_REF_RE.sub(lambda m: str(self.get(m.group(0))), text)

    def get(self, value: Any) -> Any:
        """Resolve a reference back to its value; pass anything else through."""
        if not is_ref(value):
            return value
        key = value[len(REF_PREFIX) :]
        if key not in self.blobs:
            raise KeyError(f"payload {key} missing from this trace")
        return json.loads(self.blobs[key])

    def resolve_deep(self, value: Any) -> Any:
        """Resolve references anywhere inside a nested structure.

        Handles both a value that *is* a reference and a string that merely
        *contains* one, which is how a prompt carries its document body.
        """
        if is_ref(value):
            return self.resolve_deep(self.get(value))
        if isinstance(value, str):
            return self.expand_text(value) if REF_PREFIX in value else value
        if isinstance(value, dict):
            return {k: self.resolve_deep(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.resolve_deep(v) for v in value]
        return value

    @property
    def total_bytes(self) -> int:
        return sum(len(blob) for blob in self.blobs.values())
