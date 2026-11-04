"""Ground truth, step projections, and the comparison between them.

A step's full output is large and mostly irrelevant to whether it was right.
:func:`project` reduces each step's output to the handful of normalized fields
that decide correctness - parties as names, dates as ISO strings, amounts as
``"1200.00 USD"`` - so that ground truth, a human correction and a fresh rerun
all live in one comparable shape. Every expected-vs-actual check in the system
(the diff view, root-cause selection, regression evals) goes through
:func:`compare`.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from obspipe.models.common import StepName
from obspipe.parsing import parse_date, parse_money

GOLDEN_SUFFIX = ".golden.json"


def money_key(value: Any, currency: Any) -> str | None:
    """``(1200, 'USD')`` -> ``'1200.00 USD'``. The one canonical money form."""
    if value is None:
        return None
    amount = Decimal(str(value)).quantize(Decimal("0.01"))
    return f"{amount} {currency or '?'}"


def money_text_key(text: str | None) -> str | None:
    if not text:
        return None
    value, currency = parse_money(text)
    return money_key(value, currency)


def date_key(text: str | None) -> str | None:
    if not text:
        return None
    return parse_date(text) or text.strip()


def _names(items: Iterable[str]) -> list[str]:
    seen: dict[str, str] = {}
    for item in items:
        name = (item or "").strip()
        if name and name.casefold() not in seen:
            seen[name.casefold()] = name
    return sorted(seen.values(), key=str.casefold)


# -- projections ---------------------------------------------------------


def project(step: StepName, output: dict | None, warning_codes: list[str] | None = None) -> dict:
    """The correctness-relevant view of one step's output."""
    if output is None:
        return {}
    codes = set(warning_codes or [])
    if step is StepName.INTAKE:
        return {
            "truncated": bool(output.get("truncated")),
            "char_count": output.get("char_count"),
            "original_char_count": output.get("original_char_count"),
            "high_noise": "high_noise_ratio" in codes,
            "sections": len(output.get("sections") or []),
        }
    if step is StepName.EXTRACTION:
        entities = output.get("entities") or {}
        amounts = entities.get("amounts") or []
        total = next((a for a in amounts if a.get("kind") == "total"), None)
        return {
            "parties": _names(p.get("name", "") for p in entities.get("parties") or []),
            "dates": sorted(
                {d.get("iso_date") or d.get("raw_text", "") for d in entities.get("dates") or []}
            ),
            "amounts": sorted(
                {k for a in amounts if (k := money_key(a.get("value"), a.get("currency")))}
            ),
            "total": money_key(total.get("value"), total.get("currency")) if total else None,
        }
    if step is StepName.CLASSIFICATION:
        scores = {s["doc_type"]: s["score"] for s in output.get("scores") or []}
        return {
            "doc_type": output.get("doc_type"),
            "margin": output.get("margin"),
            "scores": scores,
        }
    if step is StepName.SUMMARIZATION:
        specific = output.get("type_specific") or {}
        view: dict[str, Any] = {
            "doc_type": output.get("doc_type"),
            "headline": output.get("headline"),
            "unresolved_questions": len(output.get("unresolved_questions") or []),
        }
        if "effective_date" in specific:
            view["effective_date"] = date_key(specific.get("effective_date"))
        if "total_stated" in specific:
            view["total_stated"] = money_text_key(specific.get("total_stated"))
        if "parties" in specific:
            view["parties"] = _names(specific.get("parties") or [])
        return view
    return {}


# -- comparison ----------------------------------------------------------


class Divergence(BaseModel):
    """One place where a step's output differs from what it should have been."""

    model_config = ConfigDict(frozen=True)

    path: str
    kind: str
    """``mismatch`` (wrong value), ``missing`` (expected item absent) or
    ``unexpected`` (item present that should not be)."""
    expected: Any = None
    actual: Any = None

    def describe(self) -> str:
        if self.kind == "missing":
            return f"{self.path}: expected {self.expected!r}, which is missing"
        if self.kind == "unexpected":
            return f"{self.path}: {self.actual!r} should not be there"
        return f"{self.path}: expected {self.expected!r}, got {self.actual!r}"


def _norm(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().casefold()
    return value


def compare(expected: dict | None, actual: dict | None) -> list[Divergence]:
    """Field-by-field difference, over the keys ``expected`` names only.

    Ground truth is allowed to be partial: a key it leaves out is not checked.
    A scalar may be given as ``{"any_of": [...]}`` when more than one answer is
    defensible - a memo that is honestly both report and correspondence.
    """
    if not expected:
        return []
    actual = actual or {}
    found: list[Divergence] = []
    for key, want in expected.items():
        got = actual.get(key)
        if isinstance(want, dict) and "any_of" in want:
            options = [_norm(v) for v in want["any_of"]]
            if _norm(got) not in options:
                found.append(Divergence(path=key, kind="mismatch", expected=want["any_of"], actual=got))
            continue
        if isinstance(want, list):
            have = {_norm(v): v for v in (got or [])}
            need = {_norm(v): v for v in want}
            for norm, original in need.items():
                if norm not in have:
                    found.append(Divergence(path=key, kind="missing", expected=original))
            for norm, original in have.items():
                if norm not in need:
                    found.append(Divergence(path=key, kind="unexpected", actual=original))
            continue
        if _norm(got) != _norm(want):
            found.append(Divergence(path=key, kind="mismatch", expected=want, actual=got))
    return found


# -- ground truth ----------------------------------------------------------


class GoldenRecord(BaseModel):
    """What each step should have produced for one document.

    Keys inside each step dict are projection keys (see :func:`project`); any
    step may be omitted, and any key within a step may be omitted.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    expected: dict[StepName, dict[str, Any]] = Field(default_factory=dict)
    notes: str = ""

    def for_step(self, step: StepName) -> dict[str, Any] | None:
        return self.expected.get(step)

    @classmethod
    def load(cls, document_path: Path | str) -> "GoldenRecord | None":
        path = Path(str(document_path) + GOLDEN_SUFFIX)
        if not path.exists():
            return None
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


class GoldenIndex:
    """Every golden record under a set of directories, by ``doc_id``."""

    def __init__(self, directories: Iterable[Path | str]) -> None:
        self._records: dict[str, GoldenRecord] = {}
        self._directories: list[Path] = []
        for directory in directories:
            self.extend(directory)

    def extend(self, directory: Path | str) -> None:
        """Load every golden sidecar under ``directory``.

        A directory that does not exist is skipped rather than raising: the
        default set spans both corpora and only one of them is committed.
        """
        root = Path(directory)
        self._directories.append(root)
        if not root.is_dir():
            return
        for path in sorted(root.glob(f"*{GOLDEN_SUFFIX}")):
            record = GoldenRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
            self._records[record.doc_id] = record

    @property
    def directories(self) -> list[Path]:
        return list(self._directories)

    def get(self, doc_id: str) -> GoldenRecord | None:
        return self._records.get(doc_id)

    def __len__(self) -> int:
        return len(self._records)
