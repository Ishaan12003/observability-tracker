"""Loading the fixture corpus.

Each document ships with a ``<filename>.meta.json`` sidecar stating what it is
and how it is meant to break. Phase 1 uses those expectations as test
assertions; Phase 4 promotes the same records into the evaluation dataset that
human feedback grows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, NamedTuple

from obspipe.models.document import RawDocument
from obspipe.models.pipeline import DocumentExpectation

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CORPUS_DIR = _REPO_ROOT / "tests" / "fixtures" / "documents"
"""The committed failure-mode corpus.

These eight documents are part of the codebase, not user data: each ships a
``.meta.json`` sidecar declaring the failure it is designed to provoke, and the
test suite asserts against them. They live under ``tests/`` for that reason.

``data/`` is the opposite - a drop zone for whatever documents *you* want to run
through the pipeline. Its contents are deliberately not tracked in git.
"""

INBOX_DIR = _REPO_ROOT / "data" / "documents"
"""Where a user drops their own documents. Untracked, may not exist."""

DEMO_CORPUS_DIR = _REPO_ROOT / "data" / "demo"
"""The 50-document demo scenario, written by ``obspipe demo generate``.

Generated rather than committed, so it is untracked like the rest of ``data/``.
"""

_META_SUFFIX = ".meta.json"
_SIDECAR_SUFFIXES = (_META_SUFFIX, ".golden.json")

_BINARY_SUFFIXES = frozenset(
    {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".png", ".jpg", ".jpeg",
     ".gif", ".zip", ".rtf"}
)
"""Containers the corpus cannot read as text.

Dropping a PDF next to the fixtures should not turn it into a fixture: it has
no sidecar, and decoding it as UTF-8 produces object dictionaries that three
LLM calls will dutifully process. Convert it to text first and add the result.
"""


class CorpusEntry(NamedTuple):
    path: Path
    document: RawDocument
    expectation: DocumentExpectation | None


def document_paths(directory: Path | str = DEFAULT_CORPUS_DIR) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")
    return sorted(
        p
        for p in root.iterdir()
        if p.is_file()
        and not p.name.endswith(_SIDECAR_SUFFIXES)
        and p.suffix.lower() not in _BINARY_SUFFIXES
        and not p.name.startswith(".")
    )


def load_entry(path: Path | str) -> CorpusEntry:
    path = Path(path)
    return CorpusEntry(
        path=path,
        document=RawDocument.from_path(path),
        expectation=DocumentExpectation.load(path),
    )


def iter_corpus(directory: Path | str = DEFAULT_CORPUS_DIR) -> Iterator[CorpusEntry]:
    for path in document_paths(directory):
        yield load_entry(path)


def load_corpus(directory: Path | str = DEFAULT_CORPUS_DIR) -> list[CorpusEntry]:
    return list(iter_corpus(directory))
