from __future__ import annotations

from pathlib import Path

import pytest

from obspipe.corpus import DEFAULT_CORPUS_DIR, load_corpus, load_entry
from obspipe.llm.backends import FakeBackend
from obspipe.models.document import RawDocument
from obspipe.models.pipeline import RunContext

CORPUS_DIR = DEFAULT_CORPUS_DIR


@pytest.fixture
def backend() -> FakeBackend:
    """Deterministic offline backend - no network, no cost, no flake."""
    return FakeBackend()


@pytest.fixture
def ctx() -> RunContext:
    return RunContext(doc_id="test-doc", backend="fake", model="fake-rules-v1")


@pytest.fixture
def corpus():
    return load_corpus(CORPUS_DIR)


@pytest.fixture
def doc_of():
    def _load(name: str) -> RawDocument:
        return load_entry(Path(CORPUS_DIR) / name).document

    return _load


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    """Committed golden traces - the regression guard on trace *shape*.

    Real traces are gitignored; these are normalized (no ids, no timings) so
    they diff cleanly in review and only move when the format genuinely does.
    """
    path = Path(__file__).parent / "fixtures" / "traces"
    path.mkdir(parents=True, exist_ok=True)
    return path
