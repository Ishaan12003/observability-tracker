from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ContentType(str, Enum):
    PLAIN_TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    PDF_TEXT = "application/pdf+text"
    """Text already extracted from a PDF - carries OCR artifacts."""


class DocType(str, Enum):
    CONTRACT = "contract"
    INVOICE = "invoice"
    REPORT = "report"
    CORRESPONDENCE = "correspondence"
    UNKNOWN = "unknown"


class RawDocument(BaseModel):
    """Pipeline input. Immutable once constructed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    source_name: str
    content: str
    content_type: ContentType = ContentType.PLAIN_TEXT
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_text(
        cls,
        content: str,
        source_name: str = "inline",
        content_type: ContentType = ContentType.PLAIN_TEXT,
        doc_id: str | None = None,
    ) -> RawDocument:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        return cls(
            doc_id=doc_id or f"{Path(source_name).stem}-{digest}",
            source_name=source_name,
            content=content,
            content_type=content_type,
        )

    @classmethod
    def from_path(cls, path: str | Path, doc_id: str | None = None) -> RawDocument:
        p = Path(path)
        suffix_map = {
            ".md": ContentType.MARKDOWN,
            ".pdf.txt": ContentType.PDF_TEXT,
        }
        content_type = ContentType.PLAIN_TEXT
        for suffix, ct in suffix_map.items():
            if p.name.endswith(suffix):
                content_type = ct
                break
        return cls.from_text(
            content=p.read_text(encoding="utf-8", errors="replace"),
            source_name=p.name,
            content_type=content_type,
            doc_id=doc_id or p.stem.replace(".pdf", ""),
        )
