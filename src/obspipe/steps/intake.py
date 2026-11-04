"""Step 1 - Intake.

Deterministic normalization. No LLM, no network, no randomness: given the same
bytes it always produces the same text. That matters because every downstream
step reads ``normalized_text``; if this step is not reproducible, no failure
attribution below it can be trusted.
"""

from __future__ import annotations

import re
import unicodedata

from obspipe.models.common import StepName
from obspipe.models.document import RawDocument
from obspipe.models.intake import IntakeOutput, Section
from obspipe.models.pipeline import RunContext
from obspipe.steps.base import StepFailure, info, warn
from obspipe.tracing.tracer import annotate, traced

MAX_CHARS = 12_000
"""Hard input budget. Exceeding it is a documented truncation, never a silent one."""

MIN_USEFUL_CHARS = 80
NOISE_RATIO_THRESHOLD = 0.25

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-[ \t]*\n[ \t]*(\w)")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.{1,120})$")
_NUMBERED_HEADING_RE = re.compile(r"^\s{0,3}(\d+(?:\.\d+)*)[.)]\s+(\S.{0,110})$")
_UNDERLINE_RE = re.compile(r"^\s*[=\-_]{3,}\s*$")

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ", "​": "",
}


_BINARY_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("%PDF-", "PDF"),
    ("PK\x03\x04", "ZIP container (docx/xlsx/pptx)"),
    ("\x89PNG", "PNG image"),
    ("\xff\xd8\xff", "JPEG image"),
    ("GIF8", "GIF image"),
    ("\xd0\xcf\x11\xe0", "OLE compound file (legacy doc/xls)"),
    ("{\\rtf", "RTF"),
    ("%!PS", "PostScript"),
    ("\x7fELF", "ELF binary"),
)

MIN_NUL_COUNT = 8
NUL_FRACTION_THRESHOLD = 0.001
"""Pervasive NUL bytes mean binary; one stray NUL is corruption to strip."""


def _binary_kind(text: str) -> str | None:
    """Identify content that is not text at all.

    Decoding a PDF with ``errors="replace"`` yields a printable-looking string -
    PDF syntax is mostly ASCII - so neither the replacement-character check nor
    the noise-ratio check sees anything wrong. Three LLM calls then run against
    object dictionaries and stream keywords and confidently report that the
    document has no content. Detecting the container up front turns that into
    one cheap, correct failure.
    """
    head = text[:512]
    for signature, label in _BINARY_SIGNATURES:
        if head.startswith(signature):
            return label
    nul_count = text.count("\x00")
    if nul_count >= MIN_NUL_COUNT and nul_count / max(len(text), 1) > NUL_FRACTION_THRESHOLD:
        return f"binary data ({nul_count} NUL bytes)"
    return None


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not (3 <= len(stripped) <= 90):
        return False
    if stripped.endswith((".", ",", ";")):
        return False
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    return sum(c.isupper() for c in letters) / len(letters) > 0.85


def _detect_sections(lines: list[str]) -> list[Section]:
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        heading: str | None = None
        if match := _MD_HEADING_RE.match(line):
            heading = match.group(2).strip()
        elif match := _NUMBERED_HEADING_RE.match(line):
            heading = f"{match.group(1)} {match.group(2).strip()}"
        elif _looks_like_heading(line):
            heading = line.strip()
        elif index + 1 < len(lines) and _UNDERLINE_RE.match(lines[index + 1]) and line.strip():
            heading = line.strip()
        if heading:
            starts.append((index, heading))

    sections: list[Section] = []
    for position, (line_no, heading) in enumerate(starts):
        end = starts[position + 1][0] - 1 if position + 1 < len(starts) else len(lines) - 1
        sections.append(Section(heading=heading, start_line=line_no + 1, end_line=end + 1))
    return sections


def _noise_ratio(text: str) -> float:
    if not text:
        return 0.0
    interesting = sum(1 for c in text if c.isalnum() or c.isspace() or c in ".,;:!?'\"()-/$€£₹%&@#")
    return round(1 - interesting / len(text), 4)


@traced(StepName.INTAKE)
def run_intake(
    document: RawDocument,
    ctx: RunContext,
    max_chars: int = MAX_CHARS,
) -> IntakeOutput:
    """Normalize a raw document into the text every later step reads."""
    warnings = []
    raw = document.content
    original_char_count = len(raw)

    if not raw:
        raise StepFailure(StepName.INTAKE, "empty_document", "Document content is empty.")
    if not raw.strip():
        raise StepFailure(
            StepName.INTAKE,
            "whitespace_only_document",
            f"Document contains only whitespace ({original_char_count} chars).",
        )

    binary_kind = _binary_kind(raw)
    if binary_kind is not None:
        raise StepFailure(
            StepName.INTAKE,
            "binary_content",
            f"Content is {binary_kind}, not text. Extract the text first "
            "(for a PDF: `pdftotext -layout file.pdf file.pdf.txt`).",
        )

    text = raw
    if "�" in text:
        warnings.append(
            warn(
                "replacement_characters_present",
                f"{text.count(chr(0xFFFD))} U+FFFD replacement characters - "
                "the source was decoded with the wrong encoding.",
                field_path="normalized_text",
            )
        )

    control_hits = len(_CONTROL_RE.findall(text))
    if control_hits:
        text = _CONTROL_RE.sub("", text)
        warnings.append(
            info("control_characters_stripped", f"Removed {control_hits} control characters.")
        )

    repairs = 0
    for source, replacement in _LIGATURES.items():
        count = text.count(source)
        if count:
            text = text.replace(source, replacement)
            repairs += count

    hyphen_breaks = len(_HYPHEN_BREAK_RE.findall(text))
    if hyphen_breaks:
        text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
        repairs += hyphen_breaks

    # NFC, deliberately not NFKC. Compatibility normalization rewrites
    # "3.8 x10^9/L" as "3.8 x109/L" and "SpO_2" as "SpO2" - a silent change of
    # meaning in any scientific or clinical document. The typographic
    # substitutions worth making are enumerated in _LIGATURES above.
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WS_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text).strip()

    if repairs:
        warnings.append(
            info(
                "ocr_artifacts_repaired",
                f"Rejoined {hyphen_breaks} hyphenated line-breaks and normalized "
                f"{repairs - hyphen_breaks} typographic characters.",
            )
        )

    noise = _noise_ratio(text)
    if noise > NOISE_RATIO_THRESHOLD:
        warnings.append(
            warn(
                "high_noise_ratio",
                f"{noise:.0%} of characters are unexpected symbols - likely OCR damage. "
                "Downstream extraction is unreliable.",
                field_path="normalized_text",
            )
        )

    truncated = False
    if len(text) > max_chars:
        cut = text.rfind("\n", 0, max_chars)
        text = text[: cut if cut > max_chars // 2 else max_chars]
        truncated = True
        warnings.append(
            warn(
                "document_truncated",
                f"Input reduced from {original_char_count} to {len(text)} chars "
                f"(budget {max_chars}). Entities past the cut cannot be extracted.",
                field_path="normalized_text",
            )
        )

    lines = text.split("\n")
    sections = _detect_sections(lines)
    if not sections:
        warnings.append(
            info("no_sections_detected", "No headings found; document treated as one block.")
        )

    if len(text) < MIN_USEFUL_CHARS:
        warnings.append(
            warn(
                "very_short_document",
                f"Only {len(text)} usable characters - too little context for reliable "
                "classification or summarization.",
            )
        )

    # How much of the text survived intact - nothing to do with any model.
    annotate(
        computed_confidence=1.0 - noise,
        truncated=truncated,
        char_count=len(text),
        sections=len(sections),
    )

    return IntakeOutput(
        doc_id=document.doc_id,
        source_name=document.source_name,
        content_type=document.content_type,
        normalized_text=text,
        char_count=len(text),
        word_count=len(text.split()),
        line_count=len(lines),
        sections=sections,
        truncated=truncated,
        original_char_count=original_char_count,
        ocr_artifacts_repaired=repairs,
        warnings=warnings,
    )
