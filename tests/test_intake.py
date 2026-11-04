from __future__ import annotations

import pytest

from obspipe.models.common import Severity
from obspipe.steps.base import StepFailure
from obspipe.steps.intake import run_intake
from tests.helpers import text_doc


def codes(output) -> set[str]:
    return set(output.warning_codes())


def test_intake_is_deterministic(ctx, doc_of):
    document = doc_of("contract_no_dates.md")
    first = run_intake(document, ctx)
    second = run_intake(document, ctx)
    assert first.normalized_text == second.normalized_text
    assert first.model_dump(exclude={"warnings"}) == second.model_dump(exclude={"warnings"})


def test_empty_document_fails_the_step(ctx):
    with pytest.raises(StepFailure) as caught:
        run_intake(text_doc(""), ctx)
    assert caught.value.code == "empty_document"


def test_whitespace_only_fails_before_any_llm_call(ctx, doc_of):
    with pytest.raises(StepFailure) as caught:
        run_intake(doc_of("empty.txt"), ctx)
    assert caught.value.code == "whitespace_only_document"


def test_hyphenated_line_breaks_are_rejoined(ctx, doc_of):
    output = run_intake(doc_of("contract_ocr_noise.pdf.txt"), ctx)
    assert "Machine Works" in output.normalized_text
    assert "business day" in output.normalized_text
    assert "manufacturer's" in output.normalized_text
    assert output.ocr_artifacts_repaired > 0
    assert "ocr_artifacts_repaired" in codes(output)


def test_truncation_is_reported_not_silent(ctx, doc_of):
    output = run_intake(doc_of("contract_long_truncated.md"), ctx)
    assert output.truncated is True
    assert output.char_count < output.original_char_count
    assert "document_truncated" in codes(output)
    # The clauses carrying every date live past the cut.
    assert "TERM AND TERMINATION" not in output.normalized_text


def test_truncation_budget_is_configurable(ctx, doc_of):
    output = run_intake(doc_of("contract_long_truncated.md"), ctx, max_chars=100_000)
    assert output.truncated is False
    assert "15 January 2024" in output.normalized_text


def test_sections_are_detected_from_markdown_and_caps_headings(ctx, doc_of):
    output = run_intake(doc_of("contract_no_dates.md"), ctx)
    headings = [section.heading for section in output.sections]
    assert "MASTER SERVICES AGREEMENT" in headings
    assert any(h.startswith("6. GOVERNING LAW") or h == "6. GOVERNING LAW" for h in headings)
    assert all(s.start_line <= s.end_line for s in output.sections)


def test_replacement_characters_are_flagged(ctx):
    output = run_intake(text_doc("Rent of EUR 4.850,00 per m�nth for the equipment lease."), ctx)
    assert "replacement_characters_present" in codes(output)


def test_very_short_document_is_flagged(ctx):
    output = run_intake(text_doc("Hello."), ctx)
    assert "very_short_document" in codes(output)


def test_control_characters_are_stripped(ctx):
    output = run_intake(text_doc("Agreement\x00 between\x07 the parties for services rendered."), ctx)
    assert "\x00" not in output.normalized_text
    assert "control_characters_stripped" in codes(output)


def test_noise_ratio_flags_ocr_garbage(ctx):
    noisy = "".join("«»§¤¶‡" for _ in range(40)) + " agreement between parties"
    output = run_intake(text_doc(noisy), ctx)
    assert "high_noise_ratio" in codes(output)
    assert any(w.severity is Severity.WARN for w in output.warnings)


def test_scientific_notation_survives_normalization(ctx):
    """NFKC would rewrite x10^9 as x109 and SpO_2 as SpO2 - a change of meaning."""
    output = run_intake(
        text_doc(
            "Laboratory results for the patient visit on this date.\n"
            "WBC 3.8 ×10⁹/L, platelets 168 ×10⁹/L, SpO₂ 98% on room air.\n"
        ),
        ctx,
    )
    assert "×10⁹/L" in output.normalized_text
    assert "SpO₂" in output.normalized_text
    assert "×109/L" not in output.normalized_text


def test_ligatures_are_still_repaired_without_nfkc(ctx):
    output = run_intake(
        text_doc("The oﬃce shall conﬁrm the ﬂow of goods under this agreement."), ctx
    )
    assert "office" in output.normalized_text
    assert "confirm" in output.normalized_text


def test_a_pdf_fails_the_step_instead_of_being_decoded(ctx):
    """Three LLM calls against PDF object dictionaries is a bug, not a trace.

    The header is written inline rather than committing a binary fixture. PDF
    syntax is mostly ASCII, so a real file decodes to a printable-looking
    string and slips past both the replacement-character and noise-ratio
    checks - only the magic bytes catch it, and those are all this needs.
    """
    document = text_doc(
        "%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        "trailer\n<< /Root 1 0 R >>\n%%EOF\n",
        name="scan.pdf",
    )
    with pytest.raises(StepFailure) as caught:
        run_intake(document, ctx)
    assert caught.value.code == "binary_content"
    assert "PDF" in caught.value.message


@pytest.mark.parametrize(
    ("signature", "label"),
    [
        ("%PDF-1.7\nobj", "PDF"),
        ("PK\x03\x04\x14\x00", "ZIP"),
        ("\x89PNG\r\n\x1a\n", "PNG"),
        ("{\\rtf1\\ansi", "RTF"),
    ],
)
def test_known_container_formats_are_rejected_by_signature(ctx, signature, label):
    with pytest.raises(StepFailure) as caught:
        run_intake(text_doc(signature + "padding " * 20), ctx)
    assert caught.value.code == "binary_content"
    assert label in caught.value.message


def test_a_stray_nul_is_stripped_not_treated_as_binary(ctx):
    output = run_intake(text_doc("Agreement\x00 between the parties for services rendered."), ctx)
    assert "control_characters_stripped" in codes(output)


def test_pervasive_nuls_are_treated_as_binary(ctx):
    with pytest.raises(StepFailure) as caught:
        run_intake(text_doc("a\x00b\x00c\x00d\x00e\x00f\x00g\x00h\x00i\x00" * 4), ctx)
    assert caught.value.code == "binary_content"
