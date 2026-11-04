from __future__ import annotations

from decimal import Decimal

from obspipe.llm.backends import FakeBackend
from obspipe.models.extraction import AmountKind
from obspipe.steps.extraction import run_extraction
from obspipe.steps.intake import run_intake
from tests.helpers import text_doc


def extract(document, ctx, backend):
    return run_extraction(run_intake(document, ctx), ctx, backend)


def codes(output) -> set[str]:
    return set(output.warning_codes())


def test_amounts_keep_raw_text_and_parsed_value_separately(ctx, backend, doc_of):
    output = extract(doc_of("invoice_mixed_currency.txt"), ctx, backend)
    amounts = output.entities.amounts
    assert amounts, "the invoice plainly contains amounts"
    by_raw = {a.raw_text: a for a in amounts}
    assert by_raw["$18,000.00"].value == Decimal("18000.00")
    assert by_raw["$18,000.00"].currency == "USD"
    assert by_raw["EUR 4.250,00"].value == Decimal("4250.00")


def test_mixed_currencies_are_flagged(ctx, backend, doc_of):
    output = extract(doc_of("invoice_mixed_currency.txt"), ctx, backend)
    assert output.entities.currencies() >= {"USD", "EUR", "INR"}
    assert "mixed_currencies" in codes(output)


def test_missing_dates_are_reported_not_invented(ctx, backend, doc_of):
    output = extract(doc_of("contract_no_dates.md"), ctx, backend)
    assert output.entities.dates == []
    assert "no_dates_found" in codes(output)


def test_hallucinated_span_is_caught_as_an_error(ctx, doc_of):
    """The injected date is not in the document; step 2 must say so itself."""
    output = extract(doc_of("contract_no_dates.md"), ctx, FakeBackend(faults={"hallucinate_date"}))
    assert "hallucinated_date_span" in codes(output)
    assert output.has_errors


def test_empty_extraction_is_an_error_not_a_silent_pass(ctx, doc_of):
    output = extract(doc_of("contract_no_dates.md"), ctx, FakeBackend(faults={"empty_extraction"}))
    assert output.entities.is_empty()
    assert "no_entities_extracted" in codes(output)
    assert "low_extraction_confidence" in codes(output)
    assert output.has_errors


def test_recall_miss_is_detected_against_an_independent_scan(ctx, doc_of):
    """Dropping amounts the document visibly contains is caught without ground truth."""
    output = extract(doc_of("invoice_mixed_currency.txt"), ctx, FakeBackend(faults={"drop_amounts"}))
    assert output.entities.amounts == []
    assert "amount_recall_miss" in codes(output)


def test_total_reconciliation_is_skipped_when_roles_are_unknown(ctx, backend, doc_of):
    """Reconciling against unlabelled amounts would invent arithmetic; say so instead."""
    output = extract(doc_of("invoice_mixed_currency.txt"), ctx, backend)
    assert "total_reconciliation_skipped" in codes(output)
    assert "total_mismatch" not in codes(output)


def test_total_mismatch_is_reported_when_components_are_known(ctx, backend):
    document = text_doc(
        "INVOICE\n"
        "Consulting, 10 hours   $1,000.00\n"
        "Support, 5 hours       $500.00\n"
        "Total Due              $2,000.00\n",
        name="mismatch.txt",
    )
    output = extract(document, ctx, backend)
    kinds = {a.kind for a in output.entities.amounts}
    assert AmountKind.TOTAL in kinds and AmountKind.LINE_ITEM in kinds
    assert "total_mismatch" in codes(output)


def test_line_items_without_a_total_are_flagged(ctx, backend, doc_of):
    output = extract(doc_of("invoice_no_total.txt"), ctx, backend)
    assert output.entities.total_amount() is None
    assert "no_total_found" in codes(output)


def test_truncated_upstream_is_attributed_to_intake_not_the_model(ctx, backend, doc_of):
    output = extract(doc_of("contract_long_truncated.md"), ctx, backend)
    assert "extracted_from_truncated_text" in codes(output)
    assert "no_dates_found" in codes(output)


def test_invalid_iso_date_from_the_model_is_dropped(ctx, backend):
    """A model-supplied date that is not a real date must not survive into entities."""
    from obspipe.llm.schemas import ExtractionLLMOutput, RawDate
    from langchain_core.runnables import RunnableLambda

    class Bad(FakeBackend):
        def structured(self, schema, include_raw=False):
            if schema is ExtractionLLMOutput:
                return RunnableLambda(
                    lambda _: ExtractionLLMOutput(
                        dates=[RawDate(raw_text="30 February 2024", iso_date="2024-02-30")],
                        confidence=0.9,
                    )
                )
            return super().structured(schema, include_raw)

    document = text_doc("Signed on 30 February 2024 by both parties to this agreement.")
    output = extract(document, ctx, Bad())
    assert output.entities.dates[0].iso_date is None
    assert "invalid_iso_date" in codes(output)


def test_null_lists_from_the_model_become_empty_lists():
    """A provider that emits null for an absent section must not break the schema."""
    from obspipe.llm.schemas import ExtractionLLMOutput, ReportSummaryLLMOutput

    extraction = ExtractionLLMOutput.model_validate(
        {"parties": None, "dates": None, "amounts": None, "key_terms": None,
         "identifiers": None, "confidence": 0.5}
    )
    assert extraction.parties == []
    assert extraction.dates == []
    assert extraction.identifiers == []

    summary = ReportSummaryLLMOutput.model_validate(
        {"headline": "A study of thermal tolerance", "bullets": None,
         "unresolved_questions": None, "recommendations": None, "key_findings": None}
    )
    assert summary.recommendations == []
    assert summary.bullets == []


def test_nullable_lists_are_advertised_in_the_json_schema():
    """The provider validates against this schema, so null must be legal in it."""
    from obspipe.llm.schemas import ReportSummaryLLMOutput

    field = ReportSummaryLLMOutput.model_json_schema()["properties"]["recommendations"]
    variants = {entry.get("type") for entry in field.get("anyOf", [field])}
    assert {"array", "null"} <= variants


def test_measurements_are_not_recorded_as_monetary_amounts(ctx, backend):
    """A clinical document has no money; vitals must not become amounts."""
    from langchain_core.runnables import RunnableLambda

    from obspipe.llm.schemas import ExtractionLLMOutput, RawAmount

    class Clinical(FakeBackend):
        def structured(self, schema, include_raw=False):
            if schema is ExtractionLLMOutput:
                return RunnableLambda(
                    lambda _: ExtractionLLMOutput(
                        amounts=[
                            RawAmount(raw_text="39.1 °C", label="Temperature"),
                            RawAmount(raw_text="112 / 72 mmHg", label="Blood pressure"),
                            RawAmount(raw_text="94 beats/min", label="Pulse"),
                            RawAmount(raw_text="13.6 g/dL", label="Hemoglobin"),
                        ],
                        confidence=0.8,
                    )
                )
            return super().structured(schema, include_raw)

    document = text_doc(
        "CLINICAL REPORT\nTemperature 39.1 °C\nBlood pressure 112 / 72 mmHg\n"
        "Pulse 94 beats/min\nHemoglobin 13.6 g/dL\n",
        name="clinical.txt",
    )
    output = extract(document, ctx, Clinical())
    assert output.entities.amounts == []
    assert "currency_unresolved" not in codes(output)
    assert "no_total_found" not in codes(output)
    assert "non_monetary_amounts_discarded" in codes(output)


def test_real_money_is_never_discarded_as_a_measurement(ctx, backend, doc_of):
    output = extract(doc_of("invoice_mixed_currency.txt"), ctx, backend)
    assert len(output.entities.amounts) >= 5
    assert "non_monetary_amounts_discarded" not in codes(output)


def test_model_supplied_currency_from_a_table_header_is_not_a_hallucination(ctx):
    """Bare cells under a '(₹)' header come back denominated; the figure is real."""
    from langchain_core.runnables import RunnableLambda

    from obspipe.llm.schemas import ExtractionLLMOutput, RawAmount

    class Annotator(FakeBackend):
        def structured(self, schema, include_raw=False):
            if schema is ExtractionLLMOutput:
                return RunnableLambda(
                    lambda _: ExtractionLLMOutput(
                        amounts=[
                            RawAmount(raw_text="₹48,750.00", label="Buffet", kind="line_item"),
                            RawAmount(raw_text="₹99,999.00", label="Invented", kind="line_item"),
                        ],
                        confidence=0.9,
                    )
                )
            return super().structured(schema, include_raw)

    document = text_doc(
        "INVOICE\n| Description | Quantity | Unit Price (₹) | Amount (₹) |\n"
        "| Buffet lunch | 75 | 650.00 | 48,750.00 |\n",
        name="header_currency.txt",
    )
    output = extract(document, ctx, Annotator())
    observed = codes(output)
    assert "currency_annotated_by_model" in observed
    # The figure that is genuinely absent is still an error.
    assert "hallucinated_amount_span" in observed
    messages = [w.message for w in output.warnings if w.code == "hallucinated_amount_span"]
    assert any("99,999.00" in m for m in messages)


def test_unrecognised_enum_labels_are_mapped_not_rejected(ctx):
    """A date role the enum never anticipated must not fail the whole call."""
    from langchain_core.runnables import RunnableLambda

    from obspipe.llm.schemas import ExtractionLLMOutput, RawDate
    from obspipe.models.extraction import DateRole

    class Inventive(FakeBackend):
        def structured(self, schema, include_raw=False):
            if schema is ExtractionLLMOutput:
                return RunnableLambda(
                    lambda _: ExtractionLLMOutput(
                        dates=[
                            RawDate(raw_text="25 September 2026", role="event"),
                            RawDate(raw_text="10 Oct 2026", role="deadline"),
                            RawDate(raw_text="14 Sep 2026", role="issued"),
                        ],
                        confidence=0.9,
                    )
                )
            return super().structured(schema, include_raw)

    document = text_doc(
        "INVOICE dated 14 Sep 2026 for an event on 25 September 2026, payable by 10 Oct 2026.",
        name="roles.txt",
    )
    output = extract(document, ctx, Inventive())
    roles = [d.role for d in output.entities.dates]
    assert roles == [DateRole.MENTIONED, DateRole.DUE, DateRole.ISSUED]
    assert "unrecognised_date_role" in codes(output)


def test_advance_and_balance_do_not_break_total_reconciliation(ctx, backend):
    """An invoice quoting an advance and a balance still reconciles against its total."""
    document = text_doc(
        "INVOICE\nBuffet, 75 units      ₹78,500.00\n"
        "CGST @ 2.5%            ₹1,962.50\n"
        "SGST @ 2.5%            ₹1,962.50\n"
        "Grand Total            ₹82,425.00\n"
        "Advance Paid           ₹30,000.00\n"
        "Balance Due            ₹52,425.00\n",
        name="advance.txt",
    )
    output = extract(document, ctx, backend)
    assert "total_mismatch" not in codes(output)
