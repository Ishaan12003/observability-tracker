from __future__ import annotations

from obspipe.llm.backends import FakeBackend
from obspipe.models.document import DocType
from obspipe.steps.classification import run_classification
from obspipe.steps.extraction import run_extraction
from obspipe.steps.intake import run_intake
from obspipe.steps.summarization import run_summarization


def summarize(document, ctx, backend):
    intake = run_intake(document, ctx)
    extraction = run_extraction(intake, ctx, backend)
    classification = run_classification(intake, extraction, ctx, backend)
    return run_summarization(intake, extraction, classification, ctx, backend)


def codes(output) -> set[str]:
    return set(output.warning_codes())


def test_summary_shape_follows_the_classified_type(ctx, backend, doc_of):
    contract = summarize(doc_of("contract_no_dates.md"), ctx, backend)
    invoice = summarize(doc_of("invoice_mixed_currency.txt"), ctx, backend)
    assert contract.doc_type is DocType.CONTRACT
    assert {"effective_date", "governing_law", "termination"} <= set(contract.type_specific)
    assert invoice.doc_type is DocType.INVOICE
    assert {"total_stated", "currencies_seen", "due_date"} <= set(invoice.type_specific)
    assert "effective_date" not in invoice.type_specific


def test_missing_date_is_disclosed_instead_of_filled(ctx, backend, doc_of):
    output = summarize(doc_of("contract_no_dates.md"), ctx, backend)
    assert output.type_specific["effective_date"] is None
    assert output.unresolved_questions
    assert "missing_gap_disclosure" not in codes(output)


def test_invented_effective_date_is_caught_when_no_dates_exist(ctx, doc_of):
    """The summarizer states a date the extractor never found - an error, not a nit."""
    from langchain_core.runnables import RunnableLambda

    from obspipe.llm.schemas import ContractSummaryLLMOutput

    class Inventive(FakeBackend):
        def structured(self, schema, include_raw=False):
            if schema is ContractSummaryLLMOutput:
                return RunnableLambda(
                    lambda _: ContractSummaryLLMOutput(
                        headline="Master services agreement between two parties",
                        bullets=["Provider furnishes services", "Client pays fees"],
                        effective_date="1 January 2024",
                    )
                )
            return super().structured(schema, include_raw)

    output = summarize(doc_of("contract_no_dates.md"), ctx, Inventive())
    assert "invented_effective_date" in codes(output)
    assert output.has_errors


def test_unsupported_amount_in_summary_is_an_error(ctx, doc_of):
    output = summarize(doc_of("invoice_no_total.txt"), ctx, FakeBackend(faults={"invent_total"}))
    assert "unsupported_amount_in_summary" in codes(output)
    assert output.has_errors


def test_mixed_currency_invoice_must_not_present_one_total(ctx, backend, doc_of):
    output = summarize(doc_of("invoice_mixed_currency.txt"), ctx, backend)
    assert "single_total_over_mixed_currencies" in codes(output)
    assert set(output.type_specific["currencies_seen"]) >= {"USD", "EUR", "INR"}


def test_ambiguity_must_be_disclosed_by_the_summary(ctx, backend, doc_of):
    """Step 3 was torn; a summary that hides that is itself a defect."""
    output = summarize(doc_of("memo_ambiguous.md"), ctx, backend)
    assert "ambiguity_not_disclosed" in codes(output)


def test_unknown_type_uses_the_generic_template(ctx, backend):
    from obspipe.models.document import RawDocument

    document = RawDocument.from_text("The quick brown fox jumped over the lazy dog. " * 6)
    output = summarize(document, ctx, backend)
    assert output.doc_type is DocType.UNKNOWN
    assert "summarized_without_type" in codes(output)


def test_reformatted_amount_is_not_reported_as_a_hallucination(ctx, backend):
    """A German-notation figure rewritten in US notation is the same money."""
    from langchain_core.runnables import RunnableLambda

    from obspipe.llm.schemas import ContractSummaryLLMOutput
    from obspipe.models.document import RawDocument

    class Reformatter(FakeBackend):
        def structured(self, schema, include_raw=False):
            if schema is ContractSummaryLLMOutput:
                return RunnableLambda(
                    lambda _: ContractSummaryLLMOutput(
                        headline="Lease grants a purchase option at the stated residual value",
                        bullets=["Residual value is EUR 96,000.00", "Option needs written notice"],
                    )
                )
            return super().structured(schema, include_raw)

    document = RawDocument.from_text(
        "LEASE AGREEMENT between Alpha GmbH and Beta AG.\n"
        "The parties agree that the residual value is EUR 96.000,00 payable on exercise.\n"
        "This agreement is governed by the laws of Germany and may be terminated on notice.\n",
        source_name="reformat.txt",
    )
    output = summarize(document, ctx, Reformatter())
    assert "unsupported_amount_in_summary" not in codes(output)
    assert "amount_reformatted_in_summary" in codes(output)
    assert not output.has_errors


def test_currency_swap_in_summary_is_flagged_separately(ctx, backend):
    """Right number, wrong currency is a distinct defect from an invented one."""
    from langchain_core.runnables import RunnableLambda

    from obspipe.llm.schemas import ContractSummaryLLMOutput
    from obspipe.models.document import RawDocument

    class Swapper(FakeBackend):
        def structured(self, schema, include_raw=False):
            if schema is ContractSummaryLLMOutput:
                return RunnableLambda(
                    lambda _: ContractSummaryLLMOutput(
                        headline="Lease sets the residual value payable on exercise",
                        bullets=["Residual value is $96,000.00", "Written notice required"],
                    )
                )
            return super().structured(schema, include_raw)

    document = RawDocument.from_text(
        "LEASE AGREEMENT between Alpha GmbH and Beta AG.\n"
        "The parties agree that the residual value is EUR 96.000,00 payable on exercise.\n"
        "This agreement is governed by the laws of Germany and may be terminated on notice.\n",
        source_name="swap.txt",
    )
    output = summarize(document, ctx, Swapper())
    assert "amount_currency_changed_in_summary" in codes(output)
    assert "unsupported_amount_in_summary" not in codes(output)


def test_thin_summary_counts_type_specific_items_too(ctx, backend):
    """A report that fills key_findings and skips bullets is not a thin summary."""
    from langchain_core.runnables import RunnableLambda

    from obspipe.llm.schemas import ReportSummaryLLMOutput
    from obspipe.models.document import RawDocument

    class FindingsOnly(FakeBackend):
        def structured(self, schema, include_raw=False):
            if schema is ReportSummaryLLMOutput:
                return RunnableLambda(
                    lambda _: ReportSummaryLLMOutput(
                        headline="Quarterly fulfilment performance held steady",
                        bullets=[],
                        key_findings=["Accuracy flat at 97.2%", "Dispatch time rose to 26 hours"],
                        recommendations=["Submit a peak staffing plan"],
                    )
                )
            return super().structured(schema, include_raw)

    document = RawDocument.from_text(
        "QUARTERLY REPORT\nExecutive summary of fulfilment analysis and findings.\n"
        "Methodology draws on warehouse data. We recommend a staffing plan.\n",
        source_name="report.txt",
    )
    output = summarize(document, ctx, FindingsOnly())
    assert "thin_summary" not in codes(output)
    assert "bullets_omitted" in codes(output)


def test_summary_currency_annotation_is_not_an_invented_figure(ctx):
    """'650.00' under a '(₹)' header, quoted back as '₹650.00', is grounded."""
    from langchain_core.runnables import RunnableLambda

    from obspipe.llm.schemas import InvoiceSummaryLLMOutput
    from obspipe.models.document import RawDocument

    class Annotator(FakeBackend):
        def structured(self, schema, include_raw=False):
            if schema is InvoiceSummaryLLMOutput:
                return RunnableLambda(
                    lambda _: InvoiceSummaryLLMOutput(
                        headline="Catering invoice for an office lunch",
                        bullets=["Unit price is ₹650.00 per head", "Grand total ₹82,425.00"],
                        total_stated="₹82,425.00",
                        currencies_seen=["INR"],
                    )
                )
            return super().structured(schema, include_raw)

    document = RawDocument.from_text(
        "INVOICE\n| Description | Quantity | Unit Price (₹) | Amount (₹) |\n"
        "| Buffet lunch | 75 | 650.00 | 48,750.00 |\n"
        "Grand Total  ₹82,425.00\nPayment terms: Net 30\n",
        source_name="catering.txt",
    )
    output = summarize(document, ctx, Annotator())
    assert "unsupported_amount_in_summary" not in codes(output)
    assert "currency_annotated_by_model" in codes(output)
    assert not output.has_errors
