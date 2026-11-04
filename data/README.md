# data/

A drop zone for your own documents. **Nothing in here is tracked by git** apart
from this file.

## Run your own documents through the pipeline

Put any text, Markdown, or PDF-extracted text file in `data/documents/` and
point the pipeline at it:

```bash
obspipe run data/documents/my-contract.md --blame
```

Supported inputs are plain text, Markdown, and text already extracted from a
PDF. Binary files are rejected at intake rather than being decoded into
gibberish, so extract the text first:

```bash
pdftotext -layout invoice.pdf data/documents/invoice.pdf.txt
```

## Optional: declare what a document should produce

Add a `<filename>.meta.json` sidecar and the test suite and evaluation loop
will treat the document as labelled ground truth:

```json
{
  "doc_id": "my-contract",
  "expected_type": "contract",
  "expected_failure": "no_dates",
  "expected_warning_codes": ["no_dates_found"],
  "notes": "Why this document is interesting and how it is meant to break."
}
```

A document without a sidecar still runs; it is simply unlabelled, so nothing is
asserted about what it *should* produce.

## The demo corpus

`data/demo/` holds 50 generated documents used by the walkthrough in
`docs/DEMO.md`. It is generated, not committed:

```bash
obspipe demo generate
obspipe demo seed
```

## Where the committed fixtures live

The eight designed-failure documents the test suite asserts against are part of
the codebase, not user data, and live in `tests/fixtures/documents/`. Each ships
a sidecar naming the failure it provokes.
