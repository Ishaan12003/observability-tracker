# obspipe

An observability layer for multi-step AI pipelines. When the final output is
bad, the question is *which step broke*. obspipe traces every step, walks the
trace backwards to the root cause, explains how the error propagated, and turns
every confirmed diagnosis into a regression test.

```bash
.venv/bin/pip install -e ".[dev]"
obspipe demo generate && obspipe demo seed --reset && obspipe serve   # http://127.0.0.1:8765
```

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: diagrams of the pipeline, the
  tracing layer, the backward analysis and the feedback loop
- **[docs/DEMO.md](docs/DEMO.md)**: the 50-document demo scenario and the Loom script

| Phase | What it adds |
|---|---|
| 1 | A four-step pipeline whose steps are isolated, typed and honest about failure |
| 2 | Tracing: full-fidelity spans, LLM call capture, suspicion scoring, storage |
| 3 | Backward analysis: root cause, failure category, evidence chain |
| 4 | Visual trace explorer: pipeline graph, span details, diff view, flagging |
| 5 | Feedback loop: flags become eval cases, regression tracking, failure analytics |
| 6 | 50-document demo corpus with independent ground truth, plus the architecture doc |

---

# Phase 1: the traceable pipeline

Phase 1 builds the thing to be observed: a four-step document-understanding
pipeline whose steps are isolated, typed, and honest about their own failures.

## The pipeline

| # | Step | LLM? | Input | Output |
|---|------|------|-------|--------|
| 1 | Intake | no | `RawDocument` | `IntakeOutput` |
| 2 | Extraction | yes | `IntakeOutput` | `ExtractionOutput` |
| 3 | Classification | yes | `IntakeOutput` + `ExtractionOutput` | `ClassificationOutput` |
| 4 | Summarization | yes | all three above | `SummaryOutput` |

Every step is a plain function `(typed input, RunContext, backend) -> typed
output`. No step reads globals, calls another step, or mutates its input. Step
boundaries are therefore exactly where spans will go in Phase 2.

## Three design decisions that make failures attributable

**1. The model finds spans; Python decides what they mean.**
The extraction prompt asks only for text the model can see — `raw_text` copied
verbatim. Currency parsing, date validation, total reconciliation and margin
arithmetic all happen afterwards in `parsing.py` and the step bodies. When a
number is wrong you can tell whether the model quoted the wrong span or the
parser misread it, because both survive side by side.

**2. Every output carries `warnings[]`.**
A step that does something questionable says so, with a stable machine code, a
severity, and the field it applies to. A step that cannot produce output at all
raises `StepFailure`. Everything short of that is degraded-but-traced, because
a partial artifact is still evidence.

**3. Classification returns scores, not a label.**
All four candidates are scored; the winner, the `margin` and the `ambiguous`
verdict are computed from them. "The model was torn between report and
correspondence" survives as data instead of collapsing into one
confident-looking answer — which is what later explains a structurally wrong
summary, since step 4's prompt *and* output schema are selected by step 3's
verdict.

### Checks that catch failures without ground truth

| Check | Catches |
|---|---|
| `appears_in(span, source)` on every party, date and amount | hallucinated spans |
| Independent `MONEY_SCAN_RE` sweep vs. what the model returned | recall misses |
| Model `iso_date` vs. locally parsed date | bad normalization |
| Stated total vs. sum of its same-currency components | arithmetic drift |
| Distinct currencies in one document | invalid combined figures |
| Summary money figures vs. extracted amounts, by parsed value | invented numbers in output |
| Classified type vs. entities that type requires | step 2 and step 3 disagreeing |

## Backends

Both backends are LCEL Runnables, so step code is identical either way:

```python
chain = EXTRACTION_PROMPT | backend.structured(ExtractionLLMOutput)
raw = chain.invoke({...})
```

- **`fake`** (default) — rule-based, deterministic, offline, free. Same document
  in, same object out. Supports injected faults, so a test can manufacture the
  exact bad output the observability layer is supposed to catch:
  `hallucinate_date`, `drop_amounts`, `flat_scores`, `overconfident`,
  `invent_total`, `empty_extraction`.
- **`groq`** — `ChatGroq` with `with_structured_output`, so the model is
  constrained by the schema through tool calling rather than asked politely for
  JSON. Needs `GROQ_API_KEY` in `.env`. Default model `openai/gpt-oss-120b`.

### Running against a real model

```bash
cp .env.example .env      # then put the real key in .env, never in .env.example
obspipe run-all --backend groq --out-dir runs/groq
pytest -m live            # live suite, opt-in, needs the key
```

`.env` is gitignored; `.env.example` is the committed template and must only
ever hold a placeholder.

The default `pytest` run is offline, free and deterministic — live tests are
deselected by `addopts = "-m 'not live'"`. The live suite asserts only the
contract a step must honour whatever the model says: the run is fully traced,
spans are grounded, the summary schema matches the classified type, and gaps
are disclosed rather than filled.

### What the real model changed

Three things the fake backend could not have shown:

- **Model availability is a step failure, not a crash.** A wrong model id
  surfaced as `extraction: FAILED / extraction_call_failed` with the 404 body
  preserved on the record, downstream steps `SKIPPED`, exit code 1.
- **Grounding had to become value-aware.** The model rewrote the document's
  German `EUR 96.000,00` as `EUR 96,000.00` and the span-literal check called
  it a hallucination. It now matches on parsed value and currency first, and
  distinguishes three cases: `amount_reformatted_in_summary` (info, same
  money), `amount_currency_changed_in_summary` (warn, right number wrong
  currency) and `unsupported_amount_in_summary` (error, matches nothing).
- **Ambiguity is real, not synthetic.** `letter_multilingual.txt` classified as
  `contract` (0.85 vs 0.75) on one run and `correspondence` (0.92 vs 0.88) on
  the next. Both runs raised `ambiguous_classification`. A pipeline that
  returned only the winning label would have looked confident and stable while
  silently flipping the summary schema between runs.

## The corpus

Eight documents in `tests/fixtures/documents/`, each with a `<filename>.meta.json`
sidecar naming the failure it is designed to cause. These are the Phase 4
evaluation dataset in embryo.

| document | designed failure |
|---|---|
| `contract_no_dates.md` | a complete-looking contract with zero calendar dates |
| `invoice_mixed_currency.txt` | USD + EUR (European decimals) + INR (lakh grouping), total covers only USD |
| `invoice_no_total.txt` | line items, no stated total — tempting the summarizer to compute one |
| `memo_ambiguous.md` | genuinely between report and correspondence |
| `contract_ocr_noise.pdf.txt` | OCR damage: `1`→`l`, ligatures, hyphenated line-breaks |
| `contract_long_truncated.md` | 20k chars; every date lives past the 12k intake budget |
| `letter_multilingual.txt` | English letter whose controlling clause is German |
| `empty.txt` | whitespace only — must fail before three LLM calls are paid for |

`contract_long_truncated.md` is the sharpest case: the pipeline reports a
contract with no effective date and no value, and the root cause is **step 1**,
not the extractor. Proving that distinction is the whole point.

### Running your own documents

The corpus above is committed because the test suite asserts against it. Your
own documents go in `data/`, which is untracked:

```bash
cp my-contract.md data/documents/
obspipe run data/documents/my-contract.md --blame
```

Binary files are rejected at intake rather than decoded into gibberish, so
extract PDF text first (`pdftotext -layout in.pdf data/documents/in.pdf.txt`).
Add a `.meta.json` sidecar to declare what a document should produce and it
becomes labelled ground truth for the evaluation loop. See
[data/README.md](data/README.md).

## Usage

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"

obspipe docs                                        # the corpus and its failure modes
obspipe faults                                      # injectable faults
obspipe run-all                                     # every document, comparison table
obspipe run tests/fixtures/documents/invoice_mixed_currency.txt
obspipe run tests/fixtures/documents/contract_no_dates.md --fault hallucinate_date
obspipe run <doc> --json --out run.json             # full result as JSON
obspipe run <doc> --backend groq                    # needs GROQ_API_KEY

pytest -q                                           # full suite, offline
pytest -m live                                      # 7 live tests, needs GROQ_API_KEY
```

## Layout

```
src/obspipe/
  models/       Pydantic contracts for every intermediate artifact
  llm/          prompts, LLM-facing schemas, backends, the fake engine
  steps/        the four steps, one file each
  tracing/      Span, Trace, the @traced decorator, suspicion scoring
  storage/      JSON trace files + the SQLite index
  analysis/     backward root-cause analysis, evidence chains, taxonomy
  web/          the stdlib trace explorer served by `obspipe serve`
  feedback/     flags, eval cases, regression tracking, analytics
  demo/         the 50-document generator and seeding scenario
  parsing.py    deterministic money/date parsing and span grounding
  digest.py     how one step's output is rendered into the next one's prompt
  corpus.py     corpus loading; owns both corpus locations
  pipeline.py   the orchestrator
  workspace.py  ties traces, goldens and the eval dataset together

tests/fixtures/documents/   the committed failure-mode corpus (asserted against)
tests/fixtures/traces/      normalized golden traces (regression guard)
data/                       your own documents + generated demo corpus (untracked)
```

## What Phase 1 left to Phase 2

`RunContext` was threaded through every step already, and `StepRecord` held
step, status, timing and warnings — the shape of a span, kept by hand. Phase 2
replaced that bookkeeping with real spans **without changing a single step
signature**, exactly as planned.

---

# Phase 2: the tracing layer

Phase 1 answered "did this go wrong?". Phase 2 answers **"where?"**.

No OpenTelemetry, no LangSmith, no Langfuse. An OTel attribute must be a
primitive, which makes the one thing this project exists to store - the full
input and output of every step - the one thing it cannot hold. And the span
needs fields no vendor has: `warning_codes`, `computed_confidence`,
`suspicion`. So the trace model is our own.

## What a trace holds

```
Trace
├── trace_id, doc_id, backend, model, status, duration_ms
├── spans[]
│   ├── pipeline span
│   └── one step span each, carrying:
│       ├── inputs / output          (payload refs)
│       ├── warnings[]               (deterministic, from Phase 1)
│       ├── self_confidence 1-5      (what the model claims)
│       ├── computed_confidence 0-1  (what the pipeline measured)
│       ├── suspicion 0-1            (derived from both)
│       └── llm_calls[]
│           └── prompt, raw_response, parsed, tokens, finish_reason, latency
├── payloads{}   sha256 -> content, deduplicated
└── result       the final PipelineResult
```

## Instrumenting a step is one line

```python
@traced(StepName.EXTRACTION)
def run_extraction(intake: IntakeOutput, ctx: RunContext, backend: LLMBackend) -> ExtractionOutput:
    ...   # body unchanged from Phase 1
```

The active span lives in a `ContextVar`, which is what lets the decorator take
no arguments beyond the step name and lets the LLM backend attach a child call
to whichever step is running without the step passing anything down.

**Prompts, raw responses and token counts are captured somewhere else**, because
a step cannot see them: `with_structured_output` returns a validated object and
discards the `AIMessage` that carried them. `ObservedBackend` wraps any backend,
calls it with `include_raw=True`, records the call, and hands the step back
exactly what it expected. No step knows it is being observed.

## Two confidences, deliberately kept apart

Self-reported LLM confidence is poorly calibrated. Across a live corpus run,
extraction averaged **4.8/5 self-confidence on spans that were degraded five
times out of six**. Taken alone it would rank suspects by the model's mood.

So every step also reports a `computed_confidence` the pipeline measures itself:

| step | measures |
|---|---|
| intake | share of characters that are not noise |
| extraction | share of extracted spans verified present in the source |
| classification | margin between the top two labels |
| summarization | share of summary figures that trace back to the document |

`suspicion` weights the deterministic evidence above the self-report, and gives
the earliest failing step a bonus - cause beats symptom. Warnings that exist to
say "the problem came from upstream" (`extracted_from_truncated_text`) are
excluded, so a victim is not incriminated.

Live example - `letter_multilingual.txt`:

```
model rated itself 4/5   "clearly serves both as contractual notice and business letter"
pipeline measured  0.03   actual margin between the top two labels
suspicion          0.66
```

## Storage

```
traces/YYYY-MM-DD/<trace_id>.json    full fidelity, indented, git-ignored
traces/index.db                      SQLite: traces + spans tables
tests/fixtures/traces/*.json         normalized golden traces, committed
```

The JSON files are the source of truth; the index is derived and rebuildable
with `obspipe reindex`. The `spans` table is what turns the project's central
question into one query:

```sql
SELECT name, COUNT(*) FROM spans WHERE kind='step' AND status!='ok' GROUP BY name;
```

Payloads above 2KB are content-addressed by sha256 and stored once per trace.
The document body inside each rendered prompt is referenced rather than
inlined - otherwise a 20k-char contract is written four times over. Traces run
~25-40KB typical, 78KB for a 20k-char document.

Golden traces are normalized (ids, timestamps, durations and payload hashes
stripped) so they diff cleanly in review and only move when the trace format
genuinely changes. Regenerate deliberately:

```bash
OBSPIPE_UPDATE_GOLDEN=1 pytest tests/test_storage.py
```

## Commands

```bash
obspipe run <doc> --blame          # run, then rank the steps by responsibility
obspipe run <doc> --json --trace   # the whole trace as JSON
obspipe traces                     # stored runs, newest first
obspipe trace <id> --prompts       # one trace, span by span, prompts rehydrated
obspipe blame <id>                 # where did this go wrong?
obspipe stats                      # which step breaks most, across every run
obspipe reindex                    # rebuild SQLite from the JSON files
```

Trace ids accept an unambiguous prefix, the way git accepts short SHAs.


---

# Phase 3: backward analysis

Phase 2 ranked suspects. Phase 3 **explains**: for a bad output it names the
root-cause step and a failure category, and builds an evidence chain with one
link per affected step, each carrying the specific input/output pair behind it.

```
$ obspipe diagnose 3c4527e0
Hallucinated entity at Step 2 (Extraction)   confidence 95%  (ground truth, 0.8 ms)
intake  ->  extraction  ->  classification  ->  summarization

1. root cause  extraction
   Step 2 (Extraction) hallucinated the party 'John Smith', which does not appear
   in the source document.
2. propagation  summarization
   This propagated to Step 4 (Summarization), which included 'John Smith' in
   bullets[5] - "Parties involved: John Smith, Relecloud Inc, Adatum Health LLC."
```

- **Root cause = the earliest step that deviates.** Data only flows forward, so
  the first deviation is the cause and everything after it is propagation.
  Warnings that describe the document (`no_dates_found` on a contract with no
  dates), warnings that point upstream (`extracted_from_truncated_text`) and the
  model's self-doubt are excluded, so a victim is never convicted.
- **Ground truth, when it exists, decides.** `*.golden.json` records and human
  corrections are compared field by field against a normalized *projection* of
  each step's output. This is the only way to catch the `misclassify` fault: a
  confidently wrong label that trips no check at all.
- **16 failure categories** (`analysis/taxonomy.py`), mapped from warning codes:
  hallucination, recall miss, truncation, misclassification, unsupported claim,
  invalid aggregation, and so on.

# Phase 4: the visual trace explorer

`obspipe serve` starts a local explorer. It uses the standard library HTTP
server plus one vanilla-JS file, with no new dependencies.

- **Trace view.** Each step is a node, coloured green (healthy), yellow (low
  confidence or affected) or red (identified root cause). Clicking a node shows
  the full span: input, output, prompt, raw response, self-reported vs measured
  confidence, and warnings.
- **Diff view.** Received vs produced vs expected (from ground truth or a human
  correction), with the divergent fields and list items highlighted.
- **Flagging.** *⚑ Flag bad output* runs the backward analysis and shows the
  diagnosis. The reviewer confirms it, or overrides the step and category and
  edits the corrected output.

# Phase 5: the feedback-to-eval loop

- **Flags become eval cases.** Confirming a flag appends a self-contained case
  to `evals/dataset.jsonl`: the original input, the failing step, the failure
  codes, the bad output, the corrected output and the category.
- **Regression tracking.** `obspipe eval run --pipeline-version v3` replays
  every case. A case is fixed when its failure codes no longer recur *and* the
  step's output matches the correction. Runs are stored, and the explorer plots
  *known issues resolved* over time.
- **Pipeline versions** are concrete fixes: v2 raises the intake budget
  (truncation), and v3 adds grounding guards (hallucinated spans are dropped at
  extraction; an invented total is replaced with the grounded one).
- **Failure analytics.** The most common failure types, which step fails most,
  a step × category matrix, failure rate per batch, and time to root cause
  (automated vs human flag-to-confirmation).

# Phase 6: demo and docs

`obspipe demo generate` writes 50 documents. 37 are clean, and 13 are designed
to fail in 11 different ways. The golden records are written from what the
generator *put into* each document, never from pipeline output. The test suite
asserts that every clean document matches ground truth and that every designed
failure is diagnosed at its intended step and category.

## Commands (Phases 3–6)

```bash
obspipe diagnose <id>                  # root cause, category, evidence chain
obspipe diagnose-all                   # (re)cache diagnoses for every stored trace
obspipe flag <id> --reason "..."       # mark bad output; prints the diagnosis
obspipe confirm <flag-id> [--step S --category C --corrected '{...}']
obspipe flags                          # every flag and its outcome
obspipe eval list | run | trend        # the dataset, replay it, the trend line
obspipe analytics [--json]             # failure analytics
obspipe versions                       # pipeline versions v1-v3
obspipe serve [--port 8765]            # the visual trace explorer
obspipe demo generate | seed [--reset] # the 50-document demo scenario
obspipe run-all --dir data/demo --pipeline-version v1 --batch my-batch
```

```bash
pytest -q        # 189 tests, offline, about 6 s
```
