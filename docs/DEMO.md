# Demo scenario and Loom script

A 4–5 minute walkthrough: a bad output, the trace explorer, the root cause, the
flag, and the eval dataset growing. Everything runs offline on the fake backend,
deterministically, so every number below reproduces exactly.

## Setup (once, about 5 seconds)

```bash
.venv/bin/pip install -e ".[dev]"
obspipe demo generate                 # 50 documents -> data/demo (13 designed to fail)
obspipe demo seed --reset             # 3 batches (v1, v2, v3), 9 flags, 3 eval runs
obspipe serve                         # http://127.0.0.1:8765
```

> `--reset` deletes the stored trace files, `traces/index.db`,
> `traces/feedback.db` and `evals/dataset.jsonl` before seeding. To keep an
> existing workspace, point the demo somewhere else with
> `--dir demo_traces --dataset demo_evals/dataset.jsonl`, and pass the same
> `--dir`/`--dataset` to `obspipe serve`.

What the seed leaves behind:

| | batch-01 (v1) | batch-02 (v2) | batch-03 (v3) |
|---|---|---|---|
| failures | 13 / 50 | 11 / 50 | 8 / 50 |
| eval: known issues resolved | 0 / 9 | 2 / 9 | 4 / 9 |

Three batch-01 failures are left **unflagged** for the recording:
`009_contract_lease` (hallucinated "John Smith"), `017_invoice_wingtip`
(silent misclassification) and `021_memo_review` (ambiguous type).

## The 13 designed failures

| doc | root cause | category | how it is caused |
|---|---|---|---|
| 004_contract_long_lease | intake | input truncation | dates, fee and signature sit past the 12k budget |
| 009_contract_lease | extraction | hallucination | model invents signatory "John Smith" |
| 013_invoice_contoso | summarization | unsupported claim | summary states a total the invoice never gives |
| 017_invoice_wingtip | classification | misclassification | invoice confidently labelled a report; **no check fires** |
| 021_memo_review | classification | ambiguous classification | memo that is honestly report *and* correspondence |
| 024_invoice_bellows | extraction | recall miss | extractor returns no amounts from an itemised invoice |
| 028_scan_brandt | intake | input corruption | fax-quality scan, >25% noise characters |
| 031_contract_lease | extraction | hallucination | model invents an effective date of January 1, 2024 |
| 035_letter_brandt | classification | LLM call failure | classification call times out |
| 038_invoice_coho | summarization | invalid aggregation | USD + EUR lines, one USD total presented as the whole |
| 042_invoice_datum | extraction | normalization error | dd/mm dates read as mm/dd |
| 045_upload_blank | intake | invalid input | whitespace-only upload |
| 048_contract_long_license | intake | input truncation | second long contract: truncation is a pattern |

## Recording script

**0:00 – The problem (Traces tab).** Filter the batch to `batch-01`, then search
`009`. Open `009_contract_lease`.
> "This is a contract summary our pipeline produced. Look at the last bullet:
> *Parties involved: John Smith, …*. There is no John Smith in this contract.
> The question every AI team asks next is: which step did that?"

**0:30 – The trace view.** Point at the four nodes.
> "Every run is traced step by step. Green means healthy, yellow means low
> confidence or affected, red means identified root cause. Extraction is red,
> and Summarization is yellow because it was affected."

Click **Extraction**, then the **Prompt** and **Raw response** sub-tabs.
> "Every span keeps the full input, the output, the exact prompt, the raw model
> reply and two confidences: what the model claimed (5/5) and what the pipeline
> measured (0.83). They disagree, and the measured one is the honest one."

**1:15 – The diff view.** Open the **Diff** sub-tab.
> "What the step received, what it produced, and what it should have produced
> according to the golden dataset. The divergence is highlighted: 'John Smith'
> should not be there."

**1:40 – Flag it.** Click **⚑ Flag bad output**, type "who is John Smith?", and
click **Flag and diagnose**.
> "Flagging runs the backward analysis. In under a millisecond it walks the
> trace back from the bad output to the first step that deviated."

Read the narrative aloud:
> *Step 2 (Extraction) hallucinated the party 'John Smith', which does not
> appear in the source document. This propagated to Step 4 (Summarization),
> which included 'John Smith' in bullets[5].*

> "The corrected output is pre-filled from ground truth. I agree with the
> diagnosis, so I confirm it."

Click **Confirm diagnosis**. The toast reads *Eval case case_… added (confirmed),
dataset now 10 cases*.

**2:30 – A failure no check can see.** Open `017_invoice_wingtip` (batch-01).
> "This invoice was summarized as a report. Not a single warning fired: the
> classifier was confidently wrong. The trace view only catches it because
> ground truth disagrees."

Flag it. In the dialog, point out that the human can **override** the step and
category, and edit the corrected output, for example `{"doc_type": "invoice"}`.
Save it.

**3:10 – The eval dataset grows (Eval dataset tab).** The new rows are
highlighted at the top: 11 cases.
> "Every confirmed flag becomes a regression test: the input, the failing step,
> the bad output and the human-corrected output."

Select **v3** and click **Run regression**.
> "We shipped a grounding guard in v3. Re-running the whole dataset shows the
> John Smith case is now fixed, the misclassification is still failing, and the
> trend line of known issues resolved moves up."

**3:50 – Analytics tab.**
> "This is what tells a product team where to invest: hallucination and intake
> problems lead, extraction is the step that fails most, the failure rate fell
> from 26% to 16% across two releases, and automated time to root cause is
> about 0.2 ms per trace."

Point at the step × category matrix and finish on the framing:
> "An observability system that reduces mean time to root cause for AI pipeline
> failures from hours of manual debugging to seconds of automated diagnosis."

## The same loop from the terminal

```bash
obspipe traces --limit 5
obspipe diagnose <trace-id-prefix>
obspipe flag <trace-id-prefix> --reason "who is John Smith?"
obspipe confirm <flag-id>                      # or --step/--category/--corrected '{...}'
obspipe eval run --pipeline-version v3
obspipe eval trend
obspipe analytics
```
