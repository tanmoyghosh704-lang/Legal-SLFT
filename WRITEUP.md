# Legal Small-Language-Model Alignment: SFT vs. DPO

A portfolio project fine-tuning a small open-weight LLM (`Qwen/Qwen2.5-3B-Instruct`) to
produce IRAC-format legal analysis of contract clauses, and rigorously comparing
zero-shot, supervised fine-tuning (SFT/QLoRA), and Direct Preference Optimization (DPO)
using two independent evaluation methods.

This document is a distilled, results-oriented summary. The full chronological
decision/difficulty log — every bug, every dead end, every reasoning step, written as
work happened rather than reconstructed afterward — lives in `LOG.md`. This write-up
exists to be read top to bottom; `LOG.md` exists to be defensible line by line in an
interview.

---

## 1. Problem and Approach

**Task**: given a contract clause and its category (e.g. "Non-Compete", "Exclusivity",
"Cap On Liability"), produce a structured IRAC (Issue, Rule, Application, Conclusion)
legal analysis — grounded strictly in the clause text, with no invented facts, case
citations, or conditions.

**Core question**: how much does each stage of alignment (SFT, then DPO) actually
improve output quality over zero-shot, and does DPO need an SFT warm-start to work, or
can it get most of the way there alone?

**Constraint**: all heavy compute (data generation, training, judging) had to run on
free Kaggle GPU time — local hardware (a 4GB GTX 1650) could not fit even 4-bit
inference of the models involved, let alone training.

**Five-run ablation grid** (`config.yaml`'s `runs:` list):

| run_id | method | precision | warm start |
|---|---|---|---|
| `baseline` | none (zero-shot) | — | — |
| `sft_qlora` | SFT | 4-bit QLoRA | — |
| `sft_lora_fp` | SFT | full bf16 precision | — |
| `dpo_from_base` | DPO | 4-bit QLoRA | none (fresh LoRA on base) |
| `dpo_from_sft` | DPO | 4-bit QLoRA | `sft_qlora` adapter |

---

## 2. Data Pipeline

**Source**: CUAD (Contract Understanding Atticus Dataset), loaded via
`theatticusproject/cuad-qa` at `revision="refs/convert/parquet"` (the canonical
dataset's auto-generated Parquet export — HF deprecated script-based dataset loading
entirely in `datasets` 5.0.1, which broke the originally-planned loading path and
required finding this alternative).

- **22,450 rows**, 408 unique contracts, 41 clause categories.
- Excluded 5 purely-administrative categories (`Parties`, `Document Name`,
  `Agreement Date`, `Effective Date`, `Expiration Date`) — these are extraction tasks
  with no legal "Issue" to raise; forcing IRAC structure onto them would teach the
  model to produce vacuous, templated analysis.
- **Input unit is the extracted clause span, not the full contract** — the full
  contract averages ~8,800 words (up to ~27,000), far outside a usable sequence budget;
  the extracted span is ~31 words median. This was a deliberate scope decision: analysis
  quality on a given clause, not clause retrieval from a full document, is what SFT/DPO
  are being compared on.
- **7,620 (category, clause) rows** queued for teacher generation across 36 categories.

### Teacher generation (building the SFT targets)

- **Model**: `qwen2.5:7b-instruct-q4_0`, served via **Ollama** (not raw `transformers`).
- This was a pivot, not the original plan — see §4 for why raw `transformers` batching
  was abandoned entirely after four separate bugs.
- **Result: 7,620/7,620 rows generated, 100% `parse_ok`, 0 duplicate ids**, all 36
  categories and 384 contracts represented. Target lengths 66–280 words.

### Cleaning and splitting

- **Contract-level train/valid/test splits** (never clause-level) to prevent leakage —
  a clause from one contract must never appear in both train and test.
- Manual review of a 60-row stratified sample (25 lowest-rubric-scoring + 35 random)
  surfaced five real data-quality issues (see §5), leading to:
  - `is_substantive_clause()` — filters out non-substantive CUAD extraction fragments
    (e.g. a clause that's just the word "This").
  - `find_duplicate_documents()` — detects the same underlying filing uploaded twice
    under different contract-title strings (confirmed 2 real cases), merging them
    *before* split assignment so a duplicate can't land in both train and test under
    different names. Deliberately conservative (requires ≥3 shared identical clauses
    **and** matching normalized company-name prefix) — a naive shared-clause-count
    threshold alone would have wrongly merged two genuinely unrelated companies that
    happened to reuse the same boilerplate template.
- **Final dataset: 7,516 rows** (11 confirmed-bad + 93 fragment rows removed from
  7,620) → **5,987 / 726 / 803** train/valid/test, across 307/38/39 contracts.

### DPO preference pairs

- **7,516 pairs**, one per cleaned SFT row: `chosen` = the same clean teacher target,
  `rejected` = the same target run through one of **six programmatic corruption types**
  (drop a section, reorder sections, insert a fabricated citation, insert an unsupported
  claim, contradict the conclusion, truncate).
- **Explicit limitation, stated plainly rather than glossed over**: these are
  programmatic corruptions of already-correct answers, not real human preference
  judgments — a narrower, more mechanical objective than genuine RLHF preference data.
- Corruption type seeded deterministically per row id via `hashlib.sha256` (**not**
  Python's builtin `hash()`, which is randomized per-process unless `PYTHONHASHSEED` is
  fixed — this would have made corruption type silently non-reproducible across runs,
  caught before it shipped).
- Same contract-level split applied identically to the DPO pairs; verified directly
  (not assumed) that SFT and DPO splits agree exactly on which row ids land in
  train/valid/test.

---

## 3. Evaluation Design: Dual Metric

The project's evaluation deliberately uses **two independent metrics**, because each has
a real, different blind spot.

### 3.1 Deterministic rubric (`src/eval/rubric.py`)

Four components, reported separately (never blended into a single number that hides
which capability actually improved):

- **`structure_ok`**: did the output parse into all four IRAC sections, in order,
  non-empty.
- **`groundedness`** (the most important single check): (a) any quoted span in the
  Application section must be a real substring of the source clause — falls back to a
  word-overlap heuristic when the model paraphrases instead of quoting, a documented,
  known limitation; (b) the Rule section must not contain invented legal authority
  (named courts/statutes/"held that"/qualified "precedent" phrases). Combined with
  `min()` — either failing sinks the score.
- **`length_ok`**: graded, not a hard cutoff — catches truncation and rambling/repetition.
- **`no_contradiction`**: heuristic term-pair contradiction check between Application
  and Conclusion, plus explicit self-reversal-phrase and self-hedging-conclusion checks.

**This rubric needed six real bug fixes**, all found either by an integration test (every
DPO corruption type must score lower than its clean source under the same rubric — this
caught two real coverage gaps immediately) or by running it against the full real
7,620-row dataset and manually investigating every low-scoring row rather than trusting
the aggregate number. Every bug was a heuristic being *too broad* — matching a
technically-true keyword in a context that didn't mean what the heuristic assumed (e.g.
flagging "condition precedent," a standard contract-law term of art, as invented
case-law "precedent"). Final state: 35+ unit tests, full-dataset aggregate 0.9874.

### 3.2 LLM-as-judge (`src/eval/llm_judge.py`)

- Judge model: same `qwen2.5:7b-instruct-q4_0`, served via Ollama.
- **Three dimensions, 1–5 each, not a single quality number**: `groundedness`,
  `reasoning_quality` (is Rule→Application→Conclusion actually coherent and legally
  sound, not just present), `overall`. A single score would collapse exactly the
  distinction — well-grounded-but-poorly-reasoned vs. well-reasoned-but-ungrounded —
  this second metric exists to surface.
- Judged a **seeded common 150-id subsample** shared identically across all 5 runs
  (750 judge calls total), not the full 803×5 = 4,015 — the same
  scope-down-without-losing-signal reasoning used for the manual review sample.
- Validated locally on 3 real rows before any Kaggle run: the judge gave genuinely
  differentiated scores with specific, substantive rationale, not rubber-stamped 5s.

### 3.3 Manual validation of the judge itself

A 35-row stratified sample (15 lowest-judge-score rows, 10 rows where rubric and judge
disagreed most, 10 random) was built specifically to validate the judge's own
reliability. It was reviewed via an external AI tool, **explicitly self-labeled as
model-generated triage, not human ground truth** ("do not report as inter-rater
agreement"). Three of its most consequential claims were independently re-verified
against the raw clause/response text before being trusted — see §6.3.

---

## 4. Infrastructure and Training

### 4.1 Method

- **QLoRA** (4-bit NF4 quantization + LoRA, rank 8, alpha 16, dropout 0.05, target
  modules `q/k/v/o_proj`) for `sft_qlora`, both DPO runs.
- **SFT**: `trl.SFTTrainer`, prompt/completion dataset format (automatic
  completion-only loss masking), 3 epochs, effective batch size 16, lr 2e-4.
- **DPO**: `trl.DPOTrainer`, beta 0.1, 1 epoch, effective batch size 8, lr 5e-5. Warm
  start (`dpo_from_sft`) continues the *same* LoRA weights from `sft_qlora` via
  `PeftModel.from_pretrained(..., is_trainable=True)` rather than merging first (not
  straightforward with a 4-bit-quantized base) — `DPOTrainer` uses the model itself,
  with adapters disabled, as its own implicit reference model when it's already a
  `PeftModel`, avoiding a second full model copy in memory.
- **`max_seq_length: 1024`** — right-sized by directly tokenizing every real
  prompt+target across all three splits rather than trusting a round-number default;
  true max was 983 tokens, so the original 2048 config was never once being used.
- Every training run is driven by **one parameterized script** (`train_sft.py`,
  `train_dpo.py`) via CLI flags, not per-run copies — the four run variants (QLoRA vs.
  full precision, warm-started vs. not) are expressed entirely through
  `--load-in-4bit` and `--adapter` being present or absent.

### 4.2 The debugging chain that made real Kaggle training runs work

This is the single richest source of "what went wrong and how you found it" material in
the project. Condensed timeline, full detail in `LOG.md`:

1. **Teacher generation, raw `transformers` path (4 bugs)**: right-padding silently
   corrupting the KV cache for batched decoder-only generation (~46% parse-failure
   rate); `device_map="auto"` splitting a 7B model pipeline-parallel across a Kaggle
   T4×2 session (~15.6s/row, ~30 more hours needed against a 9-hour session cap);
   unbounded `max_new_tokens` letting one rambling response hold an entire batch
   hostage; bf16 weights alone (~14GB) leaving no memory headroom on a single T4.
   **Resolution**: abandoned raw `transformers` batching entirely in favor of
   Ollama-on-Kaggle, whose own scheduler handles padding/batching/quantization
   internally — this was the single highest-leverage infrastructure decision in the
   project.
2. **Local SFT smoke test caught a systematic loss-masking bug**: `trl`'s
   `SFTTrainer` computes its completion-only loss mask by tokenizing the prompt alone,
   then slicing the jointly-tokenized prompt+completion at that length. With no
   separator between the prompt (ending `"...section."`) and every completion
   (starting `"Issue:"`), BPE merged the boundary into one token, landing on the
   *masked* side — **every training example was silently never trained to predict its
   own first token**, since every target starts with `Issue:`. Found by reading `trl`'s
   installed source directly, not trusting the warning's own vague message. Fixed at
   the template source (`CANONICAL_PROMPT_TEMPLATE`), then regenerated the `prompt`
   field across every already-built dataset file.
3. **First real Kaggle training attempt: OOM.** `train_sft.py` never set
   `gradient_checkpointing` — invisible at the 0.5B smoke-test scale, fatal at 3B.
   Fixed, plus reduced batch size as extra margin given the failure left <100MB free.
4. **Second attempt: 12-hour session timeout at ~230s/step** (should be a few
   seconds). Root cause: `train_sft.py` passed a bare model-id string to `SFTTrainer`
   with no `device_map`, letting a Kaggle "T4×2" session split the model
   pipeline-parallel across both GPUs — the *exact same bug class* already fixed once
   for teacher generation, recurring in a code path that hadn't been touched yet.
5. **Third attempt: still ~117–125s/step even with the device pinned correctly**
   (confirmed via an explicit per-GPU memory diagnostic). Evaluated and adopted
   **Unsloth** (custom Triton kernels, purpose-built for exactly this "QLoRA on a
   single Kaggle/Colab T4" scenario) as an opt-in accelerated path, kept alongside the
   original vanilla path rather than replacing it, since Unsloth couldn't be validated
   at all on the local dev machine (a hard PyTorch version floor unrelated to the
   project's own code).
6. **First Unsloth run crashed on a `transformers` validation check**: `"Your setup
   doesn't support bf16/gpu... You need Ampere+ GPU."` This was the actual root cause
   of the mystery ~120–230s/step slowness all along — **T4/Turing lacks Tensor Core
   bf16 acceleration** (only Ampere+ does); the vanilla path had been hardcoding
   `bf16=True` since the very first version of the script and running, unaccelerated,
   the entire time, with no error, just extreme slowness. Fixed by computing GPU
   compute-capability at runtime and only using bf16 on Ampere+, falling back to fp16
   otherwise (scoped to the Unsloth branch only, to avoid reintroducing a separate,
   never-fully-diagnosed fp16/GradScaler dtype-mismatch crash in the vanilla path).
7. **Result**: `sft_qlora` trained end-to-end at a stable ~7–8s/step (~2.3–2.5 hours
   total for 1,125 steps) — the payoff of six necessary, individually-insufficient
   fixes stacked together.
8. **`sft_lora_fp` (full-precision ablation)**: training succeeded cleanly (no OOM,
   despite ~4× the base-model memory footprint of the QLoRA runs), but eval crashed on
   an unrelated `peft`/`torchao` version incompatibility that only manifests when
   loading a LoRA adapter onto a *full-precision* base (every prior run's 4-bit base
   took a different internal code path that never reached the broken check). Recovered
   without re-training: the trained adapter had already been saved to disk before the
   crash, so a minimal eval-only notebook was built instead of burning another 3+ hours.
9. **Two silent-wrong-answer bugs, caught before they mattered**: a Kaggle adapter
   "auto-discovery" helper returned the *first* directory matching a content marker
   across all five notebooks — when a dry-run checkpoint or an intermediate epoch
   checkpoint was also present in an upload (both valid, expected artifacts), directory
   traversal order (not guaranteed) could have silently selected the wrong,
   under-trained adapter with **no error at all** — the worst kind of bug for a project
   whose entire point is trustworthy numbers. Fixed to require exactly one match after
   explicitly filtering dry-run/checkpoint paths, erroring loudly on any ambiguity
   rather than guessing.

---

## 5. Data Quality Findings (Manual Review)

A 60-row stratified human review (25 lowest-rubric-scoring + 35 random) of the
teacher-generated SFT dataset — explicitly treated as a triage pass calling for a
domain-competent human read, not just an automated check — found five real issues no
automated rubric had caught:

1. **Non-substantive clause fragments** (CUAD occasionally extracts unusable spans —
   a single word, a mid-sentence truncation) that the teacher dutifully wrote a full
   IRAC analysis for anyway. Root-caused to clause *extraction*, not generation; fixed
   with a substantiveness filter.
2. **Two confirmed rubric blind spots**: a genuine self-contradiction with zero shared
   vocabulary between the conflicting claims (a purely lexical heuristic structurally
   cannot catch this), and a Conclusion that asserts something and hedges the *same
   point* in the same sentence.
3. **A fabricated general legal principle** (not a fabricated citation) — no cheap
   heuristic fix exists for this; it requires actual legal-domain understanding, which
   is precisely why the LLM-judge half of the evaluation design exists.
4. **The same clause sampled twice by the teacher** (temperature 0.3) producing two
   different readings — one correct, one with the contracting parties' roles swapped —
   and scoring identically under `groundedness` (0.0 for both), unable to distinguish
   "correct but low word-overlap" from "roles inverted."
5. **Unsupported gloss** in ~4–5/60 rows — inventing a condition (most often "without
   [Party]'s consent") not actually present in the clause. Added an explicit
   instruction against this to the teacher-generation template for any future
   regeneration (not retroactively applied to the already-generated, already-cleaned
   dataset).

A first attempt to deduplicate the dataset by bare `clause_text` found what looked like
910 duplicate rows — before trusting that number, checked whether the duplicates shared
a category, and found only 31/910 did; the other 893 were legitimately different
training prompts (same clause text, different category label, e.g. the same clause
appearing under both "Minimum Commitment" and "Volume Restriction"). Investigating *why*
led to the real finding instead: two pairs of contract filings in the dataset are the
same underlying document uploaded twice under different filenames — a genuine
train/test-leakage risk under a different name, fixed via the document-deduplication
logic described in §2.

---

## 6. Results

### 6.1 Full results table (deterministic rubric, full 803-row test set)

| run_id | structure_ok | groundedness | length_ok | no_contradiction | **aggregate** | parse_ok_rate |
|---|---|---|---|---|---|---|
| baseline (zero-shot) | 0.994 | 0.731 | 0.994 | 0.991 | 0.927 | 0.994 |
| sft_qlora | 1.000 | 0.951 | 1.000 | 1.000 | 0.988 | 1.000 |
| sft_lora_fp | 1.000 | 0.947 | 1.000 | 0.995 | 0.986 | 1.000 |
| dpo_from_base | 1.000 | 0.976 | 1.000 | 1.000 | 0.994 | 1.000 |
| **dpo_from_sft** | 1.000 | **0.985** | 1.000 | 1.000 | **0.996** | 1.000 |

### 6.2 LLM-judge results (150-row common subsample, 1–5 scale, 100% parse rate all runs)

| run_id | judge_groundedness | judge_reasoning_quality | judge_overall |
|---|---|---|---|
| baseline | 4.63 | 4.64 | 4.63 |
| sft_qlora | 4.79 | 4.81 | 4.79 |
| sft_lora_fp | 4.77 | 4.79 | 4.76 |
| dpo_from_sft | 4.81 | 4.80 | 4.79 |
| dpo_from_base | 4.83 | 4.84 | 4.82 |

### Key findings

**1. SFT closes the groundedness gap, and it's the dominant effect.** Zero-shot
`Qwen2.5-3B-Instruct` already follows the IRAC format almost perfectly
(`structure_ok` 0.994 even before any tuning) — format compliance was never the actual
gap. Groundedness was: 0.731 → 0.951 after SFT alone, a +0.220 absolute jump, by far the
largest single move in the whole grid.

**2. DPO improves further on top of SFT, concentrated on the same dimension.**
`dpo_from_sft` moves groundedness another +0.034 (0.951 → 0.985) and crosses 0.99
aggregate. Each alignment stage contributes a further, real improvement specifically on
the metric the rubric design treats as most important — not just incremental gains
spread thinly across everything.

**3. DPO alone gets surprisingly close to the full pipeline, without ever going through
SFT.** `dpo_from_base` (0.994 aggregate, 0.976 groundedness) came within 0.002
aggregate / 0.009 groundedness of the full `dpo_from_sft` pipeline (0.996 / 0.985) —
and clearly beat SFT alone (`sft_qlora`: 0.988 / 0.951) despite never being
supervised-fine-tuned first. Plausible mechanism, not just a surprising number: the
`chosen` side of every DPO pair *is* the same clean teacher target SFT trains on, so
`dpo_from_base` implicitly learns to reproduce the same targets SFT does, *plus* it gets
an explicit contrastive signal (what to avoid, via the six corruption types) that
vanilla SFT never sees. This is exactly the kind of result worth running the full
ablation grid to find, rather than assuming from first principles — the project's own
original config had `dpo_from_sft` marked `minimum_viable: true` and `dpo_from_base` as
the optional ablation, i.e., DPO-needs-SFT-first was the default assumption going in.

**4. QLoRA's 4-bit quantization cost nothing measurable.** `sft_lora_fp`
(full bf16 precision) scored marginally *below* `sft_qlora` (4-bit) on every dimension
that differed — aggregate 0.986 vs. 0.988, groundedness 0.947 vs. 0.951 — the opposite
of the naive expectation that quantization noise should only ever hurt. The gap is
small enough (0.002 aggregate) to be within single-run noise rather than a proven
effect, so the honest claim is "quantization was free here," not "full precision is
worse."

**5. The two evaluation metrics agree on direction but diverge sharply on magnitude —
and that divergence is itself a real, substantive finding.** The rubric shows a
dramatic baseline→tuned groundedness jump (0.731 → ~0.95–0.98, ~30% relative). The LLM
judge sees a much smaller one (4.63 → ~4.8/5, ~3–4% relative), and the four tuned
variants are clustered too tightly (4.76–4.84) for their relative ranking to be
meaningful at 150 samples — the judge even ranks `dpo_from_base` fractionally above
`dpo_from_sft`, the opposite order from the rubric.

### 6.3 Why the two metrics diverge: a real, verified LLM-judge reliability problem

Rather than treat this divergence as unexplained noise, both directions were
investigated. First: is the judge just being lenient/broken? Reading the rationale
behind the judge's own lowest-scoring baseline rows showed it correctly identifying a
genuine legal misreading (a most-favored-nation-styled clause the zero-shot model had
actually misinterpreted) and scoring it 2/2/2 with an accurate explanation — so the
judge is making real distinctions, just on a more forgiving absolute scale than the
rubric.

Second, and more importantly: a 35-row manual validation sample of the judge's own
scores (stratified: 15 lowest-judge-score rows, 10 largest rubric/judge disagreements,
10 random) surfaced a specific, checkable **groundedness blind spot**, and three of its
most consequential claims were independently re-verified against the raw source text
before being trusted:

- The judge scored a response **groundedness 5/5** and wrote **"there are no invented
  facts"** — while the response had repeatedly referred to the clause's actual party
  ("LMG") as a fabricated entity, **"Mergers and Acquisitions (MLG)"**, invented out of
  nowhere.
- The judge scored another response **4/5**, faulting it only for being imprecise —
  while the response had invented a specific carve-out contradicting the clause's own
  **"under no circumstances"** language.
- The judge penalized a response for "not fully capturing the requirement for a sworn
  statement" — a **factually false critique**: the response named that exact
  requirement, twice.

**Implication for the write-up, stated plainly**: the LLM judge's groundedness scores —
on the exact dimension this project's evaluation design treats as most important —
should not be taken at face value. The deterministic rubric, despite its own documented
false-positive risk on legitimate paraphrases, is checking something mechanical
(quote/word-overlap against the actual clause) rather than being satisfied by fluent,
well-structured prose regardless of fidelity. Neither metric dominates the other — the
judge is better at catching reasoning-level errors the rubric structurally cannot see
at all (e.g. a genuine legal misreading with no lexical overlap to flag it), while the
rubric is more reliable specifically on fabrication/hallucination detection. This is a
real methodological limitation of LLM-as-judge for this task, reported as a finding, not
smoothed over.

**Honesty note on the validation sample itself**: the 35-row review was completed via
an external AI tool, self-labeled explicitly as "model-generated, not human... do not
report as inter-rater agreement." The `human_verdict`/`human_notes` columns in
`data/review/llm_judge_validation_sample.csv` remain unfilled by an actual human. The
three findings above are reportable because they were independently re-verified against
primary source text (the actual clause and response), not because the AI triage pass
said so — but a genuine human pass remains the more defensible thing to cite as
"validated" if time allows before any external presentation of this work.

---

## 7. Limitations

- **Single run per configuration.** Every number in §6.1/6.2 comes from one training
  run, not a multi-seed average. Rankings that differ by ~0.002–0.01 (e.g.
  `dpo_from_base` vs. `dpo_from_sft`, `sft_qlora` vs. `sft_lora_fp`) should be read as
  directionally suggestive, not statistically proven.
- **DPO preference pairs are programmatic, not human-labeled.** The `rejected` side of
  every DPO pair is a deterministic corruption of the `chosen` side, not a real human
  preference judgment — narrower and more mechanical than genuine RLHF data.
- **The rubric's groundedness check has a known false-positive mode** on legitimate
  paraphrases that don't directly quote the clause (falls back to a looser word-overlap
  heuristic).
- **The LLM judge has a demonstrated false-negative mode on groundedness** — see §6.3.
- **The "human" judge-validation pass was AI-generated triage**, independently
  spot-checked on its most consequential claims but not a full human review.
- **LLM-judge scoring covers a 150-row common subsample, not the full 803-row test
  set** — a deliberate cost/signal tradeoff, not full per-row coverage.

## 8. What's Left

- **Done:** FastAPI serving of the best adapter (`dpo_from_sft`) — `src/serving/app.py`,
  `/health` and `/analyze` routes, 4-bit quantized load with `device_map={"":0}`,
  configurable adapter via `ADAPTER_NAME`, reuses the same prompt template, IRAC
  parser, and clause-substantiveness filter as training/eval. Validated end-to-end
  locally: well-formed IRAC output on a real clause, correct `422` rejection on a
  fragment. An MCP wrapper around this service remains optional/not started.
- A genuine human pass on the judge-validation sample, if time allows, to upgrade the
  §6.3 findings from "AI-triaged, Claude-verified" to fully human-validated.

---

*Full chronological detail, every bug and reasoning step as it happened: `LOG.md`.*
