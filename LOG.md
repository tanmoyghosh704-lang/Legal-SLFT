# Decision & Difficulty Log

Ongoing log of what was done, why, what went wrong, and how it was resolved.
Written as work happens, not reconstructed afterward. This is the primary
interview-prep artifact for this project — every entry should be something
that could be told as a story in an interview.

Entry format:

```markdown
## [Date] — [Phase / Run ID]

### What I did
### Why this approach (and what alternative was rejected, and why)
### Difficulty encountered (errors, dead ends, surprising results)
### How it was resolved (or: not resolved, and what the workaround was)
### What I'd do differently
```

---

## 2026-08-15 — Phase 0: Repo scaffold & environment setup

### What I did
- Initialized git repo at `Legal SLFT/`.
- Created repo structure: `src/{data,train,eval,serving}`, `notebooks/`,
  `data/{raw,splits}`, `adapters/`, `results/plots`, `tests/`.
- Decided platform stack: HuggingFace `transformers` + `trl` + `peft` +
  `bitsandbytes`, NOT the MLX stack that reference implementations of this
  idea use (MLX is Apple Silicon only; this machine is Windows + CUDA).
- Confirmed local hardware: RTX 1650, 4GB VRAM. Confirmed via project spec
  that even 4-bit QLoRA training of a 3B model will not fit in 4GB (~2GB
  for quantized weights alone, before gradients/optimizer state/activations).
  Decision: all real training runs happen on Kaggle (free T4, 16GB VRAM).
  Local machine is for data prep, rubric/eval code, plotting, and serving
  a downloaded adapter only.
- Confirmed Ollama is already installed locally with
  `qwen2.5:7b-instruct-q4_0` pulled (4.4GB) — this will serve as both the
  SFT teacher model (generating IRAC targets) and the LLM-as-judge for
  evaluation. Also have `qwen2.5:0.5b-instruct-q4_0` available, which is
  a candidate for the local smoke-test model (Build Order step 7) instead
  of downloading a separate Qwen2.5-0.5B-Instruct from HF.

### Why this approach (and what alternative was rejected, and why)
- Base model decision: requested gated access to `meta-llama/Llama-3.2-3B-Instruct`
  today (approval can take hours to days). Rather than sit idle waiting,
  the plan is to build and validate the entire pipeline (data prep, rubric,
  smoke test, Kaggle notebook skeleton) against `Qwen2.5-3B-Instruct`
  (ungated) first, then swap the model name in `config.yaml` once Llama
  access is granted. Model name is a config parameter everywhere, not
  hardcoded, specifically to make this swap free.
  - Alternative considered: wait for Llama access before starting. Rejected
    — no reason to block days of pipeline-building work on an approval
    queue outside my control.
- Git initialized before any code, so LOG.md and every subsequent decision
  is versioned from day 1, not reconstructed later.

### Difficulty encountered (errors, dead ends, surprising results)
- None yet — this is the setup entry.

### How it was resolved
- N/A

### What I'd do differently
- N/A yet.

---

## 2026-08-16 — Phase 1: CUAD dataset exploration

### What I did
- Attempted to load `theatticusproject/cuad-qa` (the dataset id originally
  placeholder'd in `config.yaml`) via `datasets.load_dataset(..., trust_remote_code=True)`.
- Found a parquet-native mirror instead — `chenghao/cuad_qa` — and used it.
- Computed dataset statistics on the full train split (11,178 rows):
  - 408 unique contracts (`title` field) — this is the unit for contract-level
    splitting.
  - 41 unique clause categories (`question` field), e.g. `Governing Law`,
    `Cap On Liability`, `Exclusivity`, `Non-Compete`, but also purely
    administrative ones: `Parties`, `Document Name`, `Agreement Date`,
    `Effective Date`, `Expiration Date`.
  - 0% empty answers — this mirror only includes rows where the category's
    clause is actually present in the contract (unlike the original CUAD
    SQuAD2.0-style release, which includes negative/unanswerable rows). No
    filtering needed for that.
  - `context` (full contract text) is very long: mean ~8,838 words, max
    ~26,842 words — far too long to use as the model's input directly at a
    2048-token sequence length.
  - The extracted clause span (`answers.text`) is short: median 31 words,
    p90 91 words, p99 208 words, max 479 words. Every row has exactly one
    span (no multi-span rows in this mirror).
- **Decision: SFT/DPO input is the extracted clause span, not the full
  contract.** Template will be: input = (clause category, clause span
  text), target = IRAC analysis of that clause. This fits comfortably in
  the sequence budget alongside the IRAC output.
- **Decision: exclude 5 purely-administrative categories** (`Parties`,
  `Document Name`, `Agreement Date`, `Effective Date`, `Expiration Date`)
  from the IRAC dataset. These are extraction tasks with no legal "Issue"
  to raise — forcing IRAC structure onto "what is the document name" would
  produce vacuous, templated Issue/Rule sections that teach the model
  nothing and would be an easy, uninteresting 100% on the rubric. The
  remaining 36 categories (Governing Law, Cap On Liability, Exclusivity,
  IP Ownership Assignment, Termination For Convenience, Audit Rights,
  Insurance, License Grant, Non-Compete, etc.) all raise genuine
  substantive legal questions.

### Why this approach (and what alternative was rejected, and why)
- Considered using the full contract as context and letting the model find
  the relevant clause itself (closer to a real-world retrieval setting).
  Rejected for this phase: it conflates two different capabilities (clause
  retrieval vs. clause analysis) and would blow the sequence budget for a
  3B model on Kaggle T4. Using the pre-extracted span keeps the task
  focused on analysis quality, which is what's actually being compared
  across SFT/DPO. This is a real scope-narrowing decision — worth stating
  plainly if asked "why not feed the whole contract?".
- Considered using `theatticusproject/cuad` (the original raw CUAD_v1.json +
  PDFs). Rejected for now: would require PDF text extraction and manual
  SQuAD-style span alignment work that `chenghao/cuad_qa` already did
  correctly, for no benefit at this stage. Original PDFs remain available
  if a later need (e.g. re-deriving different spans) comes up.

### Difficulty encountered (errors, dead ends, surprising results)
- `theatticusproject/cuad-qa` failed to load: `datasets` 5.0.1 has fully
  removed support for script-based datasets (`RuntimeError: Dataset
  scripts are no longer supported, but found cuad-qa.py`). The
  `trust_remote_code=True` escape hatch that used to allow this in older
  `datasets` versions no longer exists — the library printed a warning
  that the flag "is not supported anymore" and failed anyway. This is a
  breaking change in the HF ecosystem (loading scripts deprecated in favor
  of Parquet-native datasets) that isn't obvious until you hit it.
- Surprising result: expected some fraction of empty/negative answers
  (CUAD's original release is SQuAD2.0-style with unanswerable questions)
  but found 0% in this mirror — worth remembering that different
  community mirrors of the same dataset are not always faithful 1:1
  copies of the original release; the row-construction choices differ.

### How it was resolved
- Switched to `chenghao/cuad_qa`, a community re-upload of the same
  underlying CUAD data in Parquet format with an identical schema (`id`,
  `title`, `context`, `question`, `answers`, `source`, `lan`). Verified by
  sampling rows that the content matches the expected CUAD structure
  (clause categories, contract titles, extractive answer spans).

### What I'd do differently
- Would check a dataset's file format (script vs. parquet) via
  `HfApi.dataset_info(...).siblings` *before* attempting to load it, to
  skip the failed load attempt entirely.

---

## 2026-08-16 — Phase 1: IRAC template, teacher generation, Kaggle handoff

### What I did
- Built `src/data/build_irac.py`: the canonical student-facing prompt
  template (used identically for SFT input, DPO prompt, baseline
  zero-shot eval, and serving), a separate (stricter) teacher-generation
  prompt with explicit groundedness constraints, an `IRAC_PARSE_RE`-based
  parser (`parse_irac`) that returns `None` on malformed output instead of
  guessing, and `load_cuad_filtered()` / `build_queue()` to turn CUAD rows
  into a generation queue JSONL.
- Ran `build_queue()` twice: once with `--cap-per-category 3` (108 rows)
  as a fast correctness check, then uncapped for the real queue —
  **7,619 rows across 36 categories** written to
  `data/raw/sft_generation_queue.jsonl`.
- Built `src/data/generate_teacher_targets.py` with two backends behind
  one shared parsing/validation/resume path: `ollama` (local, for smoke
  testing only) and `hf` (batched HF `transformers` generation, GPU — this
  is what will actually run on Kaggle). Resumable by id so a Kaggle
  session that hits its runtime cap can pick up where it left off.
- **Benchmarked local Ollama generation before committing to a plan:**
  single-call latency 38s (10.7s model load + 23.8s generation), warm
  throughput ~10.3 tok/s. `ollama ps` showed the 4.9GB q4 model split
  53%/47% CPU/GPU — it does not fit in the RTX 1650's 4GB VRAM, so part of
  every forward pass runs on CPU. At this rate, generating all 7,619
  eligible rows locally would take **~55 hours** — not viable within the
  project timeline.
- Smoke-tested the `ollama` backend end-to-end on 5 real queue rows
  (mixed categories) before deciding anything about scale: 5/5 parsed
  cleanly on the first try (0 retries needed). Manually read all 5 outputs
  — grounded in the clause text, no invented case law, IRAC sections
  logically consistent. One data-quality issue spotted in the *source*
  CUAD row itself (not the generation): a row labeled category
  `Renewal Term` had clause text that was actually about data protection
  obligations, not renewal — CUAD's per-category span extraction isn't
  perfectly clean. Noting this for the manual SFT-quality-review step
  (spec section 3.1) — expect a small percentage of mislabeled-category
  rows and decide then whether to filter them.
- Also double-checked an apparent `�` in printed output — confirmed via
  `chr(0xFFFD) in target` that it's a Windows terminal display artifact
  (git-bash not rendering a curly apostrophe U+2019), not actual data
  corruption; the stored JSONL is valid UTF-8.

### Why this approach (and what alternative was rejected, and why)
- **User explicitly redirected all heavy computation to Kaggle**, not just
  training — this changes the plan from the original spec, which assumed
  local Ollama would handle teacher generation too ("free, no token
  limits"). Given the measured ~10 tok/s local throughput, this redirect
  is also the practically correct call, not just a preference: 55 hours
  locally vs. an estimated 2.5–4.5 hours on a Kaggle T4 doing batched
  bf16 generation (rough estimate: batch=8, ~150–250 tok/s aggregate,
  ~2.3M output tokens total for 7,619 rows × ~300 tokens each).
  Still "free, no token limits" — just using Kaggle GPU-hours instead of
  local CPU cycles for this step, which is a hardware-driven adaptation
  worth stating plainly if asked why this deviates from a common
  reference approach.
- Kept the `ollama` backend in the same script rather than writing
  Kaggle-only generation code, so the generation/parsing/validation logic
  used to smoke-test 5 examples locally is *exactly* the logic that runs
  at scale on Kaggle — no drift between a "test version" and a "real
  version" of the prompt or parser.
- Chose to build the **full uncapped queue (7,619 rows)** rather than a
  stratified per-category cap, now that generation moved to Kaggle GPU and
  the time cost dropped from ~55 hours to an estimated few hours — full
  scale is more defensible for the results without a meaningful time
  penalty once the compute moved off this laptop.
- Deliberately smoke-tested on 5 *real* rows pulled from the actual queue
  (not hand-written toy examples) before writing the Kaggle notebook —
  catches prompt/parsing issues against the real data distribution, not
  an idealized case.

### Difficulty encountered (errors, dead ends, surprising results)
- The 4GB VRAM constraint bit here too, not just for training (see Phase
  0) — even *inference* with a 7B q4 model doesn't fully fit locally.
  Confirms the VRAM ceiling on this machine is lower than expected across
  both training and any full-size-model inference; the 0.5B/1.5B models
  are the only ones this laptop can run entirely on GPU.
- CUAD source data has occasional category/clause-text mismatches (see
  the "Renewal Term" example above) — a real data quality issue in the
  upstream dataset, not something introduced by this pipeline.

### How it was resolved
- Moved teacher generation to a Kaggle notebook (`notebooks/kaggle_teacher_gen.ipynb`,
  next step) using the `hf` backend of `generate_teacher_targets.py`.
  Local Ollama is retained solely as the fast validation loop for prompt
  and parser changes on a handful of rows before spending Kaggle GPU time.

### What I'd do differently
- Would benchmark generation throughput *before* writing any generation
  code, not after — in this case it worked out fine since the benchmark
  came right before the full script, but doing it first would have saved
  writing an `ollama`-only version that then needed a second backend
  bolted on.

---

## 2026-08-17 — Phase 1: Pivot off raw `transformers` generation → Ollama-on-Kaggle

### What I did
- Came across an external reference implementation (a public repo doing a
  similar CUAD/LegalBench IRAC-generation pipeline) while looking for a
  more robust CUAD loading path. Reviewed it for architecture ideas, not
  to copy wholesale — see "Why this approach" below for where the line was
  drawn and why.
- **Adopted, independently verified:** `theatticusproject/cuad-qa` loads
  correctly via `revision="refs/convert/parquet"` — HF's auto-generated
  Parquet export of the *canonical* dataset repo, not a community mirror.
  Verified myself by loading it directly (22,450 rows, includes the
  negative/unanswerable examples the `chenghao/cuad_qa` mirror had
  stripped). Switched `load_cuad_filtered()` in `build_irac.py` off
  `chenghao/cuad_qa` onto this. Rebuilt the queue: 7,620 rows across the
  same 36 categories (vs. 7,619 before) — confirms the two sources are the
  same underlying CUAD data, this is a provenance/robustness upgrade, not
  a data change.
- **Adopted, own reasoning:** rewrote `generate_teacher_targets.py` to
  drop `generate_batch_hf()` (the raw-`transformers` batching path)
  entirely and run *only* through Ollama's HTTP API — the same code path
  already used and validated for local smoke-testing, now with a
  `ThreadPoolExecutor`-based `--concurrency` flag so multiple requests hit
  the Ollama server in parallel. Ollama's own scheduler (`OLLAMA_NUM_PARALLEL`)
  handles GPU batching, padding, and KV-cache management internally.
  Updated `notebooks/kaggle_teacher_gen.ipynb` to install Ollama, start
  `ollama serve` on the Kaggle T4 with `OLLAMA_NUM_PARALLEL=8`, pull
  `qwen2.5:7b-instruct-q4_0`, do a warmup call + `ollama ps` check to
  confirm GPU placement *before* the real run (rather than discovering a
  CPU fallback 3 hours in), then call `run(..., concurrency=8)`.
- **Not adopted:** the reference repo's LegalBench integration, DPO
  corruption logic, hand-picked `TRICKY_CLAUSES`, and IRAC prompt
  template. Those are someone else's design decisions and debugging work
  (their code comments cite specific verified stats from runs I didn't
  do — e.g. "48/300 duplicate prompts," "290/300 repetition in an earlier
  run"). Reusing that logic without having lived through those failures
  myself would mean claiming debugging history in an interview that isn't
  mine. The corruption functions and DPO pair design are still to be
  built from this project's own spec (section 3.2), independently.

### Why this approach (and what alternative was rejected, and why)
- Drew a deliberate line between "generic infrastructure" (JSONL I/O,
  YAML config loading, the `data/raw/` vs `data/datasets/` cache/derived
  split, HF's `refs/convert/parquet` mechanism) — which nobody owns and
  is fair to reuse or reimplement freely — and "the actual intellectual
  content of the project" (prompt/template design, corruption logic,
  evaluation design, the specific bugs hit and how they were diagnosed).
  The latter has to be genuinely mine, since `LOG.md` exists specifically
  to be defensible in an interview. Verified the one technical claim I
  did adopt (the parquet-branch fix) myself before trusting it, rather
  than taking the reference repo's word for it.
- Considered keeping `generate_batch_hf()` around as a documented-but-
  unused alternative path (for narrative completeness). Rejected: dead
  code that will rot, and the four bugs already fixed there are
  permanently recorded in this log regardless of whether the code stays
  — keeping broken/superseded code around adds no value once the fix
  is understood and written up.
- Considered switching to LegalBench's CUAD subset (`nguha/legalbench`,
  38 per-category Yes/No classification tasks) instead of CUAD-QA
  directly. Rejected: verified it's the *same* underlying CUAD source
  data reshaped into a binary-classification format, not new coverage —
  switching would mean re-deriving `(clause, category)` pairs through an
  extra filtering step (`answer == "Yes"` per config) for no gain over
  the extractive QA format already in use.
- Abandoned the 440/7,619 rows already generated via the `chenghao/cuad_qa`
  + raw-`transformers` path rather than trying to preserve them — cheap
  to lose (~3h of Kaggle GPU time), and the row `id`s are tied to the old
  source's category-label format, not worth reconciling against the new
  canonical-source ids.

### Difficulty encountered (errors, dead ends, surprising results)
- None yet on the Ollama-on-Kaggle path specifically — restarting the
  Kaggle run with this rewrite is the next step, and the real test of
  whether `OLLAMA_NUM_PARALLEL` concurrency actually delivers the
  throughput needed to fit the full queue inside a 9h session.

### How it was resolved
- N/A yet — pending the actual Kaggle rerun.

### What I'd do differently
- N/A yet.

---

## 2026-08-16 — Phase 1: Kaggle teacher generation — two real bugs, caught mid-run

### What I did
- Launched `notebooks/kaggle_teacher_gen.ipynb` on Kaggle (GPU T4 x2
  session) against the full 7,619-row queue using
  `Qwen/Qwen2.5-7B-Instruct` in bf16 via `AutoModelForCausalLM.from_pretrained(..., device_map="auto")`.
- After ~3.1 hours (11,077s) and 832/7,619 rows, checked the run's health
  instead of letting it continue unattended:
  - Per-batch (8 rows) timing from consecutive log lines: batch
    824→832 took 125s → **15.6s/row**. Extrapolated remaining-time:
    (7619-832) rows × 15.6s ≈ **29.5 more hours** — but Kaggle GPU
    sessions cap at **9 hours**. This run would have been killed with
    ~77% of the queue never generated.
  - `parse_failures_so_far=380` out of 832 done — a **45.7% parse
    failure rate**, vs. 0/5 (0%) in the local Ollama smoke test on the
    same prompt template. Something in the Kaggle-specific `hf` backend
    path was producing malformed output at a rate the local validation
    never surfaced.
  - The Kaggle logs also repeated a `transformers` warning on every batch:
    *"A decoder-only architecture is being used, but right-padding was
    detected! For correct generation results, please set
    `padding_side='left'`."* — present but easy to miss/ignore among
    hundreds of repeated log lines.
- **Stopped the run** rather than let it either finish 29.5 hours later
  (impossible — session cap) or silently accept a ~46%-corrupted dataset.
- Diagnosed both root causes before touching code:
  1. **Padding side bug**: `generate_teacher_targets.py`'s `hf` backend
     tokenized the batch with default (right) padding. For decoder-only
     models, batched `generate()` with right-padding misaligns the KV
     cache position for every sequence shorter than the batch's longest
     one once generation starts appending new tokens after the padding —
     the model effectively continues generating from the wrong position
     for most of the batch. Only the longest sequence in each batch (no
     padding) generated correctly, roughly explaining the ~46% failure
     rate (varies with how much shorter each row's clause+prompt was than
     the batch max).
  2. **Multi-GPU pipeline-parallelism bug**: the Kaggle session had `GPU
     T4 x2`, and `device_map="auto"` split the 7B model's layers across
     both GPUs (pipeline parallelism) rather than placing it on one. A 7B
     model in bf16 is ~14GB — it fits entirely within one T4's 16GB.
     Autoregressive generation is inherently sequential layer-by-layer,
     so splitting layers across 2 GPUs means every single decoding step
     pays a cross-GPU (PCIe, not NVLink, on Kaggle's T4x2) transfer —
     this is almost certainly the dominant cause of the 15.6s/row rate,
     which is far slower than the ~2.5-4.5h/7,619-row estimate made
     before this run (that estimate assumed single-GPU generation).
- Fixed both in `src/data/generate_teacher_targets.py`:
  `tok.padding_side = "left"` before tokenizing, and
  `device_map={"": 0}` to pin the whole model to GPU 0.

### Why this approach (and what alternative was rejected, and why)
- Considered letting the run continue and filtering `parse_ok=False` rows
  out afterward. Rejected: doesn't fix the timing problem (still would be
  killed at 9h with most of the queue ungenerated), and discarding ~46%
  of expensive Kaggle GPU-hour output is wasteful when the actual fix is
  a two-line change.
- Considered keeping the 832 already-generated rows and only fixing the
  bug for the remainder. Rejected for this restart: the corrupted and
  valid rows are randomly interleaved within batches (depends on each
  row's relative sequence length vs. its batch's max), and re-verifying
  which of the 452 "parse_ok=True" rows from the buggy run are trustworthy
  isn't worth it when only 11% of the queue was processed — cheaper to
  restart clean with both fixes in place.

### Difficulty encountered (errors, dead ends, surprising results)
- The bug did not appear at all in local smoke testing (Ollama backend,
  5/5 clean). This is a real backend-parity gap: the `ollama` backend
  never batches (one HTTP request per row), so the right-padding bug is
  structurally impossible to hit there — it only exists in the batched
  `hf` path. Smoke-testing the `ollama` backend validated the *prompt
  template and parser*, but gave false confidence about the *`hf` batched
  generation path specifically*, which was never locally exercised before
  spending real Kaggle GPU-hours on it.
- The `transformers` library did warn about the padding issue, every
  single batch — but a repeated warning buried in hundreds of log lines
  is easy to scroll past. The `parse_failures_so_far` counter printed
  alongside it was the signal that actually prompted investigation.

### How it was resolved
- Both fixes applied locally (`padding_side="left"`, `device_map={"": 0}`)
  before re-uploading to Kaggle and restarting with a fresh session (fresh
  9-hour window) and a fresh output file (not resuming the corrupted
  partial run).

### What I'd do differently
- Should have smoke-tested the `hf` backend itself — even just 2-3 rows,
  batched, on any available GPU (could have used a local CPU-only run of
  a tiny model to at least exercise the batching/padding code path) —
  before the first real Kaggle run, not just the `ollama` backend. "The
  parsing logic is shared" is true, but *how the batch is constructed and
  padded* was backend-specific code that never ran until it hit Kaggle.
- Should have set a checkpoint to sanity-check progress (timing + failure
  rate) at, say, the 15-30 minute mark of a long unattended run, rather
  than only checking in ~3 hours later. Would have caught this with far
  less wasted GPU-hour.
- The device_map="auto" assumption ("auto will do something reasonable")
  was wrong for this specific case (small model, multi-GPU session) —
  worth remembering that "auto" placement strategies are tuned for
  *fitting* a model, not for *minimizing generation latency*; they can
  actively hurt throughput when the model would fit on one device anyway.

### Addendum — third fix before restart: max_new_tokens cap
- Before re-running, also reduced `max_new_tokens` from 400 to 320 and
  passed `eos_token_id` explicitly. Reasoning: HF's batched `generate()`
  only stops a batch early once *every* sequence in it has produced EOS —
  one rambling response holds the entire batch of 8 to the cap regardless
  of how quickly the other 7 finished. The local smoke test's real outputs
  ran ~245-280 tokens, so 320 keeps margin while bounding the worst case
  much tighter than 400 did. This is a pure efficiency change (bounds tail
  latency), not a dataset-size cut — decided to make this change
  regardless of what the restarted run's throughput turns out to be,
  since it's low-risk and directly addresses a mechanism (batch held
  hostage by its slowest member) rather than guessing at speed.
- Deliberately did *not* reduce the number of queued rows (7,619) at the
  same time — that's a different kind of tradeoff (weakens the SFT
  signal, thins the per-category breakdown) and shouldn't be decided
  before seeing whether the two correctness/placement fixes alone bring
  throughput into a workable range.

### Addendum — fourth bug: OOM on first forward pass (bf16 7B doesn't fit
### a T4 with any headroom)
- Restarted on Kaggle with the padding/single-GPU/max_new_tokens fixes.
  The multi-GPU pipeline-parallelism issue was confirmed gone (crash
  happened on GPU 0 alone, not mid-run) — but hit a new failure
  immediately, on the very first batch's forward pass:
  `OutOfMemoryError: CUDA out of memory. Tried to allocate 38.00 MiB.
  GPU 0 has a total capacity of 14.56 GiB of which 26.81 MiB is free...
  this process has 14.53 GiB memory in use.`
- Root cause: loading `Qwen/Qwen2.5-7B-Instruct` in bf16 consumes ~14GB
  just for weights (7B params x 2 bytes), which is nearly the *entire*
  usable capacity of a single T4 (14.56 GiB, not the nominal 16GB — some
  is reserved by the system/CUDA context). That left essentially no
  headroom for batch activations or KV cache, so it failed allocating an
  extra 38MB on the very first prefill, before generation even started.
- **Fixed by loading the teacher in 4-bit** (`BitsAndBytesConfig`,
  nf4, bf16 compute dtype) instead of full bf16 — shrinks weights to
  ~4-5GB, leaving ~9-10GB free for batch=8 generation.
- Worth noting: this makes the Kaggle `hf` backend *more* consistent with
  what was already validated locally, not less — the local Ollama smoke
  test used `qwen2.5:7b-instruct-q4_0`, which is already 4-bit. The bf16
  choice for the Kaggle path was an unexamined default, not a deliberate
  decision; 4-bit was the right choice from the start, both because it
  fits and because it matches what was already smoke-tested.
- Four real bugs found and fixed across three restarts on this single
  data-generation step (right-padding, multi-GPU pipeline-parallelism,
  no bound on batch worst-case latency, bf16-doesn't-fit-a-T4). All four
  are genuinely defensible "what went wrong" interview material — a
  concrete illustration of gap between "generation logic is correct" and
  "generation runs efficiently and correctly at batch scale on real
  hardware," which is a distinction easy to state abstractly but only
  really understood by hitting each failure mode directly.

### Addendum — throughput checkpoint caught a quota-exhaustion risk before it happened
- With all four correctness/memory fixes in place, checked the run's
  health again after ~344/7,619 rows (rather than waiting longer):
  `parse_failures_so_far=0` — the padding fix fully resolved the quality
  problem. But timing (batches 288→344, ~109s/8 rows) gave **13.7s/row**,
  extrapolating to **~30.3 more hours** for the remaining queue.
- This exposed a problem one level up from correctness: **30+ hours for
  one data-generation step is roughly this account's entire weekly T4
  quota (~30 GPU-hrs/week)**, on a step that isn't even training a model
  yet. Continuing as-is would have meant ~4 forced 9-hour session
  restarts consuming the whole week's budget before any SFT/DPO training
  run — the actual point of the project — could happen at all.
- Diagnosed the cause: batch_size was still 8, left over from when the
  model was loaded in bf16 (no memory headroom to raise it). After
  switching to 4-bit (previous fix), weights dropped to ~4-5GB of the
  14.56GB T4, leaving ~9-10GB of VRAM sitting idle at batch=8 — the KV
  cache for a ~600-token sequence under Qwen2.5-7B's GQA (4 KV heads) is
  only ~35MB, so batch=8 was using a small fraction of available memory.
  Raised `batch_size` to 24.
- **Decision, with user input: fix throughput (bigger batch) rather than
  cut dataset size.** Framed this explicitly as a choice (increase batch
  vs. also cap the queue vs. accept the multi-session cost) rather than
  picking unilaterally, since it trades the user's finite weekly Kaggle
  quota — a resource-budget call, not a code-correctness one. User chose
  batch-size increase only, keeping the full 7,619-row queue.

### Why this approach (and what alternative was rejected, and why)
- Could have just let the under-provisioned run continue across multiple
  forced session restarts. Rejected: technically works but spends the
  same weekly GPU-hour budget needed for the actual training ablation
  grid (5 runs: baseline, sft_qlora, sft_lora_fp, dpo_from_sft,
  dpo_from_base) on label generation alone — the wrong place to spend a
  scarce, shared resource.
- Could have cut the dataset size instead of raising batch size. Rejected
  as the first lever (kept as a fallback) because raising batch size is
  free — no scientific tradeoff, no thinner per-category breakdown —
  and the 4-bit switch had already created idle headroom that made this
  the obviously cheaper fix to try first.

---

## 2026-08-17 — Phase 1: Deterministic rubric + DPO corruption functions (built while the Kaggle generation run was in progress)

### What I did
- Built `src/eval/rubric.py` before any training, per the project's own
  build order ("it defines what good means") — four components scored
  independently and reported separately, not blended into one number
  (spec section 5.1):
  - `structure_ok`: reuses `parse_irac` (already enforces section order
    and non-emptiness via the anchored regex).
  - `groundedness`: the hallucination guard, spec's "most important single
    check." Two signals combined with `min()`: (a) any quoted span in
    Application must be a real substring of the source clause, falling
    back to a word-overlap heuristic when the model paraphrases instead
    of quoting (documented limitation — the current prompt template
    doesn't force literal quotation); (b) Rule must not contain invented
    legal authority (named courts/statutes/"held that"/"precedent") — the
    teacher prompt explicitly instructs against this, so a rubric that
    claims to be a hallucination guard needs to check it too.
  - `length_ok`: graded, not a hard cutoff (catches truncation/rambling).
  - `no_contradiction`: heuristic term-pair check (enforceable/
    unenforceable, valid/invalid, etc.) between Application and
    Conclusion, plus an explicit self-reversal phrase check (see below).
  - 9 unit tests, plus manual validation against the 5 real smoke-test
    outputs from 2026-08-16 — scores landed 0.86-1.00 with sensible
    groundedness variation, no false positives from the stricter checks
    added later in this session.
- Built `src/data/corrupt.py`: six corruption types from spec section 3.2
  (drop section, reorder sections, wrong citation, unsupported claim,
  contradict conclusion, truncate), each recording which type produced it
  for the later per-corruption-type breakdown. `make_rejected(chosen_text,
  rng)` requires a well-formed `chosen_text` (raises `ValueError`
  otherwise) since corruption only makes sense starting from something
  already correct.
- Wrote `tests/test_corrupt.py`, including one integration test that
  matters more than the others: every corruption type must make the
  rubric's aggregate score go *down* relative to the uncorrupted chosen
  response, since that's the entire premise of a DPO pair. Ran it — it
  failed, twice, against two different corruption types, on first run.

### Why this approach (and what alternative was rejected, and why)
- Deliberately wrote the "rejected-scores-lower-than-chosen" test to
  exercise all six corruption types against the actual rubric, rather
  than just checking "the corrupted text differs from the original."
  Corruption functions and the rubric are meant to validate each other in
  this project (corrupted samples should be reliably *worse* under the
  same rubric used to judge trained-model outputs later) — a test that
  only checks "text changed" wouldn't catch a corruption type the rubric
  is blind to, which is exactly what happened.
- Considered making `no_contradiction` and `groundedness` more
  sophisticated (e.g. an actual NLI model) instead of keyword heuristics.
  Rejected for now: the project spec explicitly scopes these as
  heuristics, and a second model in the loop for *scoring* would blur the
  dual-metric design (deterministic rubric vs. LLM judge) the project is
  built around — the rubric's value is being fast, free, and fully
  reproducible.

### Difficulty encountered (errors, dead ends, surprising results)
- **Bug 1 — `contradict_conclusion` invisible to `no_contradiction`.**
  The corruption replaces Conclusion with generic reversal language
  ("contrary to the analysis above, this clause imposes no enforceable
  obligation..."). The chosen fixture's Application never mentions
  "enforceable" (that word was in Rule), so the term-pair heuristic had
  no matching vocabulary on the Application side and scored the corrupted
  Conclusion as consistent. First real gap: a hand-picked test fixture +
  a hand-picked corruption template can fail to overlap in vocabulary
  purely by accident, and a purely lexical heuristic has no way to notice.
- **Bug 2 — `unsupported_claim` invisible to `groundedness`.** This
  corruption inserts fabricated case law/statutory citations into the
  **Rule** section. `groundedness` at that point only ever looked at
  Application (checking clause-text grounding) — Rule wasn't checked by
  *any* rubric component. Structure, length, and contradiction were all
  unaffected too, so the corrupted sample scored identically to the
  clean one. This is a more fundamental gap than bug 1: the rubric
  literally had zero coverage of one of the four IRAC sections, despite
  the teacher prompt explicitly warning against exactly this failure mode
  in that section ("do not invent case citations").
- Both bugs were caught by the same single integration test, on the
  first run, before either shipped into the actual DPO dataset generation
  — this is the test earning its keep immediately, not just theoretical
  value.

### How it was resolved
- Bug 1: added `REVERSAL_CUES`, a small list of explicit self-reversal
  phrases ("contrary to the", "notwithstanding the application",
  "despite the foregoing", etc.) checked directly against Conclusion,
  independent of the term-pair vocabulary overlap.
- Bug 2: extended `score_groundedness` to take `rule_text` as well as
  `application_text`, added `AUTHORITY_RED_FLAGS` (named courts/statutes/
  "held that"/"precedent"/etc.), and combined the two signals (clause
  grounding, authority grounding) with `min()` — either one failing sinks
  the score, appropriate for a check spec calls "the most important
  single check."
- Re-ran the full test suite (23 tests) and re-validated against the 5
  real smoke-test outputs to confirm the stricter checks introduced no
  false positives on genuine teacher output — identical scores to before
  the fix.

### What I'd do differently
- Would write the corruption↔rubric integration test *before* writing
  the six corruption functions, not after — in this case both bugs were
  still caught before anything shipped, but designing the validation
  check first would have made the coverage gaps (Rule section entirely
  unchecked; Conclusion-reversal vocabulary not overlapping) visible
  while writing `rubric.py`'s components, rather than as a separate
  discovery step afterward.

---

## 2026-08-17 — Phase 1: Contract-level splits

### What I did
- Built `src/data/split.py`: `assign_splits()` maps each unique contract
  to train/valid/test once (sorted-then-shuffled for order-independent
  determinism, seed-controlled), `split_records()` applies that mapping to
  any list of records with a `contract` field — used identically for SFT
  rows and, later, DPO pairs, so a contract lands in the same split in
  both datasets.
- 7 unit tests, including the one that actually matters:
  `test_no_contract_leaks_across_splits` — builds 200 synthetic contracts
  with 5 clauses each, splits them, and asserts no single contract's
  clauses appear in more than one split.
- Ran it against the real 7,620-row queue (384 contracts): **307/38/39**
  contracts (train/valid/test), **6,039/897/684** rows — close to the
  configured 80/10/10 at the contract level; row-level ratios drift
  slightly (≈79%/12%/9%) since contracts don't all contain the same
  number of eligible clauses, which is expected and fine given the split
  unit is deliberately the contract, not the row.

### Why this approach (and what alternative was rejected, and why)
- Sort-then-shuffle rather than shuffling a set/dict directly — Python
  set iteration order isn't guaranteed stable across processes even with
  the same seed for other RNG state, so sorting first removes that as a
  hidden source of non-determinism before the seeded shuffle runs.
- `split_records()` takes a generic `contract_key` and operates on plain
  dicts rather than being hardcoded to the SFT row schema, specifically
  so the same function applies unchanged to DPO pairs once they exist —
  one splitting implementation, not one per dataset shape.

### Difficulty encountered
- None of note — this one worked as designed on the first pass, unlike
  the rubric/corruption pair. Worth noting precisely because it's the
  exception: most of this session's components have needed at least one
  real fix after the first test run.

---

## 2026-08-17 — Phase 1: Kaggle run completed; rubric validated (and fixed twice more) against the real dataset

### What I did
- The Ollama-on-Kaggle run finished: downloaded `sft_teacher_targets.jsonl`
  to `data/raw/`. **7,620/7,620 rows, 100% `parse_ok`, 0 duplicate ids, all
  36 categories and all 384 contracts represented, target word counts
  66-280 (well inside the rubric's 40-400 sane range).** The Ollama pivot
  fully resolved the generation reliability problems from the earlier
  `transformers` path — a dramatic improvement over the ~46% parse-failure
  rate seen there.
- Ran the full rubric (`src/eval/rubric.py`) over all 7,620 real targets —
  the first real stress test of the rubric against production-scale data,
  not just hand-written fixtures or the 5-row smoke test. First pass:
  aggregate 0.986, only 4 rows (0.1%) below 0.7. Investigated all 4 by
  hand rather than accepting the number at face value.
- **Bug 3 (of the day): `("applies", "does not apply")` and analogous
  scope-conditional term pairs are structurally unsound for a term-pair
  contradiction check.** One of the 4 low scorers: Application said "the
  clause applies to X ... does not extend to solicitation outside these
  regions," Conclusion said "does not apply to solicitation activities
  outside these regions" — the *same claim about limited scope*, stated
  twice, flagged as a contradiction because Application contained
  "applies" (positive) and Conclusion contained "does not apply"
  (negative). Two more of the 4 were a different pattern: Application
  hedges a risk ("may render the clause... potentially unenforceable"),
  Conclusion recommends a fix ("should be clarified... to ensure it is
  enforceable") — legitimate defect-then-remedy reasoning, not a
  contradiction, but a bare keyword match can't tell assertion from
  hedged-conditional language.
- **Fix:** dropped the scope-conditional pairs entirely from
  `CONTRADICTION_PAIRS`; added sentence-level hedge-word guarding
  (`_asserted()`) so a term inside a hedged sentence — "may", "could",
  "to ensure", "unless", etc. — doesn't count as an assertion.
- Re-running the fix's own test suite caught a bug in the fix itself: the
  hedge-word `_asserted()` rewrite used plain substring matching instead
  of the original mutual-exclusion logic, so `"enforceable"` (a literal
  substring of `"unenforceable"`) started matching *both* polarities of
  the same pair simultaneously, silently defeating the check —
  `test_contradiction_detected` (an existing, already-passing test)
  failed immediately. Fixed with `\b`-word-boundary regex matching
  instead of `in`.
- Re-scored the full dataset again: **0 rows below 0.7** (down from 4).
  Aggregate 0.987, `no_contradiction` 0.9993.
- Checked the remaining sub-0.9 rows (all groundedness-driven) and found
  **bug 4**: `AUTHORITY_RED_FLAGS` included bare `"precedent"`, which
  matched inside `"condition precedent"` — a completely standard,
  legitimate contract-law term of art (a condition that must occur before
  a duty arises), unrelated to case-law precedent. Fixed by replacing the
  bare term with qualified phrases that actually indicate case-law usage
  (`"case precedent"`, `"legal precedent"`, `"binding precedent"`, `"sets
  a precedent"`, etc.) — `"condition precedent"` no longer matches any of
  them, while a genuine invented case-law reference still does (covered
  by a new regression test).
- Final state: 35 unit tests passing, full dataset aggregate **0.9874**,
  `structure_ok`/`length_ok` perfect (1.0000), `groundedness` 0.9504,
  `no_contradiction` 0.9993. Remaining sub-0.9 rows (343, 4.5%) are
  legitimately looser paraphrases on short clauses, not further rubric
  bugs — good candidates to prioritize in the manual quality review
  sample, not something to keep chasing with more heuristic patches.

### Why this approach (and what alternative was rejected, and why)
- Investigated every low-scoring row by hand rather than trusting the
  aggregate number — the whole reason to build a rubric before training
  is to know precisely what it's measuring, and an unexamined 0.986
  could just as easily have meant "4 genuine teacher failures" as "4
  rubric false positives." They were the latter, twice over, plus a third
  false-positive pattern (`condition precedent`) found by continuing to
  check even after the contradiction fixes looked complete.
- Stopped iterating on the rubric once low scorers stopped revealing new
  *systematic* patterns and started being individually-explainable loose
  paraphrases — chasing 4.5% of rows with more heuristics risks
  overfitting the rubric to this specific dataset's phrasing quirks
  rather than keeping it a general, defensible measure.

### Difficulty encountered (errors, dead ends, surprising results)
- Four real bugs found in one scoring pass against real data, on top of
  the two found earlier by the corruption↔rubric integration test — six
  total rubric bugs caught before any of this touched a training run.
  Every one was a heuristic being *too broad* (matching a technically-true
  substring/keyword in a context where it didn't mean what the heuristic
  assumed), never too narrow. Worth remembering as a pattern: cheap
  lexical heuristics fail almost exclusively in the direction of false
  positives on legitimate, nuanced language — hedges, scope limits, and
  terms of art — not by missing real problems.
- My own debugging process had a bug too: an early `next(x for x in rows
  if 'HEALTHGATEDATACORP' in x['id'])` substring match silently grabbed
  the wrong row (there were multiple rows sharing that contract-name
  prefix across different categories/clause indices) — wasted one
  investigation cycle chasing a discrepancy that was just my own lookup
  being ambiguous, not a rubric bug. Switched to exact `id` matching.

### How it was resolved
- All four fixes applied directly in `src/eval/rubric.py`; regression
  tests added to `tests/test_rubric.py` for each (scope-limited
  application, hedged risk-then-remedy, the `enforceable`/`unenforceable`
  substring-overlap bug, and `condition precedent`). Full suite (35
  tests) passing; full real-dataset re-score confirms no regressions.

### What I'd do differently
- Would run the rubric against a larger real sample (even just 50-100
  rows from the smoke test) *before* declaring it "done" the first time,
  rather than only against 5 hand-picked smoke-test rows and hand-written
  fixtures — every one of today's four bugs only surfaced once real,
  varied, model-generated language was thrown at the heuristics.

---

## 2026-08-17 — Phase 1: Manual review sample size — scope adjustment from spec

### What I did
- Spec section 3.1 says review "~20-25%" of the SFT data. At the dataset's
  actual size (7,620 rows), that's 1,500-1,900 rows — not reviewable by
  hand on a solo project timeline. Raised this explicitly rather than
  silently picking a number; agreed on a fixed 60-row stratified sample
  instead (25 lowest rubric-scoring rows + 35 random, spread across
  categories) — comparable in spirit to the judge-validation sample size
  the spec itself uses elsewhere (30-40 examples, section 5.2).
- Built `src/eval/sample_for_review.py`: scores every row with the rubric,
  takes the 25 lowest-aggregate rows plus 35 seeded-random rows from the
  remainder, shuffles them together (so low-scorers aren't all presented
  first, which would bias a reviewer's sense of overall quality), and
  writes `data/review/sft_review_sample.csv` with blank
  `reviewer_verdict`/`reviewer_notes` columns to fill in by hand.
- Kept the completed CSV in git (unlike `data/raw/`, which is gitignored)
  — the filled-in review is evidence the QA step actually happened, not
  a regenerable cache.

### Why this approach (and what alternative was rejected, and why)
- Considered a pure percentage-based sample scaled down (e.g. literally
  25% of a *capped* subset). Rejected: the point of reviewing low-scoring
  rows specifically is that they're the most likely place to find a real
  problem (as demonstrated by every rubric bug found today) — a pure
  random sample of any size is less efficient at catching rare failure
  modes than a sample deliberately weighted toward the hardest cases,
  plus a random component to catch failures the rubric itself might be
  blind to.

---

## 2026-08-17 — Phase 1: Human review of the 60-row sample — the most valuable single review pass in the project so far

### What I did
- The manual review came back: 32 good, 17 minor_issue, 11 bad (of 60).
  This is a genuinely different kind of signal than anything automated so
  far — a domain-competent human reading the actual legal reasoning, not
  a lexical heuristic checking for keyword patterns. Findings, in order of
  how much they changed the pipeline:
  1. **Non-substantive clause fragments.** CUAD occasionally extracts
     unusable spans as "the clause": a single word ("This"), a 3-word
     fragment ("transferable or assignable."), a mid-sentence truncation
     ("If the Minimum Efficiency Level has"). The teacher dutifully wrote
     a full four-section IRAC analysis for each anyway — exactly the
     behavior not to train in. This is a root-cause finding: the fix
     belongs in clause *extraction*, not the generation prompt.
  2. **Two confirmed rubric blind spots** with real examples: a
     self-contradiction with zero shared vocabulary between the
     conflicting claims (`"each party's individual liability is not
     capped separately"` vs. `"no single party... can be held responsible
     for more than the stated maximum"` — logically incompatible, lexically
     unrelated), and a Conclusion that asserts something then hedges the
     *same point* in the same sentence (`"...combined, but it is unclear
     if this applies... individually or collectively"`).
  3. **A fabricated general legal principle**, not a fabricated citation —
     a Rule section invented "the grantor is typically required to
     provide the grantee with the option... within a reasonable time
     frame, as per standard contract law principles." Nothing in
     `AUTHORITY_RED_FLAGS` fires (no named court/statute), because this
     isn't a fake citation, it's a fake *rule*. No cheap fix exists for
     this without an actual legal-knowledge check.
  4. **Exact duplicate clause_text pairs**, including one case where the
     *same clause, same contract, same category* got sampled twice by the
     teacher (temperature 0.3) and produced two different readings — one
     correct, one with the parties swapped. `rubric_groundedness` scored
     both 0.0 identically, unable to distinguish "correct but low word-
     overlap" from "party roles inverted."
  5. **Unsupported gloss**: ~4-5/60 rows added a condition not present in
     the clause, most often inventing "without [Party]'s consent" where
     no consent mechanism exists in the source text.
- Built `is_substantive_clause()` in `build_irac.py` (word-count floor,
  min 8 words) and wired it into `load_cuad_filtered()` for any future
  regeneration, plus a post-hoc `src/data/clean_sft.py` to apply the same
  filter to the already-generated dataset.
- Added a targeted rubric fix for finding 2's *second* pattern only
  (self-hedging Conclusion: `_self_hedging_conclusion()`, requires an
  uncertainty marker *and* a contrastive conjunction in the same sentence,
  narrow enough not to penalize an honestly-uncertain Conclusion that
  never made a contradicted claim in the first place). Did **not** attempt
  a fix for the first pattern (zero shared vocabulary) or finding 3
  (fabricated general principle) — both require actual semantic
  understanding, which is precisely why the dual-metric design
  (deterministic rubric + LLM judge) exists. Documented as rubric
  limitations rather than force-fit into more keyword heuristics.
- Added a "do not introduce conditions/mechanisms absent from the clause"
  line to `TEACHER_GENERATION_TEMPLATE`, for any future regeneration —
  doesn't retroactively fix the existing dataset.

### Difficulty encountered — my own dedup fix was wrong, caught before shipping
- First attempt at "fix the duplicates" deduplicated the whole dataset by
  bare `clause_text`, discovering (what looked like) 910 duplicate rows to
  remove. Before trusting that number, checked whether the duplicate
  groups shared a category: **only 31/910 did.** The other 893 were the
  same `clause_text` under a *different* category label — e.g. the
  max/min capacity clause under both "Minimum Commitment" and "Volume
  Restriction" that prompted this investigation. Category is part of
  `CANONICAL_PROMPT_TEMPLATE`, so those are genuinely different training
  prompts with the same clause embedded, not duplicates — a naive
  clause-text-only dedup would have silently discarded 893 legitimate
  training examples (11.7% of the dataset) to fix a problem that only
  actually affected 31 rows.
- Investigating *why* the same clause_text sometimes repeats across
  *different contract titles* surfaced the real, more important finding:
  two pairs of contract-title strings in the dataset are **the same
  underlying document filed twice under different filenames** (confirmed:
  `ARMSTRONGFLOORING,INC_01_07_2019-EX-10.2-...` and
  `ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_...` share 16
  identical clauses, obviously the same exhibit with a different filename
  convention; similarly an Adurobiotech pair with a literal `(1)` suffix).
  This threatens contract-level splitting specifically: if a duplicate
  filing lands in `train` and its twin lands in `test`, that's leakage
  under a different name — exactly what contract-level splitting exists
  to prevent, undermined by a data-source quirk the split logic didn't
  know to look for.
- A naive "shared clause count" threshold to detect this isn't reliable
  on its own: a third candidate pair (Intelligent Highway Solutions,
  Sibannac — two genuinely different, unrelated companies) shared 5
  identical clauses purely from reusing the same boilerplate agreement
  template. Shared-clause-count alone would have wrongly merged two real,
  distinct contracts.

### How it was resolved
- Reverted the clause-text-only dedup entirely from `clean_sft.py` —
  confirmed via `find_duplicate_documents`-style analysis that
  `(contract, category, clause_text)` triples have **zero** collisions in
  the real dataset, so no row-level dedup is actually needed at that
  granularity.
- Built `find_duplicate_documents()` in `split.py`: merges two contract
  titles only when they share ≥3 identical clauses *and* their normalized
  company-name prefixes match (strip punctuation, uppercase, compare) —
  both signals required. Wired into `split_records()` via a
  `dedupe_documents` flag (default on) so duplicate filings are
  canonicalized to one title before split assignment, guaranteeing they
  land in the same split. 4 tests added, including one asserting the
  Intelligent Highway/Sibannac case is *not* merged and one asserting no
  split ever contains a duplicate document's clauses split across
  partitions.
- Final `clean_sft.py` run: 7,620 → 7,516 rows (11 confirmed-bad + 93
  fragment rows removed; no clause-text dedup). Final splits: 5,987/726/803
  rows across 307/38/39 contracts (2 duplicate-document pairs correctly
  merged into single contracts for assignment purposes).

### What I'd do differently
- Would compute the same-category-vs-different-category breakdown
  *before* reporting the first "910 duplicates" number to the user, not
  after — the instinct to verify was right, but it should be the default
  step before presenting any aggregate count as a finding, not a
  follow-up triggered by the number feeling too large.

---

## 2026-08-17 — Phase 1: DPO pairs built, splits finalized, DVC vs. git-lfs reconciled

### What I did
- Built `src/data/build_dpo.py`: one DPO pair per cleaned SFT row (7,516
  pairs), using `corrupt.py`'s six corruption types. Corruption RNG seeded
  from a `hashlib.sha256` digest of each row's `id`, not Python's builtin
  `hash()` — builtin `hash()` on strings is randomized per-process
  (`PYTHONHASHSEED`) unless explicitly disabled, so seeding with it would
  have made corruption type *silently* non-reproducible across runs
  (it would always produce *a* valid result, just a different one each
  time) — caught before running, not after. Regression test added.
  Resulting corruption-type distribution is roughly even: 1,195-1,296 per
  type across the 6 types.
- Applied the same contract-level split (with document-dedup) to the DPO
  pairs using the same seed/ratios as the SFT split. Verified directly
  (not assumed) that both datasets agree exactly on which row ids land in
  train/valid/test: 5,987/726/803, identical sets in both.
- Ran `dvc init` + `dvc add data/splits data/datasets` per the original
  spec ("version them (DVC, consistent with the MLOps project)").
- **Found a real tooling conflict before it caused problems**: `git
  status` showed this repo already has a GitHub remote and 3 commits I
  didn't make — the user has been committing/pushing this project's code
  in parallel, outside these tool calls. One of those commits
  ("git lfs tracked") had already configured git-lfs to track `*.jsonl`
  directly via `.gitattributes`. Running both DVC and git-lfs on the same
  file type is redundant (DVC's auto-generated `data/.gitignore` excludes
  `data/splits/` and `data/datasets/` from git entirely, so git-lfs's
  filter would never even see those files) and confusing for anyone
  reading the repo later. Checked `git ls-files` / `git lfs ls-files`
  first — confirmed no `.jsonl` content had actually been committed under
  either system yet, so no data was at risk either way.
- User chose DVC (matches the original spec). Removed `.gitattributes`
  entirely rather than leave a dead LFS rule in the repo.

### Why this approach (and what alternative was rejected, and why)
- Investigated the unexpected "up to date with origin/master" message
  immediately rather than proceeding — an unfamiliar remote/commit
  history in a repo I'm actively modifying is exactly the kind of thing
  to understand before making more changes, not after.
- Did not commit any of this session's changes — the user has their own
  cadence for committing (3 commits so far, none triggered by me), and
  the instruction is to only commit when explicitly asked. Left
  everything staged/modified for the user to commit on their own terms.

### Difficulty encountered
- None beyond the two things caught before they became real problems
  (the `hash()` non-determinism, the DVC/git-lfs redundancy) — both
  caught by checking rather than assuming, consistent with the rest of
  this session.

---

## 2026-08-17 — Phase 2: local SFT smoke test, and a real completion-masking bug it caught

### What I did
- Wrote `src/train/train_sft.py`: parameterized `trl.SFTTrainer` +
  `peft` LoRA/QLoRA script (model, paths, batch size, LoRA rank, 4-bit
  on/off all CLI args), deliberately the *same* script for the local
  smoke test and the real Kaggle 3B ablation runs — no separate
  "local version" to drift out of sync with the real one.
- Built a 60/20-row seeded subset of the real train/valid splits
  (`data/raw/smoke_{train,valid}.jsonl`) and ran a full smoke test with
  `Qwen/Qwen2.5-0.5B-Instruct`, 4-bit QLoRA, on the local GTX 1650.
- First run completed "successfully" (loss 1.66→1.43, saved an adapter)
  but printed the same warning ~46 times, once per example, in both the
  train and eval tokenization passes:
  `[RANK 0] Mismatch between tokenized prompt and the start of tokenized
  prompt+completion...`. Per this project's own established pattern
  (verify the "910 duplicates" number, verify the bitsandbytes install,
  verify the DVC/git-lfs state before trusting it), did not accept a
  repeated warning at face value just because training "worked."

### Why this approach (and what alternative was rejected, and why)
- Investigated by hand rather than trusting the warning's own vague
  explanation ("may be due to unexpected tokenizer behavior..."):
  tokenized one real `prompt` alone and `prompt + completion` together
  with the same tokenizer and diffed the token ids at the boundary.
  Found the actual mechanism: the prompt ended with `"...section."` (no
  trailing whitespace) and every completion starts with `"Issue:"` — no
  separator between them. Tokenized jointly, BPE merges `"."` + `"I"`
  into a single token (`".I"`), a token that doesn't exist when the
  prompt is tokenized alone.
- Read `trl`'s own `sft_trainer.py` source (installed package, not just
  docs) to see exactly how it uses this comparison. It computes
  `completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids)
  - len(prompt_ids))` — i.e. it tokenizes the prompt *alone* to get a
  length, then slices the *jointly*-tokenized sequence at that length to
  decide what's prompt (loss-masked) vs. completion (trained on). Because
  of the merge, the token at exactly that slice boundary in the joint
  sequence is the merged `".I"` token, which lands on the masked side.
  Net effect: **the model was never being trained to predict the first
  token of any completion** — and since every single target starts with
  `Issue:`, this wasn't a rare edge case, it was systematic across the
  entire dataset. Confirmed the identical mismatch-detection code exists
  in `dpo_trainer.py` and `kto_trainer.py` too, so this would have
  silently recurred in the DPO run if only patched locally in the SFT
  script.
- Fixed at the root instead of papering over it in the training script:
  added a trailing `"\n\n"` to `CANONICAL_PROMPT_TEMPLATE` in
  `build_irac.py` (the single shared template every stage imports, per
  the project's own "don't maintain two versions of a template" rule).
  Verified the fix directly — re-tokenized prompt+completion and
  confirmed `prompt_ids == full_ids[:len(prompt_ids)]` now holds exactly.
  Considered patching only `train_sft.py`'s data loader instead (e.g.
  prepending `"\n\n"` to the completion at load time without touching the
  stored `prompt` field), but rejected it: the same `prompt` field is
  reused for DPO training, LLM-judge context, and eventual FastAPI
  serving, so fixing it in one consumer and not at the source would leave
  the bug live everywhere else that reads `prompt` from the JSONL files.
- Since `prompt` is 100% template-derived from `category` + `clause_text`
  (not manually authored or reviewed content), regenerated it in place
  across every already-built file — `data/splits/{sft,dpo}_{train,valid,
  test}.jsonl`, `data/datasets/{sft_clean,dpo_pairs}.jsonl`,
  `data/raw/sft_teacher_targets.jsonl`, and the smoke subset — rather
  than only fixing it going forward. This did *not* require touching
  `data/review/sft_review_sample.csv` (no `prompt` column there) and
  does not invalidate the manual review, since reviewers were assessing
  `target` content and groundedness, not prompt whitespace.
- While re-running the full test suite to confirm the fix didn't break
  anything, `test_rejected_scores_lower_than_chosen_on_rubric` failed
  non-deterministically across reruns with zero code changes between
  runs. Root cause: that test itself seeds with `random.Random(hash(
  corruption_type) % 1000)` — the exact same builtin-`hash()`-is-
  process-randomized footgun already caught and fixed in
  `build_dpo.py`'s `_seed_from_id` earlier this session, just not
  noticed in the test that was added alongside it. Fixed by seeding with
  `random.Random(corruption_type)` directly — `random.Random`'s own
  seeding of `str`/`bytes` uses SHA-512 internally and is stable
  regardless of `PYTHONHASHSEED`, so no separate hashlib call is needed
  here. Verified fixed by rerunning 5x.

### Difficulty encountered
- The warning's own message ("may be due to unexpected tokenizer
  behavior, whitespace issues...") technically named the right category
  of cause but was too vague to act on directly — required reading the
  actual `trl` source to understand it was reporting a *known, mechanical*
  masking bug, not a generic heads-up.
- Easy to have rationalized this away: the smoke test's loss did
  decrease (1.66→1.43) and its metrics looked directionally sane even
  with the bug present, since dropping one token's loss out of ~1,300
  tokens per example is a small perturbation to the aggregate numbers —
  it would not have been obvious from metrics alone, only from reading
  the warning and refusing to dismiss it.

### Result after the fix
- Reran the identical smoke test command: **zero mismatch warnings**,
  full 15-step run, loss 1.642→1.414, `eval_loss: 1.389`,
  `eval_mean_token_accuracy: 0.6413`.
- Loaded the saved adapter back with `PeftModel.from_pretrained` and
  generated on a held-out clause — output was well-formed IRAC starting
  cleanly at `Issue:` (content quality is naturally rough: 0.5B model,
  60 examples, 1 epoch — that's expected and not what this test was
  checking). This is the actual "does the pipeline work end-to-end"
  validation the smoke test exists for, not just a non-crashing run.
- Full test suite: 45/45 passing, deterministically, after the seeding
  fix.

### What I'd do differently
- Would inspect the actual token ids at the prompt/completion boundary
  as part of building the SFT script the first time, before ever running
  training — this class of bug (silent off-by-one in loss masking due to
  BPE merges at a formatting boundary) is common enough with raw
  prompt/completion concatenation (as opposed to chat-template-based
  formatting, which handles turn boundaries more carefully) that it's
  worth a proactive check, not something to discover via a repeated
  warning after the fact.

---

## 2026-08-17 — Phase 2: eval harness (`generate.py`/`score.py`) + Kaggle baseline/sft_qlora notebook

### What I did
- Wrote `src/eval/generate.py`: batched generation against a fixed test-set
  prompt file, either zero-shot (no `--adapter`) or with a trained PEFT
  adapter attached — one script for every eval run (baseline, `sft_qlora`,
  `dpo_from_sft`, etc.), not one per run type. Resumable the same way
  `generate_teacher_targets.py` is (skips ids already in the output file).
  Applies the batching fixes learned the hard way on the teacher-generation
  run even though this uses plain `transformers` rather than Ollama: left
  padding (right padding corrupts the KV cache for anything queued behind
  the batch's shortest sequence), `device_map={"":0}` instead of `"auto"`
  (avoids the pipeline-parallel multi-GPU slowdown measured at ~15.6s/row
  on the teacher run), bounded `max_new_tokens`.
- Wrote `src/eval/score.py`: applies `rubric.py`'s `score_response()` to a
  `generate.py` output file, writes per-row scores, and appends one
  aggregate row (`structure_ok`/`groundedness`/`length_ok`/
  `no_contradiction`/`aggregate`/`parse_ok_rate`) to `results/summary.csv`
  keyed by `run_id`. Deliberately separate from generation — scoring is
  cheap/CPU-only and needs to be re-runnable without regenerating, since
  this rubric has already needed six bug fixes after full-dataset
  validation and manual review.
- Validated both scripts locally end-to-end against the smoke adapter
  (0.5B model, 8 rows from `smoke_valid.jsonl`) before writing a single
  line of Kaggle notebook code: ran `generate.py` (confirmed resumability
  message, correct batch timing), then `score.py` (aggregate 0.81,
  `parse_ok_rate` 0.875 — plausible for a 0.5B/60-example/1-epoch model),
  and inspected one raw output directly — confirms the prompt-template fix
  from the smoke test still holds downstream (`raw_response` starts
  cleanly with `"Issue:"`).
- Added `tests/test_score.py` (2 tests) covering `score_file`'s aggregation
  and `append_summary_row`'s header-once behavior with a tiny synthetic
  fixture — `generate.py` itself isn't unit-tested (needs a real model/
  GPU), but everything downstream of it that's pure logic is.
- Built `notebooks/kaggle_train_eval.ipynb`: chains zero-shot baseline eval
  → `sft_qlora` training (`train_sft.py`, unchanged from the smoke test) →
  `sft_qlora` eval, all against the real `Qwen/Qwen2.5-3B-Instruct` model
  and the full 803-row test split / 5,987-row train split. Training
  hyperparameters are read directly out of `config.yaml` at notebook
  runtime (`peft`, `training.sft`, `training.max_seq_length` sections)
  rather than re-typed into the notebook, so the notebook can't silently
  drift from what `results/summary.csv` will claim the run used. A tiny
  dry run (8-row generation, 5-step training) runs automatically before
  each full step, matching this project's established pattern (the
  teacher-gen notebook's warmup-generate + `ollama ps` GPU check) of
  proving the path/config cheaply before committing GPU hours.

### Why this approach (and what alternative was rejected, and why)
- Considered converting the fine-tuned adapter to GGUF and serving eval
  generation through Ollama, mirroring the teacher-generation pipeline
  exactly. Rejected: the multi-bug Ollama pivot for teacher generation was
  specifically forced by a 7B model split `device_map="auto"` across a
  Kaggle T4x2 session — a 3B model in fp16 fits comfortably on one T4
  (~6GB of 16GB) without that pressure, so `device_map={"":0}` alone
  removes the actual root cause without needing a GGUF conversion step
  that adds its own new failure surface (quantization-format compatibility
  with a freshly-trained LoRA merge) for no throughput benefit at this
  model size.
- Read hyperparameters from `config.yaml` inside the notebook instead of
  hardcoding them as notebook constants, even though it's more code —
  the project's own stated rule (`irac_template` section comment: "Never
  redefine inline ... template drift invalidates the cross-run
  comparison") applies just as much to training hyperparameters as to the
  prompt template. A notebook with its own hardcoded `learning_rate = 2e-4`
  is exactly the kind of thing that quietly diverges from `config.yaml`
  after the next hyperparameter tweak.
- Validated `generate.py`/`score.py` against the local smoke adapter before
  writing any Kaggle-specific code, same reasoning as the smoke test
  itself: a bug in shared logic is far cheaper to catch on a 0.5B model
  locally in seconds than after burning Kaggle GPU-hours on the 3B run.

### Difficulty encountered
- None new — the harness worked on the first real run once the earlier
  prompt-template fix was in place; the only surprise was a `/tmp` path
  resolution difference between Git Bash (which auto-translates `/tmp/...`
  *arguments* to the real Windows path) and the same string typed inside a
  quoted `python -c` string (which does not get translated, so Python
  resolves it as `C:\tmp\...` instead) — a shell-tooling quirk of this
  Windows dev environment, not a bug in either script. Confirmed by `ls`
  on the actual Git Bash `/tmp` path.

### What I'd do differently
- Nothing new to flag yet — the honest test is the first real Kaggle run,
  since `batch_size=16` for the full eval passes is untested at 3B-model
  scale (the notebook's dry run only proves `batch_size=8` fits); noted
  directly in the notebook as a fallback (drop to 8-12 and resume) rather
  than assumed safe.

---

## 2026-08-18 — First real Kaggle run: baseline eval succeeded, `sft_qlora` training OOM'd

### What I did
- Ran `notebooks/kaggle_train_eval.ipynb` for real on Kaggle (T4, after the
  earlier P100/sm_60-incompatibility and repo-nesting fixes). Zero-shot
  baseline eval completed cleanly on the full 803-row test split:
  `structure_ok` 0.994, **`groundedness` 0.731**, `length_ok` 0.994,
  `no_contradiction` 0.991, `aggregate` 0.927, `parse_ok_rate` 0.994.
  Genuinely useful baseline number, not just a smoke-test placeholder:
  `Qwen2.5-3B-Instruct` zero-shot already follows the IRAC format almost
  perfectly, so `groundedness` — the hallucination-guard check, the one
  the spec calls out as the most important single rubric component — is
  the actual headroom SFT/DPO need to close.
- The SFT training *dry run* (5 steps, same `per_device_batch_size=4`/
  `max_seq_length=2048` as the full run, per `config.yaml`) then hit
  `torch.OutOfMemoryError` on step 1's backward pass: "GPU 0 has a total
  capacity of 14.56 GiB of which 100.81 MiB is free." This is exactly what
  the dry run exists to catch — it failed after ~5 steps and a couple
  minutes, not partway through a 3-epoch/5,987-row run burning real GPU
  quota.

### Why this approach (and what alternative was rejected, and why)
- Root-caused before patching: `train_sft.py`'s `SFTConfig` never set
  `gradient_checkpointing`. Fine for the local smoke test (0.5B params is
  tiny enough that activation memory was never the bottleneck), but at 3B
  params with `max_seq_length=2048` and `per_device_batch_size=4`, storing
  every layer's activations for the whole forward pass (needed for
  backward) is exactly the kind of memory pressure gradient checkpointing
  exists to remove — recompute activations during backward instead of
  keeping them all, trading compute for memory. This is why the local
  smoke test's clean run couldn't have caught this: the bug only exists at
  a model scale the smoke test deliberately avoids (its whole purpose is
  pipeline correctness, not memory budgeting at full scale).
- Added `gradient_checkpointing=True` +
  `gradient_checkpointing_kwargs={"use_reentrant": False}` to
  `train_sft.py`'s `SFTConfig` (non-reentrant is the currently recommended
  mode — avoids known issues with the older reentrant autograd
  implementation under `peft`).
- Also reduced `config.yaml`'s `training.sft.per_device_batch_size` 4→2
  and doubled `gradient_accumulation_steps` 4→8 (same effective batch size
  of 16, same optimizer semantics) as an extra safety margin on top of the
  checkpointing fix, rather than shipping the minimal fix and hoping it's
  enough. Reasoning: the OOM happened with essentially zero headroom (100
  MiB free, 8.59 GiB "in use" by the process), and confirming whether
  checkpointing alone closes that gap costs another Kaggle session and
  more weekly GPU-hour quota either way — worth spending a small amount of
  training-loop overhead now (more, smaller micro-batches per optimizer
  step) to raise the odds of the very next attempt succeeding, rather than
  re-probing incrementally on a scarce resource.
- Verified the fix doesn't break anything before sending the user back to
  Kaggle: reran the local smoke test end-to-end with gradient checkpointing
  enabled. Identical loss/accuracy trend as the pre-fix run (loss
  1.64→1.41, `eval_mean_token_accuracy` 0.6405 vs. the earlier run's
  0.6413 — noise-level difference, not a regression) and the full 47-test
  suite still passes. Gradient checkpointing has known compatibility edge
  cases with certain `peft`/reentrant-autograd combinations, so this was
  worth checking on a 0.5B model in under a minute rather than discovering
  an incompatibility on the next multi-hour Kaggle attempt.
- Didn't need to touch the notebook itself: both the dry-run and full-run
  training cells build their CLI args from `config.yaml` at notebook
  runtime (`train_sft_args()`), so the config.yaml fix alone propagates to
  both once the updated `src/`+`config.yaml` zip is re-uploaded — exactly
  the payoff of not hardcoding hyperparameters into the notebook in the
  first place.

### Difficulty encountered
- None beyond the OOM itself — caught cheaply by the dry run exactly as
  designed, not partway through the expensive full run.

### What I'd do differently
- Would have added `gradient_checkpointing=True` to `train_sft.py`
  proactively when first writing it, rather than waiting to discover its
  absence via a real OOM — it's close to a default-on setting for QLoRA
  fine-tuning at this model scale, and the smoke test's own success
  masked its absence rather than validating it, precisely because a 0.5B
  model doesn't need it. Worth explicitly asking "does this default
  change behavior at the real target model size, not just the smoke-test
  size" for any training-loop setting, not just relying on the smoke test
  passing.

---

## 2026-08-18 — Second Kaggle attempt: 12-hour session timeout, root-caused to un-pinned multi-GPU training

### What I did
- With the OOM fix applied, the user reran `notebooks/kaggle_train_eval.ipynb`
  (baseline cells commented out per the previous entry, training only) on
  Kaggle's **"GPU T4 x2"** accelerator (two GPUs, visible in the run
  screenshot). The session ran for the full 12-hour Kaggle limit (43200s)
  and was killed by the platform (exit code 137) without finishing.
  Downloaded partial output showed training had only reached
  `checkpoint-188` out of an expected ~1,125 total steps (5,987 rows,
  effective batch size 16, 3 epochs) — roughly **230 seconds/step**, wildly
  slow for a 3B QLoRA step that should take a few seconds.
- Root-caused before patching, not just re-tried with more time: noticed
  `train_sft.py` passed a bare model-id *string* to `SFTTrainer` and never
  set a `device_map`, unlike `generate.py`, which explicitly pins
  `device_map={"":0}` specifically because of an *already-documented*
  incident (2026-08-16/17 LOG.md entries): letting `transformers`/
  `accelerate` auto-place a model across >1 visible GPU can split it
  pipeline-parallel, serializing forward/backward across GPUs connected by
  comparatively slow interconnect — measured ~15.6s/row during teacher
  generation on a T4x2 session versus one pinned GPU. With "T4 x2" selected
  this run and nothing pinning the device during *training*, this is the
  same bug recurring in a part of the pipeline that hadn't been touched by
  the earlier fix.
- Fixed `train_sft.py` to explicitly load the model via
  `AutoModelForCausalLM.from_pretrained(..., device_map={"":0}, ...)`
  itself and pass the loaded model object to `SFTTrainer`, instead of
  letting `SFTTrainer` load a bare model-id string internally with no
  placement control.
- Also tried, then reverted, a second fix in the same pass: made `bf16`
  vs. `fp16` conditional on GPU compute capability (reasoning: Turing/T4
  lacks Tensor Core bf16 acceleration, only Ampere+ does). This crashed
  the very next local smoke-test verification with
  `"_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for
  'BFloat16'` — some component ended up bf16 regardless of the requested
  dtype, which breaks fp16's `GradScaler`-based mixed precision (fp16
  needs loss scaling since it has much less dynamic range than bf16; bf16
  doesn't need a scaler at all, which is part of why it's the safer
  default). Also re-examined the evidence for the hypothesis this was
  meant to fix: a local matmul micro-benchmark (`torch.randn(4096,4096)
  @ ...`) actually showed bf16 running *faster* than fp16 on this Turing
  GPU (82.5ms vs. 402ms), the opposite of what the hypothesis predicted —
  though this specific GPU (GTX 1650) turned out not to be a valid proxy
  for the T4 anyway, since GTX 16-series chips have no Tensor Cores at all
  (unlike the datacenter T4), so the benchmark doesn't actually test what
  it was meant to test either way. Reverted to unconditional `bf16=True`,
  which had already been validated clean across every prior local smoke
  test — not worth carrying a change that (a) just crashed and (b) whose
  own justification didn't survive a first empirical check.

### Why this approach (and what alternative was rejected, and why)
- Considered just re-running with more patience/a longer session, since
  Kaggle sessions can sometimes just be slow. Rejected: 230s/step is not
  "slow," it's two orders of magnitude off from what a 3B QLoRA step
  should take, and this project has *already* diagnosed and fixed the
  exact same symptom once this session (teacher generation) — re-running
  without applying that same fix to the newly-affected code path would be
  ignoring a lesson already paid for in GPU-hours once.
- Chose to verify the OOM-fix-plus-device-pin combination locally with the
  smoke test again before sending the user back to Kaggle a third time —
  this is what caught the fp16/GradScaler crash cheaply (a few seconds on
  a 0.5B model) instead of on another multi-hour Kaggle attempt. Directly
  validates this session's established rule: any training-loop change
  needs to be locally re-verified before spending Kaggle GPU-hours on it,
  not assumed correct because the reasoning sounded right.
- Did not fabricate confidence in the reverted fp16 change by leaving weak
  supporting evidence unexamined — ran the actual matmul benchmark instead
  of just trusting the "Turing lacks bf16 Tensor Cores" reasoning by
  itself, found it didn't hold up (on the available hardware, imperfect
  proxy as it was), and said so plainly rather than quietly dropping the
  change without explaining why the reasoning didn't survive contact with
  a real test.

### Difficulty encountered
- Two full Kaggle sessions (one OOM, one 12-hour timeout) were needed to
  reach a training run that should actually work, both consuming real
  weekly GPU-hour quota. Neither was avoidable in hindsight without
  Kaggle-scale hardware to test against locally — the OOM only manifests
  at 3B-model memory pressure the smoke test deliberately avoids, and the
  multi-GPU pipeline-parallelism bug only manifests with >1 GPU visible,
  which the local single-GPU dev machine can never reproduce.
- Practical, non-technical cost worth naming: the "T4 x2" accelerator
  choice itself was silently doubling the Kaggle weekly quota consumption
  for a run that (with the fix) only needs one GPU — worth flagging to the
  user directly as an action item, not just a code fix.

### What I'd do differently
- Would have applied `generate.py`'s `device_map={"":0}` pattern to
  `train_sft.py` at the same time it was first written, given both scripts
  load a model onto a Kaggle GPU and the multi-GPU pipeline-parallelism
  risk applies equally to both — the fix existed in one file already, and
  should have been treated as a checklist item for every model-loading
  code path in this project, not discovered independently per script via
  a real failure each time.

### Addendum: `max_seq_length` was never once being used
- User asked whether P100 or TPU would be a better fit than T4, worried
  T4 would time out again. Both ruled out directly: P100 already confirmed
  incompatible with Kaggle's current PyTorch build (sm_60 unsupported);
  TPU is a non-starter for this pipeline specifically because `bitsandbytes`
  (the 4-bit QLoRA quantization library) is CUDA-only with no TPU support
  at all — using a TPU would mean dropping 4-bit quantization and rewriting
  the training stack around `torch_xla`/JAX, not a config change. T4 (x1,
  not x2) is correct; the earlier timeout was the un-pinned multi-GPU bug,
  not an inherent T4 speed problem.
- While re-checking whether T4-x1 would actually finish in time, tokenized
  every row's `prompt+target`/`prompt+test-target` across all three splits
  to sanity-check `training.max_seq_length: 2048` against real data,
  instead of just asserting the fix above was sufficient. Found the true
  max was 983 tokens (train), 775 (test), 733 (valid), with p99 under 630
  everywhere — 2048 was never once being hit, just adding needless
  worst-case memory/compute headroom to every batch containing a
  comparatively longer example. Reduced to 1024 (config.yaml) — ~4% margin
  above the true max, zero truncation risk, one more lever (on top of the
  device pin) working in favor of the next attempt finishing inside a
  session.

---

## 2026-08-18 — Third Kaggle attempt still too slow (confirmed single-GPU); evaluated Unsloth

### What I did
- With the device-pin + reduced-`max_seq_length` fixes applied, the user
  reran training on Kaggle. The `GPU 0`/`GPU 1` memory diagnostic added
  the previous entry confirmed the pin genuinely works this time (`GPU 0:
  2.20 GB`, `GPU 1: 0.00 GB`) — the multi-GPU pipeline-parallelism bug is
  real fixed. But the run was still measured at ~117-125s/step via the
  (now-unbuffered, thanks to `PYTHONUNBUFFERED=1`) tqdm output, against
  564 total steps — projecting to ~19.5 hours, still over the 12-hour cap.
  This means single-GPU pinning alone wasn't sufficient; something else
  is still making a 3B QLoRA step ~15-20x slower than the 2-8s/step a
  properly single-GPU-pinned run should see.
- Ran a local timing comparison (0.5B model, same batch/seq-len config as
  Kaggle) with vs. without gradient checkpointing to see if checkpointing
  itself was the dominant cost, to decide whether to trade the earlier
  OOM-safety margin for speed. Got 12.3s/step *with* checkpointing on the
  0.5B local model before the comparison could finish (the "without"
  side was interrupted when the conversation moved to evaluating a
  different fix instead — not concluded).
- User proposed switching to Unsloth (custom Triton kernels for QLoRA,
  specifically built for exactly this "3B-7B QLoRA on a single Kaggle/
  Colab T4" scenario) instead of continuing to chase the vanilla `trl`/
  `bitsandbytes` slowdown blind. Strong, well-targeted idea — evaluated it
  seriously rather than either dismissing it or adopting it uncritically.
- Also asked about P100 (already ruled out earlier — Kaggle's current
  PyTorch build doesn't support sm_60) combined with Unsloth. Ruled out
  independently: `bitsandbytes` generally needs compute capability 7.0+,
  and Unsloth's Triton kernels are built/tested primarily for Turing+.
  Stacking Unsloth on top of P100 adds risk on top of an already-known
  hard incompatibility, for zero benefit (same 16GB VRAM as T4 either
  way). T4 + Unsloth is Unsloth's actual flagship supported combination.
- Tried to validate Unsloth locally before touching `train_sft.py`, per
  this project's standing rule of local-testing training-loop changes
  before spending Kaggle GPU-hours on them. `pip install unsloth` (no
  flags) silently upgraded `torch` from the working `2.5.1+cu121` to
  `2.11.0` with no CUDA tag, breaking `torch.cuda.is_available()` —
  exactly the kind of environment-clobbering the user's own pasted
  reference doc had already warned about implicitly, by specifying
  `unsloth[colab-new]` + `--no-deps` for the rest rather than a bare
  install. Restored `torch` via `pip install torch==2.5.1 torchvision
  --index-url .../cu121`, confirmed CUDA back (`True`) and the full test
  suite green before proceeding.
- Retried with `pip install --no-deps --force-reinstall unsloth
  unsloth_zoo`, which left the pinned `torch`/`trl`/`peft`/etc. alone this
  time. Importing `FastLanguageModel` still failed, but on a different,
  deeper issue: `unsloth_zoo` imports `transformers`, which unconditionally
  imports `torchao`'s quantizer classes, which reference `torch.int1` — a
  sub-byte dtype that doesn't exist in `torch==2.5.1`, only in newer
  PyTorch releases. This left `torchao` (pulled in transitively by the
  *first*, non-`--no-deps` install attempt, never explicitly removed)
  sitting in a broken state against the restored older `torch`, which
  cascaded into breaking `peft`'s own import (`transformers` had also
  drifted to a version missing `BloomPreTrainedModel`, which `peft` 0.20.0
  still references) — `import peft` itself, and therefore `trl`'s
  `SFTTrainer`, stopped working entirely.
- Fully repaired the environment: uninstalled `unsloth`, `unsloth_zoo`,
  `torchao`, `xformers`, `triton-windows`; force-reinstalled the exact
  pinned versions (`transformers==5.15.0`, `trl==1.10.0`, `peft==0.20.0`,
  `bitsandbytes==0.50.1`, `accelerate==1.14.0`) with `--no-deps` each.
  Verified with the full 47-test suite (green) rather than trusting
  version numbers alone as proof of a working environment.

### Why this approach (and what alternative was rejected, and why)
- Stopped trying to force a local Unsloth validation once the depth of
  the problem became clear: the crash is specifically because this local
  dev machine is pinned to an older `torch` (needed for CUDA to work at
  all on this particular GTX 1650 setup), which predates what modern
  `transformers`/`torchao`/Unsloth expect. Kaggle's own container image
  is almost certainly newer (it already rejected the P100 specifically
  for being *too old* an architecture for its current PyTorch build,
  implying a fairly recent PyTorch already) — so this specific crash is
  very likely an artifact of the local environment's version pin, not
  evidence Unsloth won't work on Kaggle. Continuing to chase full local
  parity here has bad ROI: this GPU (no Tensor Cores) was never a valid
  timing proxy for the T4 target anyway, and every additional local
  install attempt risked (and did) break the working dev environment
  further, jeopardizing the ability to validate *anything* else in this
  project in the meantime.
- Chose to fully restore pinned versions via explicit `--no-deps`
  reinstalls of each package individually, rather than trying to
  surgically patch just the broken import — faster and more certain to
  reach a known-good state than debugging a partially-drifted dependency
  graph piece by piece.

### Difficulty encountered
- Two consecutive `pip install unsloth` attempts caused two different,
  real problems (silent `torch` upgrade breaking CUDA; then transitively
  broken `torchao`/`peft`/`transformers` even with `--no-deps` on the
  *second* install, because the *first* install's damage to `torchao`/
  `transformers` was never cleaned up before the second attempt). Both
  were caught before being carried into the next Kaggle run, consistent
  with this project's standing rule to verify locally first — but this
  was the first time doing so actually *broke* something that had to be
  recovered from, not just found a bug in newly-written code.

### What I'd do differently
- Would use an isolated environment (separate venv, or at minimum `pip
  install unsloth --dry-run` to preview exactly what would change) before
  installing an unfamiliar package with a large, unpinned dependency list
  directly into the project's main working `.venv` — the two-attempt
  cleanup cost more time than an isolated trial install would have.
- Next step: since local validation isn't feasible for this specific
  package on this specific hardware, the plan is to add Unsloth's
  model-loading path to `train_sft.py` as directly as reasonably
  possible (mirroring Unsloth's own documented `trl`-integration pattern,
  since it's explicitly designed as a drop-in accelerator for
  `SFTTrainer` rather than a replacement), and validate it on Kaggle
  itself with a cheap, small `--max-steps` dry run *before* any full
  training commitment — the same "prove it cheaply first" pattern used
  throughout this project, just shifted from local to Kaggle since local
  isn't a viable substitute here.

### Implementation: `--use-unsloth` opt-in flag in `train_sft.py`
- Added as a flag, not a replacement of the vanilla path: the vanilla
  `AutoModelForCausalLM`/`LoraConfig` path is what's actually been
  validated end-to-end locally (smoke test, adapter reload, real
  generation); Unsloth couldn't be locally verified at all, so it doesn't
  get to silently become the only path.
- Real structural constraint handled: Unsloth must be imported *before*
  `torch`/`transformers` for its monkey-patches to apply (a documented
  Unsloth requirement), but which path to take is a runtime `argparse`
  decision. Resolved by checking the raw `sys.argv` for `--use-unsloth`
  at the very top of the file, before any other import, and only
  importing `unsloth` there if present — avoids forking this into two
  separate scripts (would violate this project's own "one script" rule
  stated in the module's own docstring) while still respecting the
  import-order requirement.
- `FastLanguageModel.from_pretrained` + `FastLanguageModel.get_peft_model`
  replace `AutoModelForCausalLM.from_pretrained` + `LoraConfig` for this
  path — Unsloth handles quantization and device placement internally via
  `load_in_4bit` (no separate `BitsAndBytesConfig`/`device_map` needed),
  and applies LoRA directly rather than through `SFTTrainer`'s
  `peft_config=` argument, so that argument is set to `None` on this path
  to avoid double-applying LoRA.
- `use_gradient_checkpointing="unsloth"` (Unsloth's own async-offloading
  implementation) used instead of the vanilla boolean flag, and
  `SFTConfig`'s own `gradient_checkpointing` was made conditional on `not
  args.use_unsloth` to avoid enabling both implementations simultaneously
  on the same model.
- Verified the *vanilla* path is unaffected by this restructuring: reran
  the local smoke test (identical loss curve to every prior run) and the
  full 47-test suite (all green) after the changes. This is the honest
  limit of what could be verified locally — the Unsloth branch itself
  remains unverified until it runs on Kaggle.

### First real Unsloth run on Kaggle: crashed on bf16 validation, but it's a good crash

- The `--use-unsloth` dry run got past model loading and the GPU-memory
  diagnostic (`GPU 0: 2.42 GB`, `GPU 1: 0.04 GB` -- still essentially
  single-GPU) but crashed immediately after with a hard `ValueError` from
  `transformers`' own `TrainingArguments._validate_args()`: `"Your setup
  doesn't support bf16/gpu... You need Ampere+ GPU with cuda>=11.0."`
- This is a real, useful finding, not just a blocker: `train_sft.py` had
  been hardcoding `bf16=torch.cuda.is_available()` since the very first
  version of this script, with no check for whether the GPU actually
  supports bf16 (Ampere, compute capability 8.0+) versus merely being
  *able* to run bf16 ops unaccelerated (T4/Turing, 7.5). The *vanilla*
  path never raised this error and appeared to train "successfully" —
  just at 117-125s/step against an expected 2-8s/step. Unsloth's patched
  `SFTConfig`/`SFTTrainer` validates this more strictly and refuses to
  proceed instead of silently running unaccelerated. Together, this is
  strong evidence that **bf16 running without Tensor Core acceleration on
  the T4 was the actual, never-fully-explained root cause of the mystery
  slowness** across every earlier Kaggle attempt this session, not just
  an Unsloth-specific quirk.
- This directly revisits a hypothesis tried and reverted earlier the same
  day (2026-08-18, "Third Kaggle attempt" entry above): making bf16
  conditional on GPU compute capability. That attempt was reverted after
  it crashed the *vanilla* path locally with a dtype-mismatch error
  (`_amp_foreach_non_finite_check_and_unscale_cuda not implemented for
  'BFloat16'`) and a local matmul benchmark seemed to contradict the
  underlying reasoning. Re-examining that in light of this new evidence:
  the local benchmark was run on a GTX 1650, which has *no Tensor Cores
  at all* (unlike the T4), so it was never capable of validating or
  refuting a Tensor-Core-specific hypothesis either way -- noted as a
  caveat at the time, but the conclusion (revert bf16-conditionality
  entirely) went further than the caveat actually supported.
- Fixed by computing `use_bf16 = torch.cuda.is_available() and
  torch.cuda.get_device_capability(0) >= (8, 0)` and applying it to
  `bf16`/`fp16` **only in the Unsloth branch** of `SFTConfig`, deliberately
  leaving the vanilla branch's `bf16=torch.cuda.is_available()` untouched.
  Not reverting the earlier revert wholesale: the vanilla path currently
  works end-to-end (just slowly) and the fp16 crash that motivated
  reverting it was never root-caused, so changing it again without being
  able to test the specific failure mode locally would trade a known-slow
  but working path for an unknown risk, for a path (`--use-unsloth`) that's
  meant to replace it anyway.
- Also noticed, not yet acted on: Unsloth logged `"Dropout = 0 is
  supported for fast patching. You are using dropout = 0.05... Unsloth
  will patch all other layers, except LoRA matrices, causing a
  performance hit."` `config.yaml`'s `peft.lora_dropout: 0.05` is shared
  across every planned run (SFT and DPO, several ablations) for
  cross-run comparability, so didn't change it unilaterally for one run's
  sake -- worth a deliberate decision later if Unsloth's speed with
  0.05 still isn't sufficient, not a silent change now.
- Verified the fix doesn't regress the vanilla path (unaffected branch):
  reran the local smoke test — identical to every prior run — and the
  full 47-test suite, both green, before sending this back to Kaggle.

### It worked: `sft_qlora` trained end-to-end, real results in

- With Unsloth + the capability-aware bf16/fp16 fix, the full 3-epoch
  training run (1,125 steps, 5,987 rows) completed at a stable ~7-8s/step
  (confirmed steady across the whole run, not just a fast start) — total
  training time roughly 2.3-2.5 hours, comfortably inside one Kaggle
  session. This is the payoff of the whole day's debugging chain: OOM fix
  → device-pin fix → `max_seq_length` right-sizing → Unsloth → the bf16
  eligibility fix, each one necessary, none sufficient alone.
- Also (separately, by user error/oversight, not a code issue): an old,
  pre-Unsloth Kaggle notebook version had been left running in parallel
  this whole time, stuck at the same ~125s/step rate on a stale code/
  config snapshot (564 steps vs. the current 1,125 — implies an older
  effective batch size of 32 rather than 16, from before the
  `per_device_batch_size` 4→2 change). Caught by comparing tab
  URLs/script version IDs between two simultaneously-open Kaggle tabs
  showing different step counts for what looked like "the same" run —
  flagged to stop it, since it was burning GPU-hour quota in parallel for
  a run that could never finish before timeout.
- Downloaded results, placed in the repo: `adapters/sft_qlora/` (final
  adapter only -- deleted the intermediate `checkpoint-{375,750,1125}`
  subdirectories downloaded alongside it, since `trainer.save_model()`
  already writes the final state to the top level and the periodic
  checkpoints are redundant for anything downstream), `results/
  sft_qlora_gen.jsonl`, `results/sft_qlora_scored.jsonl`. Merged
  `results/summary.csv` by hand (this Kaggle session's own `summary.csv`
  only had the `sft_qlora` row, since baseline was intentionally skipped
  this run) using the exact `baseline` row values already captured
  earlier in this session — the original baseline download folder had
  been overwritten locally by the newer `results.zip`, but the numbers
  were already on record, so nothing was lost.
- Real numbers, full 803-row test set, same test set for both runs:

  | metric | baseline | sft_qlora | Δ |
  |---|---|---|---|
  | structure_ok | 0.994 | 1.000 | +0.006 |
  | groundedness | 0.731 | 0.951 | **+0.220** |
  | length_ok | 0.994 | 1.000 | +0.006 |
  | no_contradiction | 0.991 | 1.000 | +0.009 |
  | aggregate | 0.927 | 0.988 | +0.061 |
  | parse_ok_rate | 0.994 | 1.000 | +0.006 |

  Groundedness -- the hallucination-guard check, the rubric component the
  spec calls out as most important -- improved the most by a wide margin.
  Directionally exactly what SFT on grounded teacher targets should do:
  the base model already followed the IRAC format well zero-shot (0.994
  structure_ok even before tuning), so format compliance was never the
  gap; faithfulness to the source clause was, and that's what moved.
- Did not just trust the aggregate number: spot-checked 3 random scored
  rows by hand (`random.seed(1)`) and read the actual generated IRAC
  text, not just the score. All three were well-formed, legally
  coherent, and faithful to their source clause with no obvious
  hallucination -- the somewhat lower per-row `groundedness` scores on a
  couple of these (0.57, 0.63) trace to the rubric's own documented
  word-overlap fallback heuristic triggering on paraphrased (not
  directly quoted) Application text, a known limitation already
  documented in `rubric.py`'s own docstring, not a real quality problem.

---

## 2026-08-19 — `dpo_from_sft`: built `train_dpo.py` + Kaggle notebook

### What I did
- Wrote `src/train/train_dpo.py`: same parameterized-script philosophy as
  `train_sft.py` (one script for both `dpo_from_sft`, warm-started via
  `--adapter`, and `dpo_from_base`, fresh LoRA, `--adapter` omitted --
  not two scripts to keep in sync). Reuses every fix already found for
  the SFT script: single-GPU `device_map={"":0}` pin, gradient
  checkpointing, `--use-unsloth` opt-in path, and the capability-aware
  bf16/fp16 fix.
- Read `trl`'s actual installed source (`dpo_config.py`, `dpo_trainer.py`)
  directly rather than relying on memory for the DPO-specific API,
  since `trl.DPOTrainer` fails to import at all locally (see below) --
  confirmed two things worth knowing before writing a line of training
  code: (1) `DPOConfig` defaults `bf16=True` unconditionally per its own
  docstring ("bf16: Defaults to True if fp16 is not set, instead of
  False") -- the *exact* T4/Turing bug just fixed for `SFTConfig`, silently
  reintroduced by default if not explicitly overridden here too; (2) when
  `ref_model=None` and the model passed in is already a `PeftModel`,
  `DPOTrainer` uses the model itself (adapters disabled internally) as
  its own implicit reference rather than requiring a second full model
  copy -- exactly the mechanism needed for an efficient warm-start.
- Warm-start (`--adapter`) loads the existing PEFT adapter with
  `PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)`
  (vanilla path) or `FastLanguageModel.from_pretrained(model_name=
  adapter_path, ...)` (Unsloth path, which supports loading a saved
  adapter checkpoint directly) and continues training those same LoRA
  weights -- rather than merging the SFT adapter into the base model
  first, which isn't straightforward with a 4-bit-quantized base anyway.
- Real, honest limitation: `trl.DPOTrainer` cannot even be *imported* on
  this local dev machine -- it needs `torch.distributed.fsdp.FSDPModule`,
  which doesn't exist in the locally-pinned `torch==2.5.1` (added in a
  later PyTorch release). This is one level worse than the earlier
  Unsloth situation (that one could at least partially run before
  failing); this fails at the `from trl import DPOTrainer` line itself.
  Verified this wasn't fixable by a quick local patch (deliberately did
  *not* repeat the "just upgrade torch and see" mistake from the Unsloth
  saga, having already twice broken and had to fully repair the local
  environment doing exactly that on 2026-08-18) -- accepted the
  limitation and adjusted scope instead of forcing it.
- Moved the `from trl import DPOConfig, DPOTrainer` import from module
  level into `main()` specifically so `load_dpo_dataset` (pure logic,
  needs only `datasets`) stays importable and unit-testable locally even
  though `DPOTrainer` itself can't be. Verified `load_dpo_dataset` against
  the real `data/splits/dpo_valid.jsonl` (726 rows, correct `prompt`/
  `chosen`/`rejected` columns) and added `tests/test_train_dpo.py` (2
  tests, both passing). Full suite: 49/49 green.
- Built `notebooks/kaggle_dpo.ipynb`, structurally identical to
  `kaggle_train_eval.ipynb` (Unsloth install + CUDA-survived check,
  marker-based repo/data finder -- extended with a `find_dir()` helper for
  locating the uploaded `sft_qlora` adapter directory the same
  content-marker way -- dry run before full run, hyperparameters read
  live from `config.yaml`). `USE_UNSLOTH = True` from the start this
  time, rather than rediscovering the same 117-125s/step problem before
  reaching for it.

### Why this approach (and what alternative was rejected, and why)
- Did not try to force local validation of the actual `DPOTrainer`
  construction/training loop, unlike the earlier (partially successful)
  effort for `SFTTrainer`. The FSDPModule import failure is a hard
  version floor, not a fixable local config issue, and the last two
  attempts to "just upgrade a package to fix an import" on this machine
  both broke the working environment and cost real time to repair. Called
  this one early instead: documented the limitation plainly (in the
  script's own module docstring and the notebook's markdown) and leaned
  on reading `trl`'s actual source code as the substitute for runtime
  verification, plus emphasized the Kaggle dry-run cell more heavily than
  usual as the real first test of this code.
- Chose warm-start-by-continuing-the-adapter over merge-then-fresh-LoRA:
  simpler (no merge step, which is genuinely awkward with a 4-bit
  quantized base), and it's what `DPOTrainer`'s own implicit-reference-
  model mechanism is built around when the model is already a `PeftModel`
  -- fighting that design by merging first would have meant giving up the
  free reference-model memory savings for no benefit.

### Difficulty encountered
- The FSDPModule import failure, and the discipline not to try to
  "just fix" it locally given the recent cost of doing exactly that
  twice with Unsloth. Reading library source directly (rather than
  either guessing from memory or insisting on a local runtime test) was
  the resolution -- confirmed correct API usage without needing the
  import to succeed.

### What I'd do differently
- Nothing to flag yet -- the honest test is the Kaggle dry run, same as
  every other unverified-locally code path this session.

---

## 2026-08-19 — `kaggle_dpo.ipynb`'s first run: CUDA-survived assertion caught a real failure

### What I did
- User's first real Kaggle run of `kaggle_dpo.ipynb` hit the CUDA-survived
  assertion right after the Unsloth install cell: `"CUDA available after
  unsloth install: False"`. Exactly the failure this project already knew
  about from the local dev machine (2026-08-18) -- but this time it
  happened on Kaggle itself, in a run whose earlier sibling (the
  `sft_qlora` notebook) had used the *identical* bare `pip install -q
  unsloth` command and *not* broken CUDA. Confirms the failure is
  non-deterministic across sessions (likely: pip resolving a different,
  incompatible `torch` build depending on index/cache state at
  install time), not something that only affects local environments.
- The assertion did exactly its job: stopped the notebook after ~80
  seconds instead of letting it proceed to train for hours on a broken
  GPU setup, or silently fall back to slow/CPU execution unnoticed.
- Fixed by switching the install to `pip install --no-deps unsloth
  unsloth_zoo` in both `kaggle_dpo.ipynb` and (proactively, not reactively)
  `kaggle_train_eval.ipynb` -- the second one hadn't failed yet, but it
  has the exact same underlying risk (identical bare install command),
  and there's no reason to wait for it to fail there too before applying
  a fix already known to be correct. `unsloth_zoo` needs `--no-deps`
  alongside `unsloth` itself since it's unsloth's own runtime dependency
  and wouldn't get installed at all otherwise.
- Separately, made the adapter-finding logic in `kaggle_dpo.ipynb` more
  robust while looking at the user's actual Kaggle dataset structure:
  originally `find_dir(name="sft_qlora")` required a directory *literally
  named* `sft_qlora` containing `adapter_config.json`. Changed to
  `find_adapter_dir()`, which searches by content marker only
  (`adapter_config.json` + `adapter_model.safetensors` present together,
  regardless of folder name) -- consistent with every other finder in
  this project (`find_repo`, `find_data_file`) after repeated real nesting
  surprises this session. In this specific case the user's upload turned
  out to already have a properly-named `sft_qlora/` folder, so the
  original name-matching version would have worked too -- but the more
  robust version costs nothing and removes a class of failure before it
  has a chance to happen, matching the project's established pattern of
  not waiting for a name-matching assumption to break before fixing it.

### Why this approach (and what alternative was rejected, and why)
- Did not try to root-cause *why* pip resolved differently between the
  two Kaggle sessions (e.g. diffing exact package versions before/after)
  -- `--no-deps` sidesteps the non-determinism entirely rather than
  chasing a moving target across ephemeral Kaggle container states that
  can't be inspected after the fact anyway (the failed session's exact
  pip resolution isn't preserved anywhere to diff against).

### Difficulty encountered
- None beyond the assertion doing exactly what it was built for --
  this is the intended outcome of adding it in the first place, not a
  new problem.

---

## 2026-08-19 — `dpo_from_sft` complete: DPO improves further on top of SFT

### What I did
- Second `--no-deps` attempt still hit `CUDA available: False`, but this
  time checked the *first* cell's output (before Unsloth was ever
  touched) rather than assuming the Unsloth install was still the cause.
  It was already `False` there too -- meaning the Unsloth install was
  never the problem in this instance; something about that specific
  Kaggle session never had a working GPU attached, most likely transient
  (possibly the weekly GPU-hour quota, given the volume of multi-hour
  sessions run today) rather than anything code-related. Flagged this to
  the user as a "stop debugging code, check Kaggle's side" moment instead
  of proposing a third code fix for a problem the evidence no longer
  supported blaming on `pip install unsloth`.
- A subsequent fresh session ran cleanly: `749/749` steps (5,987 DPO rows
  / effective batch 8 * 1 epoch -- matches the config exactly) at a
  stable ~12.7s/step. Slower than SFT's ~7-8s/step, which is expected and
  not a regression -- DPO computes log-probabilities for both the chosen
  *and* rejected sequence each step, roughly double SFT's per-step
  forward-pass work. Total training time ~2.6-2.7 hours. Training signal
  looked healthy before eval even ran: `rewards/accuracies: 1`,
  `logps/chosen` meaningfully higher (less negative) than
  `logps/rejected`, loss decreasing.
- Downloaded results, placed in the repo: `adapters/dpo_from_sft/` (final
  adapter only -- same cleanup as `sft_qlora`, dropped the
  `checkpoint-749`/`ref` subdirectories since the top-level files already
  have the final trained state), `results/dpo_from_sft_gen.jsonl`,
  `results/dpo_from_sft_scored.jsonl`. Appended the `dpo_from_sft` row to
  the same `results/summary.csv` used for `baseline`/`sft_qlora`.
- Full pipeline result, same 803-row test set throughout:

  | metric | baseline | sft_qlora | dpo_from_sft |
  |---|---|---|---|
  | structure_ok | 0.994 | 1.000 | 1.000 |
  | groundedness | 0.731 | 0.951 | **0.985** |
  | length_ok | 0.994 | 1.000 | 1.000 |
  | no_contradiction | 0.991 | 1.000 | 1.000 |
  | aggregate | 0.927 | 0.988 | **0.996** |
  | parse_ok_rate | 0.994 | 1.000 | 1.000 |

  DPO moved groundedness another +0.034 on top of SFT's already-large
  gain (0.731 -> 0.951 -> 0.985), and aggregate crossed 0.99. This is
  exactly the intended story for the write-up: each alignment stage
  contributes a further, real improvement on the metric the spec calls
  out as most important, not just format compliance (which was already
  near-ceiling from SFT onward).
- Spot-checked 3 random scored rows by hand (`random.seed(2)`) rather than
  trusting the aggregate alone -- all three scored a perfect 1.0 and, on
  reading the actual generated text, were genuinely well-grounded,
  faithful to the source clause, and legally coherent. The strong
  aggregate number holds up under manual inspection, not just the rubric.

### Why this approach (and what alternative was rejected, and why)
- Did not propose a third variation of "fix the pip install" once the
  first cell's own log showed `CUDA available: False` before Unsloth was
  even reached -- continuing to iterate on the Unsloth install command
  would have been debugging the wrong layer entirely. Checking the
  earliest possible evidence point (the very first cell) before proposing
  another fix is what caught this.

### Difficulty encountered
- The two consecutive "CUDA broke" failures looked identical on the
  surface (same assertion message) but had different causes -- one
  (2026-08-18/first Kaggle attempt) was a real pip/dependency issue fixed
  by `--no-deps`; this one was not code-related at all. Surface-identical
  symptoms with different root causes is exactly why checking evidence
  each time matters more than pattern-matching to the last fix that
  worked.

### What I'd do differently
- Nothing to flag on the code side. If GPU-hour quota exhaustion turns
  out to be the actual recurring cause going forward, worth tracking
  Kaggle session start times/durations somewhere to budget the remaining
  ablations (`sft_lora_fp`, `dpo_from_base`) against the weekly quota
  rather than discovering it mid-run again.

---

## 2026-08-19 — Built the two remaining ablation notebooks

### What I did
- Built `notebooks/kaggle_sft_lora_fp.ipynb` and `notebooks/
  kaggle_dpo_from_base.ipynb`, both minimal diffs off the already-proven
  `kaggle_train_eval.ipynb`/`kaggle_dpo.ipynb` (same Unsloth `--no-deps`
  install + CUDA-survived check, marker-based finders, dry-run-before-
  full-run discipline). No new training-script code needed -- both
  ablations were already directly expressible via existing `train_sft.py`/
  `train_dpo.py` CLI flags: `sft_lora_fp` just omits `--load-in-4bit`
  (full bf16 precision instead of 4-bit NF4), `dpo_from_base` just omits
  `--adapter` (fresh LoRA on the base model instead of warm-starting from
  `sft_qlora`). This is exactly the payoff of the "one parameterized
  script, not N separate scripts per run" rule stated in both scripts'
  own docstrings from the start.
- Flagged directly in `kaggle_sft_lora_fp.ipynb`'s intro markdown (not
  buried): this run has real, elevated OOM risk versus every prior
  Kaggle run this project has done, since the base model loads in full
  bf16 rather than 4-bit-quantized -- roughly 4x the base-weight memory
  footprint (~6GB vs. ~1.5GB for a 3B model) on the same 16GB T4 that's
  already been tuned close to its limits for the 4-bit case. Told the
  user explicitly: if the dry run OOMs, drop `per_device_batch_size`
  before retrying, don't just rerun the same command hoping it was a
  fluke.
- Verified both configs build correctly against the real `config.yaml`
  and parse via the actual `argparse` definitions before handing them
  over -- confirmed `sft_lora_fp`'s args produce `load_in_4bit=False` and
  `dpo_from_base`'s args produce `adapter=None`, i.e. each ablation
  actually isolates the one variable it's meant to isolate and nothing
  else drifted. Full 49-test suite still green (nothing in `src/`
  changed for this entry, but confirmed anyway before handing off).

### Why this approach (and what alternative was rejected, and why)
- Did not build a single combined notebook running both ablations
  sequentially in one session, even though that's possible (both are
  independent, don't share a warm-start dependency the way `dpo_from_sft`
  needed `sft_qlora`). Kept them as two separate notebooks instead --
  lets the user run them in parallel across two Kaggle sessions if quota
  allows (faster wall-clock), and keeps each notebook's failure surface
  isolated (an OOM in `sft_lora_fp` shouldn't block or complicate
  `dpo_from_base`'s independent, lower-risk run).

### Difficulty encountered
- None -- this was a comparatively easy addition specifically because
  every hard problem (multi-GPU pinning, OOM tuning, bf16 eligibility,
  Unsloth install fragility) had already been found and fixed for the
  first two runs, and both scripts were built parameterized enough from
  the start to express these ablations without new code.

### What I'd do differently
- Nothing to flag yet -- as with every other not-yet-run notebook this
  session, the honest test is what Kaggle reports back.

---

## 2026-08-19 — `sft_lora_fp`: training succeeded, eval crashed on a `torchao` version bug

### What I did
- `kaggle_sft_lora_fp.ipynb`'s first real run: training completed cleanly
  -- 1,125/1,125 steps, 3h15m, `train_loss: 0.8433`, adapter saved. No OOM,
  despite this being the first run at full bf16 base-model memory
  footprint (the elevated-risk warning in the notebook's own intro didn't
  materialize this time). Then it crashed immediately at the start of
  eval generation: `ImportError: Found an incompatible version of
  torchao. Found version 0.10.0, but only versions above 0.16.0 are
  supported`, raised from inside `peft.tuners.lora.torchao.
  dispatch_torchao` while `PeftModel.from_pretrained` was trying to wrap
  the base model's layers with the trained LoRA adapter.
- Root-caused before reacting: `peft` tries a sequence of "dispatcher"
  functions to decide how to wrap each target layer, based on the base
  layer's type (bitsandbytes-quantized, torchao-quantized, or plain
  `nn.Linear`). For every prior run (`sft_qlora`, `dpo_from_sft` -- both
  4-bit-quantized bases), the bitsandbytes dispatcher matches first and
  the torchao dispatcher is never reached. `sft_lora_fp` is the *first*
  run in this whole project loading an adapter onto a full-precision
  (plain `nn.Linear`) base -- which reaches the torchao dispatcher, whose
  own version-compatibility check raises `ImportError` outright (rather
  than returning `None`/false and letting dispatch fall through to the
  correct plain-`nn.Linear` handler) when it finds an installed version
  below `peft`'s required minimum. Kaggle's base image ships
  `torchao==0.10.0`; `peft==0.20.0` wants `>=0.16.0`.
- Did not lose the 3h15m of completed training: the crash happened
  *after* `"Adapter saved to /kaggle/working/adapters/sft_lora_fp"`
  printed, so the trained adapter was sitting in that failed run's
  Output tab regardless of the exception. Had the user download it
  directly rather than accepting a wasted retrain, copied it into
  `adapters/sft_lora_fp/` locally (dropping the redundant
  `checkpoint-{375,750,1125}` subdirectories, same pattern as
  `sft_qlora`/`dpo_from_sft`).
- Built `notebooks/kaggle_sft_lora_fp_eval_only.ipynb`: a minimal
  eval-only notebook that skips training entirely, loads the
  already-trained adapter (uploaded as a Kaggle Dataset input, found via
  the same content-marker `find_adapter_dir()` helper already used for
  `dpo_from_sft`'s warm-start), and runs just the generation + scoring
  steps -- with `!pip uninstall -y -q torchao` added before anything
  touches the adapter. This project doesn't use torchao's own
  quantization scheme anywhere (only `bitsandbytes` NF4), so nothing
  depends on it; uninstalling it entirely sidesteps the buggy version
  check rather than requiring a specific compatible version to be
  installed instead.
- Applied the same `torchao` uninstall defensively to all four other
  notebooks (`kaggle_train_eval.ipynb`, `kaggle_dpo.ipynb`,
  `kaggle_dpo_from_base.ipynb`, and the full (not eval-only)
  `kaggle_sft_lora_fp.ipynb`, for any future from-scratch rerun) even
  though only full-precision adapter loading actually triggers the bug --
  the other three always load adapters onto 4-bit-quantized bases via
  the bitsandbytes dispatch path, so they were never actually at risk.
  Costs nothing (the package isn't needed anywhere in this project) and
  removes a class of failure before it has a chance to recur, same
  "don't wait for it to bite elsewhere too" pattern as the `--no-deps`
  Unsloth fix.

### Why this approach (and what alternative was rejected, and why)
- Considered pinning/upgrading `torchao` to a compatible version instead
  of uninstalling it. Rejected: uninstalling is strictly simpler (no
  version to get right, no new package version to trust sight-unseen)
  and correct here specifically because this project has zero actual
  dependency on `torchao` -- it's a transitive presence in Kaggle's base
  image, not something any of this project's code imports or requires.
  Pinning would have solved this one symptom while leaving an unused
  dependency in place for no reason.
- Prioritized not losing the already-completed training over getting a
  "clean" single-notebook run -- building a second, small notebook was
  worth the extra file for saving 3+ hours of redundant GPU-hours,
  consistent with the project's demonstrated pattern (skipping baseline
  reruns, reusing already-downloaded adapters) of never re-deriving
  something already paid for in GPU-hours when it can be reused directly.

### Difficulty encountered
- None beyond the version-bug itself, which was quick to diagnose once
  the traceback was read carefully (the dispatcher-sequence mechanism
  and where exactly it raised made the "only affects full-precision
  loading" conclusion clear directly from the stack trace, not guesswork).

### What I'd do differently
- Would have uninstalled `torchao` defensively across all notebooks from
  the very first one (`kaggle_train_eval.ipynb`), rather than only after
  hitting the bug on the ablation run that happened to be the first to
  exercise the full-precision code path -- the same "audit every
  model-loading code path for a class of risk once one instance of it is
  found" lesson already written down once this session (2026-08-18, the
  `device_map` pin) and apparently not yet generalized into a habit
  applied proactively to *new* dependency-version risks, only fixed
  ones.

---

## 2026-08-20 — `find_adapter_dir()` had a silent wrong-adapter bug

### What I did
- Before running `kaggle_sft_lora_fp_eval_only.ipynb`, the user's Kaggle
  dataset upload showed both `adapters/sft_lora_fp/` (the real, fully
  trained adapter) *and* `adapters/sft_lora_fp_dryrun/` (the 5-step
  dry-run checkpoint from the same earlier failed run) present together
  -- both are valid upload artifacts, so nothing wrong with the upload
  itself. But `find_adapter_dir()` (used here and in `kaggle_dpo.ipynb`
  for the `sft_qlora` warm-start) returned the *first* directory
  `os.walk` happened to find containing `adapter_config.json` +
  `adapter_model.safetensors` -- and `os.walk`'s traversal order isn't
  guaranteed alphabetical on Kaggle's filesystem. It could have silently
  returned the untrained dry-run adapter instead of the real one, with
  no error at all -- the eval would have run to completion and produced
  a plausible-looking but meaningless score for a barely-trained model,
  the worst kind of bug (wrong answer, not a crash) for a project whose
  whole point is being able to trust the reported numbers.
- Caught by reading the actual dataset tree structure the user was about
  to run against, before running it -- not by the code failing on its
  own.
- Fixed in both notebooks: `find_adapter_dir()` now collects *every*
  directory matching the content marker, explicitly filters out any path
  containing `"dryrun"`, and requires exactly one survivor -- raising a
  clear, explicit error listing all candidates found if that's not the
  case, rather than silently picking one. Verified the fix directly with
  a local simulation reproducing the exact real+dryrun-present scenario
  from the user's actual dataset tree (not just reasoning about it) --
  confirmed it picks the real adapter, not just "would probably work."

### Why this approach (and what alternative was rejected, and why)
- Considered just telling the user to delete the dry-run folder from
  their upload instead of fixing the code. Rejected: that fixes this one
  upload but leaves the same silent-wrong-pick risk live for every future
  run, since dry-run artifacts are a natural, expected byproduct of this
  project's own established "always dry-run before the full run" pattern
  -- the code should be robust to its own artifacts being present, not
  rely on the user remembering to clean up before every upload.
- Chose "require exactly one match, error otherwise" over "prefer the
  non-dryrun one automatically without erroring" -- erroring loudly when
  the situation is ambiguous is safer than silently guessing right this
  time and wrong some other time under a different naming pattern not yet
  anticipated.

### Difficulty encountered
- None -- found by inspection before it could cause harm, not by
  debugging a failure after the fact.

### What I'd do differently
- This is the second time this session a "return the first match" helper
  turned out to have a latent correctness bug once more than one valid
  candidate existed (first the multi-GPU pipeline-parallelism assumption,
  now this). Worth treating "first match wins" as a code smell to
  double-check specifically, not just "search by content marker instead
  of assuming a name/path" -- the search strategy was already right, the
  bug was in what to do when it finds more than one.

### Addendum: the fix itself wasn't complete on the first pass
- Running the just-fixed `find_adapter_dir()` for real immediately raised
  its own new error -- correctly this time, not silently wrong -- because
  the uploaded dataset (a raw Kaggle output download, not hand-curated)
  also contained the real adapter's own intermediate epoch checkpoints
  (`checkpoint-{375,750,1125}`, from `save_strategy="epoch"`), each of
  which *also* matches the content marker. The `dryrun`-only exclusion
  filter didn't anticipate this second, independent source of the same
  ambiguity class. Extended the filter to also exclude any path
  containing `checkpoint-`, re-verified with a local simulation
  reproducing the user's *exact* observed directory listing (dry-run +
  three epoch checkpoints + the real top-level adapter, six directories
  total) before handing the fix back a second time.
- Worth being honest about the shape of this: the first fix (2026-08-20,
  same entry above) was framed as "caught by inspection before running,"
  which was true for the dry-run case -- but the checkpoint-subdirectory
  case was *not* anticipated in advance and was only found because the
  now-stricter code raised a clear, diagnosable error instead of silently
  picking wrong. That's the fix category working as intended (loud
  failure over silent wrong answer), not a sign the fix was pointless --
  but it's a real instance of "fixed it, tested the fix, still incomplete
  on the first pass," worth naming rather than only logging the version
  that worked.

---

## 2026-08-20 — `sft_lora_fp` eval complete: full precision doesn't beat QLoRA

### What I did
- `kaggle_sft_lora_fp_eval_only.ipynb` ran cleanly with the fixed
  `find_adapter_dir()` (correctly resolved to the top-level adapter, not
  a checkpoint or dry-run artifact) and the `torchao` uninstall. Real
  result: `n=803, structure_ok=1.0, groundedness=0.9473, length_ok=1.0,
  no_contradiction=0.995, aggregate=0.9856, parse_ok_rate=1.0`. Placed
  `results/sft_lora_fp_gen.jsonl`/`sft_lora_fp_scored.jsonl` and appended
  the row to `results/summary.csv`. Spot-checked 2 random rows by hand
  (`random.seed(3)`) -- both well-grounded, faithful, perfect 1.0
  aggregate.
- Genuinely interesting ablation finding, not just "another number": full
  precision (`sft_lora_fp`) scored marginally *worse* than 4-bit QLoRA
  (`sft_qlora`) on every dimension that differed at all -- aggregate
  0.9856 vs. 0.9877, groundedness 0.9473 vs. 0.9509, no_contradiction
  0.995 vs. 1.0. Counter to the naive expectation that quantization noise
  should only ever hurt, never help. The gaps are small (aggregate delta
  0.0021) -- plausibly within run-to-run training noise rather than a
  robust effect, and this project only ran each configuration once, so
  the honest conclusion for the write-up is "QLoRA's 4-bit quantization
  cost nothing here, possibly even net-neutral-to-positive" rather than
  claiming a real, replicated full-precision-is-worse effect from a
  single run each.

### Full results table so far (same 803-row test set throughout)

| run_id | aggregate | groundedness |
|---|---|---|
| baseline | 0.927 | 0.731 |
| sft_qlora | 0.988 | 0.951 |
| sft_lora_fp | 0.986 | 0.947 |
| dpo_from_sft | 0.996 | 0.985 |

### Why this approach (and what alternative was rejected, and why)
- Did not overstate the full-precision-vs-QLoRA finding as a real effect
  given only one run per configuration -- stated the honest uncertainty
  (could be noise) directly rather than let a clean-looking number imply
  more confidence than a single run supports.

### Difficulty encountered
- None -- this run went cleanly once the two adapter-finder bugs and the
  `torchao` issue were fixed.

---

## 2026-08-20 — `dpo_from_base` complete: all five planned runs done

### What I did
- Ran `kaggle_dpo_from_base.ipynb` in parallel on a second Kaggle account
  (to save wall-clock time now that every hard problem this session was
  already solved) while `sft_lora_fp`'s eval finished on the first. Hit
  one trivial setup issue along the way (`sft_test.jsonl` missing from
  the second account's dataset upload -- just a missing file, not a code
  bug) and fixed it by uploading the already-existing local copy.
- Real result: `n=803, structure_ok=1.0, groundedness=0.9763,
  length_ok=1.0, no_contradiction=1.0, aggregate=0.9941, parse_ok_rate=1.0`.
  Placed `adapters/dpo_from_base/` (final adapter only, same
  checkpoint/`ref`-subdirectory cleanup as every prior adapter) and
  `results/dpo_from_base_{gen,scored}.jsonl`, appended the row to
  `results/summary.csv`. Spot-checked 2 random rows by hand
  (`random.seed(4)`) -- both well-grounded, faithful, perfect 1.0
  aggregate.

### Full results table -- all five planned runs complete (same 803-row test set throughout)

| run_id | aggregate | groundedness | notes |
|---|---|---|---|
| baseline | 0.927 | 0.731 | zero-shot |
| sft_qlora | 0.988 | 0.951 | SFT, QLoRA 4-bit |
| sft_lora_fp | 0.986 | 0.947 | SFT, full precision (marginally *below* QLoRA) |
| dpo_from_base | 0.994 | 0.976 | DPO alone, no SFT warm-start |
| dpo_from_sft | **0.996** | **0.985** | SFT then DPO (best overall) |

### The substantive finding worth highlighting in the write-up
- `dpo_from_base` (0.994 aggregate, 0.976 groundedness) came within
  0.002 aggregate / 0.009 groundedness of the full `dpo_from_sft`
  pipeline (0.996 / 0.985) -- **and clearly beat SFT alone**
  (`sft_qlora`: 0.988 / 0.951), despite never going through supervised
  fine-tuning at all. DPO's own preference signal, applied directly to
  the base model, got further on its own than SFT alone did.
- Plausible explanation, not just a surprising number to report
  uncritically: the `chosen` side of every DPO pair *is* the same clean
  teacher-generated target SFT trains on (`build_dpo.py` -- one DPO pair
  per cleaned SFT row, `chosen` = that row's `target` verbatim). So
  `dpo_from_base` is implicitly learning to reproduce the same targets
  SFT learns to reproduce, *plus* it gets the additional, explicit signal
  of what to avoid (the six corruption types from `corrupt.py`) that
  vanilla SFT never sees at all. That extra contrastive signal likely
  explains why it doesn't just approach SFT's performance but exceeds
  it, without needing SFT as a separate stage first.
- Still the right call to run the full ablation grid rather than assume
  this from first principles beforehand: it's a substantially more
  interesting and defensible claim backed by a real head-to-head number
  than either "DPO needs SFT first" (the common default assumption this
  project's own spec started from, given `dpo_from_sft` was marked
  `minimum_viable: true` and `dpo_from_base` was the optional ablation)
  or "DPO alone is obviously enough" asserted without the comparison.

### Difficulty encountered
- None beyond the trivial missing-file setup issue on the second
  account, resolved immediately once flagged.

### What I'd do differently
- Nothing on the code/process side. On interpretation: would want to
  caveat in the write-up that this is one run per configuration, not a
  multi-seed comparison -- the `dpo_from_base` vs. `dpo_from_sft` gap
  (0.002 aggregate) is small enough that its *ranking* (DPO-then-more-DPO
  still wins) is probably robust, but the *precise* margin shouldn't be
  overstated as more precise than a single run supports, same caveat
  already applied to the `sft_lora_fp` vs. `sft_qlora` comparison.

---

## 2026-08-20 — LLM-as-judge: built, validated locally, ready for Kaggle

### What I did
- With the deterministic rubric side of the dual-metric design complete
  across all 5 runs, built the second half: `src/eval/llm_judge.py`.
  Scores three dimensions per response (1-5 each) rather than a single
  quality number -- `groundedness` (faithfulness to the clause,
  specifically meant to catch what the rubric's word-overlap proxy can't:
  a fluent paraphrase that subtly invents a condition), `reasoning_quality`
  (is Rule -> Application -> Conclusion actually coherent and legally
  sound, not just present), and `overall` (holistic usefulness). A
  required one-line rationale per score, for interpretability and for
  cross-checking against the upcoming manual validation pass.
- Reused `generate_teacher_targets.py`'s proven Ollama HTTP pattern
  directly (`--host`/`--model`/`--concurrency`, resumable by id) rather
  than inventing a new mechanism -- this is the same class of problem
  (many LLM calls against a served model) already solved once this
  project.
- Built `src/eval/sample_judge_ids.py`: picks a seeded, common 150-id
  subsample from the test set once, shared across all 5 runs' judging.
  Judging the full 803 x 5 = 4,015 rows wasn't worth it for a comparison
  metric -- same reasoning that justified the earlier 60-row manual
  review sample over reviewing the full SFT dataset by hand (see
  `sample_for_review.py` docstring). 750 judge calls total instead.
- Added `tests/test_llm_judge.py` (5 tests) for the response parser --
  the one piece of this pure logic testable without a live model: parses
  a well-formed response, tolerates a stray preamble sentence before the
  required format (a real, observed judge-model quirk despite the prompt
  saying "EXACTLY this format, nothing else"), rejects malformed/
  out-of-range responses, and handles multiline rationale text. All 54
  project tests green.
- Validated the *whole* pipeline locally against 3 real rows from
  `sft_qlora`'s generations (Ollama already installed + the judge model
  already pulled locally from earlier smoke-testing work) before writing
  a single line of Kaggle notebook code -- this project's standing rule.
  All 3 parsed cleanly (~20s/call locally, consistent with this GPU's
  known throughput). More importantly, read the actual output, not just
  the parse-success rate: the judge gave genuinely differentiated,
  substantive scores (4/3/3 on one row specifically because "makes an
  assumption about exclusivity that isn't explicitly stated" -- a real,
  specific critique the deterministic rubric's word-overlap check would
  have missed entirely) rather than rubber-stamping everything 5/5/5.
  That's the actual bar for "is this judge worth trusting," not just "did
  the regex match."
- Built `notebooks/kaggle_llm_judge.ipynb`, structurally identical to
  `kaggle_teacher_gen.ipynb` (same Ollama-on-T4 setup: zstd + curl
  install, `OLLAMA_NUM_PARALLEL` background serve, warmup + GPU-placement
  check) with the marker-based repo/data finders from the later
  notebooks. Judges all 5 runs sequentially (not parallel across runs --
  keeps the full `OLLAMA_NUM_PARALLEL` concurrency available to each
  run's own internal batching rather than splitting it 5 ways) against
  the same 150-id file, writing `{run_id}_judged.jsonl` per run plus a
  printed summary table.

### Why this approach (and what alternative was rejected, and why)
- Considered a single 1-10 "quality" score instead of three 1-5
  dimensions. Rejected: a single number collapses exactly the distinction
  this second metric exists to surface -- a response can be well-grounded
  but poorly reasoned, or vice versa, and the rubric's own aggregate
  already has this exact collapsing problem (a known, documented
  limitation from earlier in this project). Three dimensions costs one
  more field to parse and report, for real added signal.
- Considered judging the full 803-row test set per run. Rejected on cost/
  benefit: 4,015 calls vs. 750 for a comparison metric that doesn't need
  per-row precision the way, say, the training data's quality did --
  established precedent (the 60-row manual review sample) for scoping a
  human/LLM-judgment step down from "the whole dataset" without losing
  the ability to draw a real conclusion.

### Difficulty encountered
- None -- Ollama and the judge model were already set up locally from
  earlier smoke-testing work in this project, so validation was fast.

### What I'd do differently
- Nothing to flag yet -- next step is the real Kaggle run, which is the
  first test of the notebook itself (the local validation covered the
  scoring logic and prompt design, not the Kaggle-specific setup steps).

---

## 2026-08-20 — Discovered `baseline_gen.jsonl` was never actually copied into the repo

### What I did
- Listing `results/*_gen.jsonl` to double check all 5 files existed
  before pointing the user at the LLM-judge notebook turned up only 4 --
  `baseline_gen.jsonl` was missing. Root cause, traced back through this
  session's history: the *original* baseline download (2026-08-19) was
  read directly from `~/Downloads/results (1)/...` and its aggregate
  numbers recorded into `results/summary.csv` by hand, but the raw
  per-row generation file itself was never actually copied into the
  repo -- and that Downloads folder was later overwritten by a
  subsequent `results.zip` download for a different run, before the gap
  was noticed. The aggregate score was never at risk (already on
  record), but the per-row raw text needed for LLM-judging was
  genuinely gone.
- Rather than redo the full 803-row baseline eval to fill this gap,
  regenerated *only* the 150-row LLM-judge common subsample --
  `llm_judge.py`'s `--ids-file` filter would only use those 150 rows out
  of a full 803-row baseline run anyway, so a full re-run would spend
  GPU-hours reproducing 653 rows that were never going to be judged.
  Filtered `data/splits/sft_test.jsonl` down to the 150 sampled ids
  locally (`data/raw/sft_test_judge_sample.jsonl`, all 150 found) and
  built `notebooks/kaggle_baseline_judge_sample.ipynb` -- a minimal
  zero-shot generation notebook (reuses `generate.py` directly, no new
  logic) scoped to just this file.

### Why this approach (and what alternative was rejected, and why)
- Considered treating this as reason to redo the full baseline eval
  (would also refresh `results/baseline_scored.jsonl`, which is likewise
  missing for the same reason). Rejected for now: the aggregate baseline
  number is already trustworthy (recorded directly from the original
  download before it was lost), and nothing currently in progress needs
  the full per-row baseline file except the LLM-judge step, which only
  needs the 150-row subset. Revisit if a future step (e.g. the write-up's
  qualitative examples) turns out to need full-baseline per-row text.

### Difficulty encountered
- None -- caught by a routine "does the file actually exist" check
  before handing the user a notebook that assumed it did, not by a
  failure partway through the Kaggle run.

### What I'd do differently
- Would copy raw generation files into the repo immediately upon
  download, every time, rather than ever reading numbers directly out of
  a Downloads folder and treating that as "done" -- this is the second
  time this session a Downloads-folder file got overwritten before being
  properly placed (the first was the original baseline summary itself,
  recovered only because the aggregate numbers happened to already be
  quoted in the conversation). The lesson from the first near-miss
  (verify placement) didn't fully generalize to "and check every file the
  download contained," not just the one immediately needed at the time.

---

## 2026-08-21 — Two Kaggle bugs on the LLM-judge run, then real results

### What I did
- First failure: `kaggle_llm_judge.ipynb`'s Ollama readiness-check loop
  crashed with `ReadTimeout` instead of retrying. Root cause: the loop
  only caught `requests.exceptions.ConnectionError`, but Ollama's own
  install log had already printed "API is now available" before the
  server was actually ready to answer requests -- that specific
  timing window surfaces as a *read* timeout (port open, no response
  yet), a different exception class than "connection refused" (port not
  open at all), and the loop didn't retry on it. This is the exact same
  readiness-check pattern used successfully in `kaggle_teacher_gen.ipynb`
  earlier in the project -- it just never happened to hit this timing
  window there. Fixed by catching `(ConnectionError, Timeout)` instead
  of `ConnectionError` alone.
- Initially also patched the already-completed `kaggle_teacher_gen.ipynb`
  with the same fix, on autopilot from this session's "fix defensively
  everywhere" habit. User correctly pushed back: that notebook's one job
  is already done and it won't run again, so the fix had zero practical
  value there -- reverted that change. Worth naming: "fix every instance
  of a bug class" is the right instinct for code that will run *again*,
  not for a one-shot artifact that already succeeded. Conflating the two
  wastes a step without helping anything.
- Second failure, after the readiness fix worked: `ModuleNotFoundError:
  No module named 'src.eval.llm_judge'`. Simple stale-upload issue, not a
  code bug -- the `src.zip` uploaded for this notebook had been zipped
  before `llm_judge.py`/`sample_judge_ids.py` existed. Confirmed both
  files present locally (created today) and had the user re-zip and
  re-upload.
- Real results, all 5 runs, 150-row common subsample, 100% judge-parse
  rate throughout:

  | run_id | groundedness | reasoning | overall |
  |---|---|---|---|
  | baseline | 4.63 | 4.64 | 4.63 |
  | sft_qlora | 4.79 | 4.81 | 4.79 |
  | sft_lora_fp | 4.77 | 4.79 | 4.76 |
  | dpo_from_sft | 4.81 | 4.80 | 4.79 |
  | dpo_from_base | 4.83 | 4.84 | 4.82 |

  Placed as `results/{run_id}_judged.jsonl` + `results/
  llm_judge_summary.csv`.

- **A genuinely important divergence from the deterministic rubric, not
  just a different number**: the rubric showed baseline groundedness at
  0.73 vs. ~0.95-0.98 for every tuned variant -- a dramatic ~30%
  relative jump. The LLM judge sees baseline at 4.63/5 and every tuned
  variant clustered at 4.76-4.84/5 -- only a ~3-4% relative gap, and the
  four tuned variants are packed closely enough together (a 0.08-point
  spread) that their *relative ranking* is probably noise at 150 samples
  -- notably, the judge even placed `dpo_from_base` fractionally above
  `dpo_from_sft`, the opposite order from the rubric's own ranking.
- Checked whether this meant the judge was just being lenient/broken
  before writing any of that down as a finding: read the actual
  rationale text behind baseline's *lowest*-scoring rows, not just the
  aggregate. The judge correctly identified a real misinterpretation (a
  most-favored-nation-styled clause that the zero-shot model had actually
  misread) and scored it 2/2/2 with a specific, correct explanation --
  not a rubber-stamped high score. So the divergence is real judge
  behavior on a more forgiving absolute scale, not a broken metric:
  most zero-shot baseline outputs are fluent and "mostly right with
  minor speculative assumptions" in the judge's holistic view, even
  where the rubric's stricter word-overlap/quote-matching check flags
  them as ungrounded.

### Why this approach (and what alternative was rejected, and why)
- Did not try to force the two metrics into agreement (e.g. by
  re-tuning the judge prompt's scoring scale until it matched the
  rubric's spread more closely). The disagreement itself -- both metrics
  agree alignment helps *in direction*, disagree substantially on
  *magnitude*, and disagree on the *fine-grained ranking* among the four
  tuned variants -- is exactly the kind of finding a dual-metric
  evaluation design exists to surface, and is worth reporting honestly in
  the write-up rather than papering over by making the two metrics
  artificially agree.

### Difficulty encountered
- Two real Kaggle-side issues in a row (readiness-loop exception
  handling, stale zip), each diagnosed from the actual traceback/error
  message rather than guessed at.

### What I'd do differently
- Would default to "does this fix serve a run that hasn't happened yet"
  as an explicit check before applying a defensive fix to *every*
  instance of a bug pattern -- the `kaggle_teacher_gen.ipynb` detour was
  avoidable by asking that question up front instead of after the user
  had to ask it for me.

---

## 2026-08-21 — Manual judge validation sample built (35 rows)

### What I did
- Filled another leftover gap first: `baseline_scored.jsonl` (rubric
  scores) didn't exist for the 150-row judge sample either, same root
  cause as the earlier missing `baseline_gen.jsonl` -- the original full
  803-row `baseline_scored.jsonl` was lost along with it. Unlike
  generation, scoring is pure CPU work (`rubric.py` has no model
  dependency), so just ran `score.py` locally on the existing 150-row
  `baseline_gen.jsonl` -- took seconds. Result (aggregate 0.9264,
  groundedness 0.7056) closely matches the original full 803-row
  baseline (0.9275 / 0.7310), confirming the 150-id sample is
  representative, not a fluke subset.
- Built `src/eval/sample_judge_validation.py`: three strata, not the
  two `sample_for_review.py` used, because this sample has a second
  purpose that one didn't -- validating not just "is this judge score
  reasonable" but "do the rubric and the judge actually agree," since
  the LLM-judge run surfaced a real divergence between the two metrics
  (2026-08-21 entry above). Strata: 15 lowest-judge-overall rows, 10
  highest rubric/judge-disagreement rows (scores normalized to the same
  0-1 scale first), 10 unbiased random rows. 35 total, within the
  spec's "30-40 examples" target.
- Ran it for real: 35 rows written to `data/review/
  llm_judge_validation_sample.csv`, correctly stratified (15/10/10).
  Spot-checked the single highest-disagreement row before trusting the
  sample was actually useful, not just correctly counted: rubric scored
  it 0.725, judge scored it 5/5. Read the actual clause and response --
  the response *is* faithfully grounded (a paraphrase of a "prior
  written consent" export restriction, not a direct quote), so this
  looks like the rubric's already-documented word-overlap-fallback
  limitation underscoring a genuinely well-grounded paraphrase, with the
  judge correctly recognizing what the rubric's heuristic couldn't. A
  concrete, checkable example for the human validation pass, not just an
  abstract "these disagree" data point.
- Caught and fixed a real bug in my own test while writing it, not in
  the shipped script: `load_joined`'s `run_ids` parameter defaulted
  directly to the module-level `RUN_IDS` list in the function signature
  (`def load_joined(results_dir, run_ids=RUN_IDS)`) -- Python evaluates
  default argument values once, at function *definition* time, so a test
  that monkeypatches `RUN_IDS` afterward to point at synthetic fixture
  data wouldn't actually affect calls relying on that default, silently
  falling back to the real (non-existent, in a tmp_path fixture)
  `results/` filenames. Fixed by resolving `run_ids=None` inside the
  function body instead (`if run_ids is None: run_ids = RUN_IDS`), the
  standard idiom for exactly this pitfall. Added `tests/
  test_sample_judge_validation.py` (4 tests, one of which -- the
  end-to-end test -- is the one that would have silently misrepresented
  what it was testing without this fix) -- full suite 58/58 green.

### Why this approach (and what alternative was rejected, and why)
- Considered sampling at the clause level (one clause, all 5 runs'
  outputs shown side by side) instead of independent (run, id) rows, to
  make cross-run ranking comparisons easier to eyeball. Rejected for this
  pass: would only allow ~7 distinct clauses within a 35-row budget,
  much less coverage than 35 independent judgments, and the three-strata
  design (especially the disagreement stratum) already surfaces the most
  diagnostically useful individual rows without needing the paired
  layout.

### Difficulty encountered
- The default-argument late-binding bug -- caught by the test itself
  behaving unexpectedly (looking for files that shouldn't have been
  needed) rather than by inspection, which is a reasonable way for this
  specific bug class to surface, but worth remembering as a checklist
  item (avoid mutable/name-lookup default arguments when the referenced
  name might need to vary later, e.g. under test) for any future script
  with a similar "default to a module-level constant" parameter.

### What I'd do differently
- Nothing on the sampling design. Would write default arguments as
  `None`-resolved-in-body as a standing habit for any function whose
  default references a module-level constant, rather than learning this
  per-instance via a confused test failure each time.

---

## 2026-08-21 — LLM judge's own reliability: a real, verified groundedness blind spot

### What I did
- The 35-row sample came back reviewed via an external tool, with
  `model_verdict`/`model_notes` filled in (an AI-generated triage pass,
  not the human review the sample was built for -- `human_verdict`/
  `human_notes` are still blank). The file itself self-labeled this
  plainly: "CAUTION: verdicts above are model-generated, not human. Use
  for triage... do not report as inter-rater agreement." Exactly the
  same honest self-flagging pattern as the original 60-row SFT review
  sample earlier in this project (2026-08-17) -- "this is a model
  reviewing model output... treat this as a triage pass."
- Did not take the triage pass's claims on faith. Independently
  re-read the raw clause text and response for the three most
  consequential flagged failure modes before accepting any of them as a
  real finding:
  1. **"Groundedness 5 on invented content"** (rows 24, 33): verified
     both directly. Row 33's response repeatedly refers to the clause's
     actual party "LMG" as a fabricated entity, "Mergers and
     Acquisitions (MLG)" -- a genuine hallucinated name, invented out of
     nowhere -- while the judge scored groundedness a perfect 5 and
     explicitly wrote "there are no invented facts." Row 24's response
     invents a carve-out ("does not extend to the Consultant's business
     activities or employment") directly contradicting the clause's own
     "under no circumstances" language; the judge's rationale just said
     "could be more precise," missing the fabrication.
  2. **"Phantom omission critique"** (row 0, the largest single claimed
     failure mode at 6/35 rows): verified. The response explicitly names
     the clause's required "sworn statement or certificate from a senior
     officer or auditors" mechanism -- twice -- yet the judge's rationale
     faults it for "not fully capturing the requirement for a sworn
     statement." A factually false critique, not a subjective quibble.
- All three verified claims held up under direct inspection -- this
  is a real, credible finding, not an artifact of trusting an
  AI-generated triage summary uncritically. Placed the reviewed CSV and
  findings summary at `data/review/llm_judge_validation_reviewed.csv`
  and `llm_judge_validation_findings.txt`.

### Why this matters for the write-up
- This meaningfully qualifies how much weight the LLM-judge numbers
  (2026-08-21, earlier entry) should carry, specifically on
  groundedness -- the exact dimension this whole project's evaluation
  design treats as most important. The judge can miss fabricated entity
  names and invented carve-outs while explicitly asserting no
  fabrication exists, and can issue confidently-worded critiques that
  are factually contradicted by the response it's critiquing. The
  deterministic rubric's groundedness check, despite its own documented
  false-positive risk on legitimate paraphrases (2026-08-17/19 entries),
  is at least checking something mechanical (quote/word-overlap against
  the actual clause text) rather than being satisfied by fluent,
  well-structured prose regardless of fidelity -- this is now evidence,
  not just a design assumption, that the rubric's groundedness signal
  may be more trustworthy than the judge's for this specific failure
  mode, even though the judge is better at catching *reasoning*-level
  problems (the misread MFN clause found in the first LLM-judge pass,
  2026-08-21) that the rubric can't see at all. Neither metric
  dominates the other -- write this up as a real, substantive limitation
  of LLM-as-judge for this task, not a footnote.
- The overall "54% judge-verdict upheld" and per-run/per-sample-reason
  breakdowns in `validation_findings.txt` should be reported, if at all,
  with the same caution the file itself states -- as one AI system's
  triage of another's outputs, useful for knowing where to look, not as
  a validated agreement rate. The three findings above are reportable
  because they were independently verified against source text, not
  because the triage pass said so.

### Difficulty encountered
- None -- this was a straightforward verify-before-trust pass, the same
  discipline applied consistently throughout this project to any
  automated claim about data quality.

### What I'd do differently
- The `human_verdict`/`human_notes` columns remain genuinely unfilled.
  If time allows before the write-up, a real human pass -- even just on
  the specific rows already flagged above, to confirm agreement with the
  now-twice-removed (AI reviewing AI reviewing model output) analysis --
  would be the more defensible thing to cite as "validated" rather than
  "independently spot-checked by Claude."

---

## 2026-08-21 — Results consolidated; `WRITEUP.md` produced

### What I did
- User directed: treat the AI-reviewed `llm_judge_validation_reviewed.csv`
  as the judge-validation record (skip the optional real-human pass for
  now), skip plots, and move straight to results consolidation + a
  write-up distilled from this log -- with an explicit instruction to
  make the write-up thorough on methods/results/findings.
- Built `results/consolidated_summary.csv`: rubric (`summary.csv`) and
  LLM-judge (`llm_judge_summary.csv`) scores merged side by side per
  `run_id`, one table instead of two separate files to cross-reference by
  hand.
- Re-read this entire log start to finish (~2,600 lines at that point,
  every dated entry from 2026-08-15 through the judge-reliability finding)
  before writing anything, rather than relying on working memory of a
  long session -- cross-checked every number that ended up quoted in the
  write-up against the actual `results/*.csv` files afterward (not just
  against the log's own prose) and confirmed exact matches before treating
  the document as done.
- Wrote `WRITEUP.md`: a distilled, results-oriented narrative (problem/
  approach, data pipeline, dual-metric evaluation design, training
  method + the full infrastructure debugging chain condensed to its
  throughline, manual-review data-quality findings, full results tables
  for both metrics, five numbered key findings, the LLM-judge reliability
  finding with its three independently-verified examples, limitations,
  and what's left) -- meant to be read start to finish, with `LOG.md`
  remaining the line-by-line-defensible source of record it was always
  built to be.

### Why this approach (and what alternative was rejected, and why)
- Re-read the full log rather than distill from the running summary
  already held in context -- a write-up whose entire purpose is being
  quoted accurately in an interview shouldn't be built from a summary of
  a summary; going back to the primary source (and then re-verifying its
  numbers against the actual result files, a third independent check) is
  the same "verify before trusting" discipline applied throughout this
  project to everything else, now applied to producing the interview-prep
  artifact itself.
- Kept `WRITEUP.md` and `LOG.md` as two separate documents with distinct
  jobs rather than merging them -- `LOG.md`'s chronological, warts-and-all
  structure (including reverted decisions, wrong first attempts, and
  bugs in the debugging process itself) is exactly what makes it useful
  for defending *how* a conclusion was reached under interview
  questioning; collapsing that into a clean narrative would lose the
  thing that makes the log valuable, while a narrative-only document is
  what's actually readable start to finish.

### Difficulty encountered
- None -- a synthesis task, not a debugging one.

### What I'd do differently
- Nothing to flag on this pass. Remaining work per the user's own
  priority order: FastAPI serving of `dpo_from_sft` (the best-performing
  adapter on both metrics), and the still-open option of a genuine human
  pass on the judge-validation sample if time allows before any external
  presentation of this work.

---

## 2026-08-21 — FastAPI serving: local 3B download stall, then a pip-driven CUDA regression

### What I did
- Started building `src/serving/app.py`: a FastAPI app that loads the base
  model (`Qwen/Qwen2.5-3B-Instruct`, 4-bit via `BitsAndBytesConfig`,
  `device_map={"":0}` -- the same single-GPU pin used everywhere else in
  this project) and attaches a configurable adapter (`ADAPTER_NAME` env
  var, default `dpo_from_sft`, the best result on both metrics per
  `WRITEUP.md`). Reuses `CANONICAL_PROMPT_TEMPLATE`, `parse_irac`, and
  `is_substantive_clause` from `build_irac.py` rather than redefining them
  -- one `/health` and one `/analyze` route.
- Installed `fastapi`/`uvicorn` locally first, verified with the full
  test suite that this alone didn't disturb the environment (it didn't).
- Launched the server locally. This is the first time
  `Qwen/Qwen2.5-3B-Instruct`'s full weights have ever been pulled to the
  local machine -- every previous local run only ever touched the 0.5B
  smoke-test model; the real 3B model was always loaded fresh on Kaggle.
  `uvicorn.log` sat at "Fetching 2 files: 0%" for 7-8 minutes of polling
  with no visible change.
- Diagnosed the apparent stall: killed the server (assuming it was stuck),
  then found via `Get-ChildItem` on the HF cache that the model was
  actually fully downloaded -- both safetensors shards (3.97GB + 2.2GB,
  matching the expected ~6.1GB bf16 total) were present. The `blobs/`
  directory looked empty because Windows was storing the real file
  content directly under `snapshots/` instead of `blobs/`+symlink (the
  project's already-known Windows-symlink limitation, not a download
  failure) -- checking `blobs/` alone gave a false "nothing downloaded"
  read. Killing the server mid-way was an unnecessary reaction to a
  misleading progress signal, not an actual problem.
- Restarted the server. It crashed immediately (not a hang) with
  `ImportError: Using bitsandbytes 4-bit quantization requires
  accelerate: pip install 'accelerate>=1.1.0'`. Both `accelerate` and
  `bitsandbytes` were listed in `requirements.txt` but had never actually
  been `pip install`ed locally -- every prior use of them ran on Kaggle,
  which installs its own environment.
- `pip install accelerate bitsandbytes` succeeded, but silently upgraded
  `torch` from the pinned `2.5.1+cu121` to an unrelated CPU-only
  `2.11.0` build (the local `requirements.txt` had `torch` unpinned).
  Caught this via the project's standing "verify torch/CUDA after every
  pip install" discipline -- `torch.cuda.is_available()` came back
  `False`.
- Fixed by reinstalling the exact pinned build:
  `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`.
  This fixed `torch.cuda.is_available()` but left `torchvision` (auto-
  upgraded to `0.26.0`, built against torch `2.11.0`) mismatched.
  Assumed this was harmless (`torchvision` isn't imported anywhere in
  this project) and moved on.
- That assumption was wrong: running the test suite to confirm the fix
  surfaced a real failure -- `peft` -> `transformers` -> (lazily, via
  `BloomPreTrainedModel`) touches `torchvision::nms`, and the version
  mismatch broke the op registration, which cascaded into
  `ModuleNotFoundError: Could not import module 'BloomPreTrainedModel'`
  on an unrelated model family, from inside `transformers`' lazy-import
  machinery. Root-caused to the `torch`/`torchvision` ABI mismatch, not
  a real missing symbol. Fixed by installing the version paired with
  `torch==2.5.1` per the official compatibility matrix:
  `torchvision==0.20.1 --index-url .../cu121`.
- Test suite still failed after that, now on `torchaudio` (still at
  `2.11.0`, same root cause) -- `transformers`' loss-function lazy import
  chain pulls in `torchaudio`, which tries to load a native `.dll` built
  for the wrong torch ABI: `OSError: [WinError 127] The specified
  procedure could not be found`. Fixed the same way:
  `torchaudio==2.5.1 --index-url .../cu121`.
- Also installed `pytest` and `peft` locally (neither was actually
  present locally despite being in `requirements.txt` and despite an
  earlier claim in this session that the suite had been run -- that
  claim was wrong; the local environment genuinely never had them).
- With all three torch packages aligned (`torch==2.5.1+cu121`,
  `torchvision==0.20.1+cu121`, `torchaudio==2.5.1+cu121`) and
  `accelerate`/`bitsandbytes`/`peft`/`pytest` installed, the full suite
  passed clean: 58 passed.
- Pinned `torch==2.5.1` in `requirements.txt` (was unpinned) with a
  comment recording the exact failure mode and the fix command, so this
  doesn't silently repeat.

### Why this approach (and what alternative was rejected, and why)
- Didn't just pin `torchvision`/`torchaudio` blindly after the first
  CUDA-loss scare -- re-ran the actual test suite after each fix instead
  of trusting `torch.cuda.is_available()==True` alone, which is exactly
  what caught the `torchvision` ABI break that `cuda.is_available()`
  couldn't see (it's a separate package, not part of core torch's CUDA
  check).
- Didn't assume `torchvision` being unused by *this project's own code*
  meant it was safe to leave mismatched -- `transformers` imports it
  transitively (and lazily, so the failure only surfaces on specific
  code paths, not at `import transformers` time), which is exactly the
  kind of indirect breakage that "grep the codebase for the import" would
  have missed. Running the tests, not just eyeballing import statements,
  is what actually caught it.

### Difficulty encountered
- Misdiagnosed a slow-but-real download as a stall (killed a server that
  didn't need killing) because `blobs/` alone is not a reliable progress
  signal on Windows when huggingface_hub falls back to direct-copy mode
  instead of symlinks.
- A single `pip install accelerate bitsandbytes` cascaded into a three-
  package version-alignment problem (`torch`+`torchvision`+`torchaudio`)
  because `requirements.txt` had `torch` unpinned, so pip's resolver was
  free to pick whatever `accelerate`/`bitsandbytes` were compatible with
  -- which turned out to be latest-and-CPU-only, not the pinned CUDA
  build already installed and working.

### What I'd do differently
- Check `torchvision`/`torchaudio` versions (not just `torch`) as part
  of the standard "verify environment after pip install" check on this
  project going forward, given both are transitively imported by
  `transformers` and both are version-locked to a specific `torch` ABI.
- Pin `torch` (and ideally `torchvision`/`torchaudio`) in
  `requirements.txt` from the start of any project that mixes a specific
  CUDA build with packages that don't pin their own torch dependency
  tightly -- unpinned `torch` is an invitation for exactly this kind of
  silent regression the moment any other package gets installed later.

---

## 2026-08-21 — FastAPI serving validated end-to-end

### What I did
- With the environment fixed (previous entry), restarted
  `uvicorn src.serving.app:app` locally. Clean startup this time: model
  loaded from the now-populated local HF cache (no re-download), 4-bit
  weights loaded, `dpo_from_sft` adapter attached, "Application startup
  complete" in well under a minute.
- `GET /health` -> `{"status":"ok","model_id":"Qwen/Qwen2.5-3B-Instruct","adapter_name":"dpo_from_sft","cuda_available":true}`.
- `POST /analyze` with a real termination clause (Non-Compete category)
  produced a fully well-formed IRAC response -- all four sections present,
  `parse_ok: true`, substantively correct legal reasoning (correctly
  identified the clause as a standard, enforceable notice-based
  termination provision).
- Verified the rejection path too: `POST /analyze` with `clause_text:
  "This"` (a fragment, same kind the training data itself filters out)
  correctly returned `422` via `is_substantive_clause`, not a nonsense
  generation.

### Why this approach (and what alternative was rejected, and why)
- Tested both the happy path and the rejection path before calling this
  done, per this project's standing "validate before declaring success"
  rule -- a `200` on a well-formed clause alone wouldn't have confirmed
  the substantiveness filter (ported from the training-data cleaning
  step in `build_irac.py`) actually does anything at inference time.

### Difficulty encountered
- None, once the environment (previous entry) was fixed.

### What I'd do differently
- Nothing to flag. FastAPI serving of the best adapter (`dpo_from_sft`)
  is now working end-to-end locally. Remaining optional work: an MCP
  wrapper around this service, and the still-deferred genuine human pass
  on the judge-validation sample.

---

---
