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
