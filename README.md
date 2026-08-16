# Legal SLM Alignment — SFT vs DPO on Contract Clauses

Fine-tunes a small open LLM to analyze legal contract clauses in IRAC
format (Issue / Rule / Application / Conclusion), and compares alignment
methods — zero-shot baseline vs. supervised fine-tuning (SFT) vs. Direct
Preference Optimization (DPO) — under a dual-metric evaluation (a
deterministic groundedness rubric and a validated LLM judge).

**This is a controlled ablation, not a single fine-tuned model.** See
`LOG.md` for the full decision trail, including failures, and
`results/writeup.md` for the final analysis once runs are complete.

> Produces structured research analysis of contract clauses. **Not legal
> advice.**

## Stack

Windows + NVIDIA CUDA (not the Apple/MLX stack some reference
implementations use): HuggingFace `transformers` + `trl` + `peft` +
`bitsandbytes` (QLoRA 4-bit).

## Where things run

- **Local (this machine, RTX 1650 4GB):** data prep, IRAC template design,
  deterministic rubric, LLM-judge calls via local Ollama, results
  analysis/plots, FastAPI serving of a downloaded adapter, a 0.5B smoke
  test of the pipeline.
- **Kaggle (free T4, 16GB VRAM):** all real SFT/DPO training runs. A 3B
  model does not fit in 4GB VRAM even at 4-bit — this is a hardware
  constraint documented in `LOG.md`, not a "try it and see."

## Repo layout

```
LOG.md              decision & difficulty log — start here
config.yaml          models, datasets, PEFT variants, run grid
src/data/            CUAD -> IRAC pairs, DPO corruption, contract-level splits
src/train/           trl SFTTrainer / DPOTrainer + peft
src/eval/            deterministic rubric, LLM-as-judge, judge validation
src/serving/         FastAPI (+ optional MCP)
notebooks/           Kaggle training entry point
data/, adapters/     DVC-tracked (not in git)
results/             summary.csv, plots, writeup.md
```

## Setup

```
pip install -r requirements.txt   # local subset only; see LOG.md
```

Config is centralized in `config.yaml` — model choice, PEFT hyperparams,
and the run grid all live there, not hardcoded per-script.
