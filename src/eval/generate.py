"""Batched generation for eval: zero-shot baseline or a trained adapter,
against a fixed test-set prompt file (test-file schema needs `id`, `prompt`,
`clause_text`; other columns are echoed through untouched).

One script for both baseline and post-training eval runs -- `--adapter` is
the only thing that differs. Omit it for the zero-shot baseline.

Applies the batched-generation fixes learned the hard way during teacher
generation (LOG.md 2026-08-16/17), even though this uses plain `transformers`
rather than Ollama: a 3B model fits comfortably on a single Kaggle GPU in
fp16, so the memory pressure that forced the Ollama pivot for the 7B teacher
doesn't apply here, but the underlying batching bugs would recur if not
fixed the same way:
  - left padding (right padding corrupts the KV cache position for
    everything queued behind the shortest sequence in a batch)
  - device_map={"": 0}, not "auto" (pipeline-parallelism across multiple
    GPUs serializes generation step-by-step -- measured ~15.6s/row on the
    teacher run, an order of magnitude slower than one pinned GPU)
  - bounded max_new_tokens (unbounded degenerate output stalls a whole
    batch on its slowest/looping member)

Resumable: skips ids already present in the output file (same pattern as
generate_teacher_targets.py), so a Kaggle session that hits its time limit
can be resumed by re-running against the same output path.
"""
import argparse
import json
import time
from pathlib import Path

import torch


def already_done_ids(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    ids = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def load_model(model_id: str, adapter_path: str | None, load_in_4bit: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )

    device_map = {"": 0} if torch.cuda.is_available() else None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quantization_config, device_map=device_map, dtype=dtype,
    )

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_batch(model, tokenizer, prompts: list, max_new_tokens: int) -> list:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return [tokenizer.decode(seq[input_len:], skip_special_tokens=True) for seq in out]


def run(input_path: str, output_path: str, model_id: str, adapter_path: str | None,
        load_in_4bit: bool, batch_size: int, max_new_tokens: int, limit: int | None):
    with open(input_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if limit:
        rows = rows[:limit]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_done_ids(out_path)
    pending = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} total, {len(done)} already done, {len(pending)} pending")
    if not pending:
        return

    model, tokenizer = load_model(model_id, adapter_path, load_in_4bit)

    with out_path.open("a", encoding="utf-8") as f:
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            start = time.time()
            responses = generate_batch(model, tokenizer, [r["prompt"] for r in batch], max_new_tokens)
            elapsed = time.time() - start
            for row, response in zip(batch, responses):
                f.write(json.dumps({**row, "raw_response": response}, ensure_ascii=False) + "\n")
            f.flush()
            n_done = min(i + batch_size, len(pending))
            print(f"{n_done}/{len(pending)} ({elapsed:.1f}s for batch of {len(batch)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default=None, help="PEFT adapter path; omit for zero-shot baseline")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--limit", type=int, default=None, help="Subsample first N rows, for smoke-testing")
    args = parser.parse_args()

    run(
        input_path=args.input_file, output_path=args.output_file, model_id=args.model,
        adapter_path=args.adapter, load_in_4bit=args.load_in_4bit, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens, limit=args.limit,
    )
