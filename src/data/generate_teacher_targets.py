"""Generate IRAC SFT targets for queued (category, clause) rows using a
teacher model served by Ollama.

One code path for both local smoke-testing and the real Kaggle run — only
`--host`/`--model`/`--concurrency` differ:
  - Local (this laptop): --concurrency 1 against a local `ollama serve`,
    RTX 1650 4GB forces CPU/GPU split (~10 tok/s) — see LOG.md 2026-08-16.
    Used only to validate the prompt/parser on a handful of rows.
  - Kaggle: --concurrency 8+ against `ollama serve` running on the T4,
    which fits the whole 7B q4 model in VRAM (~4.9GB of 16GB). Concurrent
    requests let Ollama's own scheduler batch multiple generations on the
    GPU, instead of hand-rolling batched generation with raw `transformers`
    — that path needed four separate bug fixes (right-padding, multi-GPU
    pipeline-parallelism, unbounded batch latency, bf16 OOM) before
    producing correct output. See LOG.md 2026-08-16 and 2026-08-17.

Resumable: skips ids already present in the output file, so a Kaggle
session that hits its time limit can be resumed by re-running against the
same output path.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from src.data.build_irac import format_target, parse_irac


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


def generate_ollama(teacher_prompt: str, model: str, host: str) -> str:
    import requests

    r = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": teacher_prompt, "stream": False, "options": {"temperature": 0.3}},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["response"]


def generate_one(row: dict, model: str, host: str, retry_on_parse_fail: bool) -> dict:
    raw = generate_ollama(row["teacher_prompt"], model, host)
    sections = parse_irac(raw)
    retried_ok = False
    if sections is None and retry_on_parse_fail:
        stricter = row["teacher_prompt"] + (
            "\n\nYour previous response did not follow the required "
            "format. Respond with ONLY the four labeled sections, "
            "nothing else."
        )
        raw = generate_ollama(stricter, model, host)
        sections = parse_irac(raw)
        retried_ok = sections is not None
    return {
        "id": row["id"],
        "contract": row["contract"],
        "category": row["category"],
        "clause_text": row["clause_text"],
        "prompt": row["prompt"],
        "raw_teacher_response": raw,
        "target": format_target(sections) if sections else None,
        "parse_ok": sections is not None,
    }, retried_ok


def run(queue_path: str, output_path: str, model: str, host: str,
        max_rows: int | None = None, concurrency: int = 1, retry_on_parse_fail: bool = True):
    queue_path = Path(queue_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = already_done_ids(output_path)
    print(f"{len(done)} rows already done, resuming.")

    rows = []
    with queue_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] not in done:
                rows.append(row)
    if max_rows is not None:
        rows = rows[:max_rows]
    print(f"{len(rows)} rows to generate with concurrency={concurrency}.")

    n_parse_failures = 0
    n_retried_ok = 0
    n_done = 0
    t_start = time.time()
    write_lock = Lock()

    with output_path.open("a", encoding="utf-8") as out_f, \
            ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(generate_one, row, model, host, retry_on_parse_fail): row
            for row in rows
        }
        for future in as_completed(futures):
            record, retried_ok = future.result()
            if not record["parse_ok"]:
                n_parse_failures += 1
            if retried_ok:
                n_retried_ok += 1
            with write_lock:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta_min = (len(rows) - n_done) / rate / 60 if rate > 0 else float("inf")
                print(f"[{n_done}/{len(rows)}] parse_ok={record['parse_ok']} "
                      f"elapsed={elapsed:.0f}s eta={eta_min:.1f}min "
                      f"parse_failures_so_far={n_parse_failures}")

    print(f"Done. parse_failures={n_parse_failures} recovered_on_retry={n_retried_ok}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="data/raw/sft_generation_queue.jsonl")
    parser.add_argument("--output", default="data/raw/sft_teacher_targets.jsonl")
    parser.add_argument("--model", default="qwen2.5:7b-instruct-q4_0")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1,
                         help="Parallel Ollama requests. Use 1 locally, 8+ on Kaggle "
                              "(also set OLLAMA_NUM_PARALLEL on the server to match).")
    args = parser.parse_args()
    run(args.queue, args.output, args.model, args.host, args.max_rows, args.concurrency)
