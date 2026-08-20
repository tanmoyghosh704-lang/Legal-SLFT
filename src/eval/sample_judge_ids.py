"""Pick a common subsample of test-set ids to LLM-judge across every run.

Judging the full 803-row test set x 5 runs (4,015 judge calls) is
unnecessary for a comparison metric -- a shared random subsample judged
identically across all five `*_gen.jsonl` files gives an apples-to-apples
comparison at a fraction of the cost, the same reasoning that justified a
60-row manual review sample over reviewing the full SFT dataset by hand
(see sample_for_review.py / LOG.md 2026-08-17).

Writes a JSON list of ids that src/eval/llm_judge.py's --ids-file
consumes to restrict judging to the same rows in every run.
"""
import argparse
import json
import random


def sample_ids(test_file: str, n: int, seed: int = 42) -> list:
    with open(test_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    ids = [r["id"] for r in rows]
    rng = random.Random(seed)
    return sorted(rng.sample(ids, min(n, len(ids))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="data/splits/sft_test.jsonl")
    parser.add_argument("--output", default="data/raw/llm_judge_ids.json")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ids = sample_ids(args.test_file, args.n, args.seed)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=1)
    print(f"Wrote {len(ids)} ids to {args.output}")
