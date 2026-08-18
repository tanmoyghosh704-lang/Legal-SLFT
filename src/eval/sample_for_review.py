"""Build the manual quality-review sample for the teacher-generated SFT
dataset.

Spec section 3.1 calls for reviewing "~20-25%" of the data for quality —
written before the dataset landed at 7,620 rows, where that percentage
(1,500-1,900 rows) isn't reviewable by hand on a solo timeline. Fixed,
stratified sample instead: 25 of the lowest rubric-scoring rows (the
legitimate hard cases most likely to reveal a real problem) + 35 random
rows spread across categories (a broad quality check that isn't biased
toward already-known-hard cases). See LOG.md 2026-08-17 for the reasoning.

Output is a CSV meant to be opened and annotated by hand: each row gets a
blank `reviewer_verdict` (good / minor_issue / bad) and `reviewer_notes`
column filled in during review. Kept in git (not `data/raw/`) since the
completed review is evidence the QA step happened, not a regenerable cache.
"""
import argparse
import csv
import json
import random
from pathlib import Path

from src.eval.rubric import score_response


def build_sample(input_path: str, n_low_scoring: int = 25, n_random: int = 35, seed: int = 42) -> list:
    with open(input_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    scored = []
    for r in rows:
        s = score_response(r["target"], r["clause_text"])
        scored.append({**r, "rubric_aggregate": round(s.aggregate, 3), **{
            f"rubric_{k}": round(v, 3) for k, v in s.to_dict().items() if k != "aggregate"
        }})

    scored.sort(key=lambda r: (r["rubric_aggregate"], r["id"]))
    low_scoring = scored[:n_low_scoring]
    low_scoring_ids = {r["id"] for r in low_scoring}

    remaining = [r for r in scored if r["id"] not in low_scoring_ids]
    rng = random.Random(seed)
    random_sample = rng.sample(remaining, min(n_random, len(remaining)))

    for r in low_scoring:
        r["sample_reason"] = "low_rubric_score"
    for r in random_sample:
        r["sample_reason"] = "random"

    combined = low_scoring + random_sample
    rng.shuffle(combined)  # don't present all the low-scorers first
    return combined


def write_review_csv(sample: list, output_path: str):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "sample_reason", "category", "contract", "clause_text", "target",
        "rubric_aggregate", "rubric_structure_ok", "rubric_groundedness",
        "rubric_length_ok", "rubric_no_contradiction",
        "reviewer_verdict", "reviewer_notes",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sample:
            writer.writerow({**row, "reviewer_verdict": "", "reviewer_notes": ""})
    print(f"Wrote {len(sample)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/sft_teacher_targets.jsonl")
    parser.add_argument("--output", default="data/review/sft_review_sample.csv")
    parser.add_argument("--n-low-scoring", type=int, default=25)
    parser.add_argument("--n-random", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample = build_sample(args.input, args.n_low_scoring, args.n_random, args.seed)
    write_review_csv(sample, args.output)
