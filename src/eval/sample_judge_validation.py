"""Build the manual validation sample for the LLM judge's own scores.

Same rationale as sample_for_review.py's stratified sample for the SFT
dataset: an LLM (or, here, an LLM judge) scoring its own kind of output
isn't a substitute for human judgment, it's a triage pass that needs
spot-checking against it. Three strata, not two, because this sample has
a second purpose sample_for_review.py's didn't -- validating not just
"is this score reasonable" but "do the rubric and the judge actually
agree," since the LLM-judge run surfaced a real divergence between the
two metrics (see LOG.md 2026-08-21):

  - n_low_judge: the judge's own lowest-scoring rows -- the cases most
    likely to be either a legitimate hard case or a judge error.
  - n_disagreement: rows where the rubric and judge scores diverge most
    (normalized to the same 0-1 scale) -- exactly where human
    adjudication matters most for a dual-metric design, since agreement
    everywhere would make the second metric redundant.
  - n_random: an unbiased broad check, not just already-known-interesting
    cases.

Requires both `{run_id}_scored.jsonl` (rubric) and `{run_id}_judged.jsonl`
(LLM judge) to exist for every run -- join key is (run_id, id).
"""
import argparse
import csv
import json
import random
from pathlib import Path

RUN_IDS = ["baseline", "sft_qlora", "sft_lora_fp", "dpo_from_sft", "dpo_from_base"]


def load_joined(results_dir: str, run_ids: list | None = None) -> list:
    # run_ids=None resolved here, not defaulted to RUN_IDS directly in the
    # signature -- a mutable/module-level default bound at function
    # definition time wouldn't pick up a test monkeypatching RUN_IDS later,
    # since default arguments are evaluated once at def time in Python.
    if run_ids is None:
        run_ids = RUN_IDS
    joined = []
    for run_id in run_ids:
        with open(f"{results_dir}/{run_id}_scored.jsonl", encoding="utf-8") as f:
            scored = {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}
        with open(f"{results_dir}/{run_id}_judged.jsonl", encoding="utf-8") as f:
            judged = {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}

        common_ids = set(scored) & set(judged)
        for id_ in common_ids:
            s, j = scored[id_], judged[id_]
            if not j.get("judge_parse_ok"):
                continue
            judge_overall_normalized = (j["overall"] - 1) / 4  # 1-5 -> 0-1
            joined.append({
                "run_id": run_id,
                "id": id_,
                "category": s["category"],
                "clause_text": s["clause_text"],
                "raw_response": s["raw_response"],
                "rubric_aggregate": round(s["aggregate"], 3),
                "judge_groundedness": j["groundedness"],
                "judge_reasoning_quality": j["reasoning_quality"],
                "judge_overall": j["overall"],
                "judge_rationale": j["rationale"],
                "disagreement": round(abs(s["aggregate"] - judge_overall_normalized), 3),
            })
    return joined


def build_sample(results_dir: str, n_low_judge: int = 15, n_disagreement: int = 10,
                  n_random: int = 10, seed: int = 42) -> list:
    joined = load_joined(results_dir)

    by_low_judge = sorted(joined, key=lambda r: (r["judge_overall"], r["id"]))
    low_judge = by_low_judge[:n_low_judge]
    low_judge_keys = {(r["run_id"], r["id"]) for r in low_judge}

    remaining = [r for r in joined if (r["run_id"], r["id"]) not in low_judge_keys]
    by_disagreement = sorted(remaining, key=lambda r: (-r["disagreement"], r["id"]))
    disagreement = by_disagreement[:n_disagreement]
    disagreement_keys = {(r["run_id"], r["id"]) for r in disagreement}

    remaining2 = [r for r in remaining if (r["run_id"], r["id"]) not in disagreement_keys]
    rng = random.Random(seed)
    random_sample = rng.sample(remaining2, min(n_random, len(remaining2)))

    for r in low_judge:
        r["sample_reason"] = "low_judge_score"
    for r in disagreement:
        r["sample_reason"] = "rubric_judge_disagreement"
    for r in random_sample:
        r["sample_reason"] = "random"

    combined = low_judge + disagreement + random_sample
    rng.shuffle(combined)
    return combined


def write_review_csv(sample: list, output_path: str):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id", "id", "sample_reason", "category", "clause_text", "raw_response",
        "rubric_aggregate", "judge_groundedness", "judge_reasoning_quality", "judge_overall",
        "judge_rationale", "disagreement",
        "human_verdict",  # blank: judge_right / judge_wrong / both_wrong / both_right, fill in by hand
        "human_notes",    # blank: fill in by hand
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sample:
            writer.writerow({**{k: row.get(k, "") for k in fieldnames}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="data/review/llm_judge_validation_sample.csv")
    parser.add_argument("--n-low-judge", type=int, default=15)
    parser.add_argument("--n-disagreement", type=int, default=10)
    parser.add_argument("--n-random", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample = build_sample(args.results_dir, args.n_low_judge, args.n_disagreement,
                           args.n_random, args.seed)
    write_review_csv(sample, args.output)
    print(f"Wrote {len(sample)} rows to {args.output}")
    from collections import Counter
    print("sample_reason distribution:", dict(Counter(r["sample_reason"] for r in sample)))
