"""Build DPO preference pairs from the cleaned SFT dataset.

Each cleaned SFT row (already validated: parse_ok, substantive clause, not
manually flagged bad) becomes one DPO pair: the same (prompt, chosen) as
the SFT example, plus a `rejected` response produced by corrupt.py's
deterministic corruption functions (spec section 3.2). One pair per SFT
row -- corruption is free (no LLM calls), so there's no reason to
subsample.

Honesty note (spec section 3.2, restated in LOG.md): these are programmatic
corruptions of already-correct answers, not real human preference
judgments. Narrower, more mechanical objective than genuine RLHF data --
state this in the write-up.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from src.data.corrupt import make_rejected


def _seed_from_id(id_: str) -> int:
    """Deterministic seed from a row id. Python's builtin hash() is
    randomized per-process (PYTHONHASHSEED) unless explicitly disabled --
    using it here would make corruption type non-reproducible across runs,
    silently, since it would still "work" (produce *a* valid seed) every
    time, just a different one each process. hashlib is deterministic."""
    return int(hashlib.sha256(id_.encode("utf-8")).hexdigest(), 16) % (2**32)


def build_dpo_pairs(sft_rows: list) -> list:
    pairs = []
    for row in sft_rows:
        # Seed derived from id, not a running counter: makes each row's
        # corruption independent of processing order and resumable/
        # reproducible if the input set changes (e.g. after further
        # cleaning) without reshuffling every other row's corruption type.
        rng = random.Random(_seed_from_id(row["id"]))
        rejected, corruption_type = make_rejected(row["target"], rng)
        pairs.append({
            "id": row["id"],
            "contract": row["contract"],
            "category": row["category"],
            "clause_text": row["clause_text"],
            "prompt": row["prompt"],
            "chosen": row["target"],
            "rejected": rejected,
            "corruption_type": corruption_type,
        })
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/datasets/sft_clean.jsonl")
    parser.add_argument("--output", default="data/datasets/dpo_pairs.jsonl")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        sft_rows = [json.loads(line) for line in f if line.strip()]

    pairs = build_dpo_pairs(sft_rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    from collections import Counter
    dist = Counter(p["corruption_type"] for p in pairs)
    print(f"Wrote {len(pairs)} DPO pairs to {out}")
    print("Corruption type distribution:", dict(dist))
