"""Post-hoc cleaning of the teacher-generated SFT dataset.

Applies fixes found via manual review of the real generated dataset
(data/review/sft_review_sample.csv, see LOG.md 2026-08-17) that are cheap
to apply retroactively, without needing to re-run generation:

1. Drop rows whose clause_text is a non-substantive fragment
   (is_substantive_clause) -- these should have been filtered before
   generation; load_cuad_filtered() now does that for any future
   regeneration, but the existing dataset needs the same filter applied
   after the fact.
2. Drop rows whose id was manually flagged "bad" in the review sample.

Does NOT deduplicate by clause_text alone -- that was tried and reverted.
The same clause_text under two different categories is a genuinely
different training prompt (category is part of CANONICAL_PROMPT_TEMPLATE),
not a duplicate; 893/7,620 rows fell into exactly this shape and would
have been wrongly discarded. The real duplicate-document problem (the same
underlying contract filed under two different title strings) is a
*split-integrity* issue, not a row-count issue -- handled separately in
split.py's find_duplicate_documents(), applied when the splits are built.

Does NOT attempt to catch or repair: polarity inversions the rubric can't
detect, fabricated general legal principles, or category/clause mismatches
-- those need either the LLM judge or more manual review, not a mechanical
filter. See LOG.md 2026-08-17 for why these are left as documented
limitations rather than force-fixed.
"""
import argparse
import csv
import json
from pathlib import Path

from src.data.build_irac import is_substantive_clause


def load_known_bad_ids(review_csv_path: str) -> set:
    path = Path(review_csv_path)
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f) if row["reviewer_verdict"] == "bad"}


def clean(input_path: str, review_csv_path: str) -> tuple:
    with open(input_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    n_start = len(rows)

    known_bad_ids = load_known_bad_ids(review_csv_path)
    rows = [r for r in rows if r["id"] not in known_bad_ids]
    n_after_bad = len(rows)

    rows = [r for r in rows if is_substantive_clause(r["clause_text"])]
    n_final = len(rows)

    report = {
        "start": n_start,
        "removed_known_bad": n_start - n_after_bad,
        "removed_fragments": n_after_bad - n_final,
        "final": n_final,
    }
    return rows, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/sft_teacher_targets.jsonl")
    parser.add_argument("--review-csv", default="data/review/sft_review_sample.csv")
    parser.add_argument("--output", default="data/datasets/sft_clean.jsonl")
    args = parser.parse_args()

    rows, report = clean(args.input, args.review_csv)
    print(json.dumps(report, indent=2))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {out}")
