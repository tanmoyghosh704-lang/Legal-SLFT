"""Score a generate.py output file with the deterministic rubric (rubric.py)
and append one aggregate row to results/summary.csv.

Deliberately separate from generate.py: scoring is cheap/CPU-only and
re-runnable without regenerating, e.g. after a rubric fix (this project has
already needed that -- six rubric bugs found via full-dataset validation
and manual review, see LOG.md 2026-08-16/17).
"""
import argparse
import csv
import json
from pathlib import Path

from src.eval.rubric import score_response


def score_file(input_path: str, output_path: str) -> dict:
    with open(input_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    scored = []
    for row in rows:
        score = score_response(row["raw_response"], row["clause_text"])
        scored.append({**row, **score.to_dict()})

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(scored)
    summary = {
        "n": n,
        "structure_ok": sum(r["structure_ok"] for r in scored) / n,
        "groundedness": sum(r["groundedness"] for r in scored) / n,
        "length_ok": sum(r["length_ok"] for r in scored) / n,
        "no_contradiction": sum(r["no_contradiction"] for r in scored) / n,
        "aggregate": sum(r["aggregate"] for r in scored) / n,
        "parse_ok_rate": sum(1 for r in scored if r["structure_ok"] == 1.0) / n,
    }
    return summary


def append_summary_row(run_id: str, summary: dict, summary_csv: str):
    path = Path(summary_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "n", "structure_ok", "groundedness", "length_ok",
                  "no_contradiction", "aggregate", "parse_ok_rate"]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({"run_id": run_id, **summary})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="generate.py output (has raw_response, clause_text)")
    parser.add_argument("--output-file", required=True, help="per-row scored JSONL")
    parser.add_argument("--run-id", required=True, help="e.g. baseline, sft_qlora, dpo_from_sft")
    parser.add_argument("--summary-csv", default="results/summary.csv")
    args = parser.parse_args()

    summary = score_file(args.input_file, args.output_file)
    append_summary_row(args.run_id, summary, args.summary_csv)

    print(f"run_id={args.run_id} n={summary['n']}")
    for k in ["structure_ok", "groundedness", "length_ok", "no_contradiction", "aggregate", "parse_ok_rate"]:
        print(f"  {k}: {summary[k]:.4f}")
