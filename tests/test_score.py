import csv
import json

from src.eval.score import append_summary_row, score_file

CLAUSE = (
    "Either party may terminate this Agreement upon thirty (30) days "
    "written notice to the other party."
)

GOOD_RESPONSE = (
    "Issue: Does this clause permit unilateral termination?\n\n"
    "Rule: A termination-for-convenience clause is enforceable if it "
    "clearly states the notice period.\n\n"
    "Application: The clause permits either party to terminate upon "
    "thirty days written notice to the other party.\n\n"
    "Conclusion: The clause is a valid mutual termination-for-convenience "
    "provision conditioned on thirty days' written notice."
)

MALFORMED_RESPONSE = "This is not IRAC-formatted output at all."


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_score_file_writes_per_row_scores_and_aggregate(tmp_path):
    input_path = tmp_path / "gen.jsonl"
    output_path = tmp_path / "scored.jsonl"
    _write_jsonl(input_path, [
        {"id": "a", "clause_text": CLAUSE, "raw_response": GOOD_RESPONSE},
        {"id": "b", "clause_text": CLAUSE, "raw_response": MALFORMED_RESPONSE},
    ])

    summary = score_file(str(input_path), str(output_path))

    assert summary["n"] == 2
    assert summary["parse_ok_rate"] == 0.5

    scored_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert scored_rows[0]["id"] == "a"
    assert scored_rows[0]["structure_ok"] == 1.0
    assert scored_rows[1]["id"] == "b"
    assert scored_rows[1]["structure_ok"] == 0.0


def test_append_summary_row_writes_header_once(tmp_path):
    summary_csv = tmp_path / "summary.csv"
    append_summary_row("baseline", {"n": 2, "structure_ok": 1.0, "groundedness": 0.5,
                                     "length_ok": 1.0, "no_contradiction": 1.0,
                                     "aggregate": 0.875, "parse_ok_rate": 1.0}, str(summary_csv))
    append_summary_row("sft_qlora", {"n": 2, "structure_ok": 1.0, "groundedness": 0.9,
                                      "length_ok": 1.0, "no_contradiction": 1.0,
                                      "aggregate": 0.975, "parse_ok_rate": 1.0}, str(summary_csv))

    with summary_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["run_id"] == "baseline"
    assert rows[1]["run_id"] == "sft_qlora"
