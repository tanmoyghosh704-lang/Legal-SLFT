import json

from src.eval.sample_judge_validation import build_sample, load_joined

RUN_IDS = ["run_a", "run_b"]


def _write_scored_and_judged(tmp_path, run_id, rows):
    """rows: list of (id, aggregate, groundedness, reasoning_quality, overall)"""
    scored_path = tmp_path / f"{run_id}_scored.jsonl"
    judged_path = tmp_path / f"{run_id}_judged.jsonl"
    with scored_path.open("w", encoding="utf-8") as f:
        for id_, aggregate, *_ in rows:
            f.write(json.dumps({
                "id": id_, "category": "Cat", "clause_text": f"clause {id_}",
                "raw_response": f"response {id_}", "aggregate": aggregate,
            }) + "\n")
    with judged_path.open("w", encoding="utf-8") as f:
        for id_, _, groundedness, reasoning_quality, overall in rows:
            f.write(json.dumps({
                "id": id_, "judge_parse_ok": True, "groundedness": groundedness,
                "reasoning_quality": reasoning_quality, "overall": overall,
                "rationale": f"rationale {id_}",
            }) + "\n")


def test_load_joined_skips_judge_parse_failures(tmp_path):
    scored_path = tmp_path / "run_a_scored.jsonl"
    judged_path = tmp_path / "run_a_judged.jsonl"
    with scored_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "x", "category": "C", "clause_text": "t",
                             "raw_response": "r", "aggregate": 0.9}) + "\n")
    with judged_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "x", "judge_parse_ok": False, "judge_raw": "garbage"}) + "\n")

    joined = load_joined(str(tmp_path), run_ids=["run_a"])
    assert joined == []


def test_load_joined_computes_disagreement_on_normalized_scale(tmp_path):
    # rubric aggregate 1.0 vs judge overall 5 (normalized (5-1)/4 = 1.0) -> disagreement 0
    _write_scored_and_judged(tmp_path, "run_a", [("x", 1.0, 5, 5, 5)])
    joined = load_joined(str(tmp_path), run_ids=["run_a"])
    assert len(joined) == 1
    assert joined[0]["disagreement"] == 0.0

    # rubric aggregate 0.0 vs judge overall 5 (normalized 1.0) -> disagreement 1.0
    _write_scored_and_judged(tmp_path, "run_b", [("y", 0.0, 5, 5, 5)])
    joined2 = load_joined(str(tmp_path), run_ids=["run_b"])
    assert joined2[0]["disagreement"] == 1.0


def test_build_sample_stratifies_correctly(tmp_path):
    # 10 rows per run x 2 runs = 20 rows total, enough to draw from all 3 strata
    rows_a = [(f"a{i}", 0.9, 3, 3, i % 5 + 1) for i in range(10)]
    rows_b = [(f"b{i}", 0.9, 3, 3, i % 5 + 1) for i in range(10)]
    _write_scored_and_judged(tmp_path, "run_a", rows_a)
    _write_scored_and_judged(tmp_path, "run_b", rows_b)

    # Patch RUN_IDS via monkeypatch-free approach: call load_joined directly
    # with explicit run_ids, then build_sample's internals use the module's
    # RUN_IDS constant, so test load_joined + the stratification logic
    # separately by calling build_sample with the same run_ids it defaults to
    # is not possible without editing the module -- so exercise the pieces
    # build_sample composes instead, at the level that's actually testable.
    joined = load_joined(str(tmp_path), run_ids=["run_a", "run_b"])
    assert len(joined) == 20

    from src.eval.sample_judge_validation import write_review_csv
    joined.sort(key=lambda r: (r["judge_overall"], r["id"]))
    low = joined[:5]
    remaining = joined[5:]
    remaining.sort(key=lambda r: -r["disagreement"])
    disagreement = remaining[:3]
    for r in low:
        r["sample_reason"] = "low_judge_score"
    for r in disagreement:
        r["sample_reason"] = "rubric_judge_disagreement"
    sample = low + disagreement

    out = tmp_path / "sample.csv"
    write_review_csv(sample, str(out))

    import csv
    with out.open(encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert len(written) == 8
    assert "human_verdict" in written[0]
    assert "human_notes" in written[0]
    assert written[0]["human_verdict"] == ""


def test_build_sample_end_to_end_respects_requested_counts(tmp_path):
    """Full build_sample() call, monkeypatching the module's RUN_IDS to a
    small synthetic set so this doesn't depend on the real results/ files."""
    import src.eval.sample_judge_validation as svm

    rows_a = [(f"a{i}", (i % 10) / 10, 3, 3, i % 5 + 1) for i in range(30)]
    _write_scored_and_judged(tmp_path, "run_a", rows_a)

    original_run_ids = svm.RUN_IDS
    svm.RUN_IDS = ["run_a"]
    try:
        sample = build_sample(str(tmp_path), n_low_judge=5, n_disagreement=3, n_random=2, seed=1)
    finally:
        svm.RUN_IDS = original_run_ids

    assert len(sample) == 10
    reasons = [r["sample_reason"] for r in sample]
    assert reasons.count("low_judge_score") == 5
    assert reasons.count("rubric_judge_disagreement") == 3
    assert reasons.count("random") == 2
    # no id appears twice across strata
    ids = [(r["run_id"], r["id"]) for r in sample]
    assert len(ids) == len(set(ids))
