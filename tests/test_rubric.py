from src.eval.rubric import score_no_contradiction, score_response

CLAUSE = (
    "Either party may terminate this Agreement upon thirty (30) days "
    "written notice to the other party."
)

WELL_FORMED = (
    "Issue: Does this clause permit unilateral termination, and under what conditions?\n\n"
    "Rule: A termination-for-convenience clause is enforceable if it clearly states "
    "the notice period and applies equally to both parties.\n\n"
    "Application: The clause permits either party to terminate upon thirty days written "
    "notice to the other party, without requiring cause. The right is mutual.\n\n"
    "Conclusion: The clause is a valid mutual termination-for-convenience provision "
    "conditioned on thirty days' written notice."
)


def test_well_formed_response_scores_high():
    score = score_response(WELL_FORMED, CLAUSE)
    assert score.structure_ok == 1.0
    assert score.groundedness > 0.5
    assert score.length_ok == 1.0
    assert score.no_contradiction == 1.0
    assert score.aggregate > 0.8


def test_missing_section_scores_zero_structure():
    missing_conclusion = (
        "Issue: Does this clause permit termination?\n\n"
        "Rule: Termination clauses must state a notice period.\n\n"
        "Application: The clause requires thirty days written notice."
    )
    score = score_response(missing_conclusion, CLAUSE)
    assert score.structure_ok == 0.0
    assert score.aggregate == 0.0


def test_wrong_order_scores_zero_structure():
    wrong_order = (
        "Conclusion: The clause is valid.\n\n"
        "Issue: Does this clause permit termination?\n\n"
        "Rule: Termination clauses must state a notice period.\n\n"
        "Application: The clause requires thirty days written notice."
    )
    score = score_response(wrong_order, CLAUSE)
    assert score.structure_ok == 0.0


def test_hallucinated_quote_scores_low_groundedness():
    hallucinated = (
        "Issue: Does this clause permit termination?\n\n"
        "Rule: Termination clauses must state a notice period.\n\n"
        'Application: The clause states "either party may terminate immediately '
        'without notice if the counterparty files for bankruptcy protection."\n\n'
        "Conclusion: The clause permits immediate termination upon insolvency."
    )
    score = score_response(hallucinated, CLAUSE)
    assert score.groundedness == 0.0  # quoted text is not a substring of CLAUSE


def test_accurate_quote_scores_high_groundedness():
    quoted = (
        "Issue: Does this clause permit termination?\n\n"
        "Rule: Termination clauses must state a notice period.\n\n"
        'Application: The clause states "either party may terminate this Agreement upon '
        'thirty (30) days written notice to the other party."\n\n'
        "Conclusion: The clause permits mutual termination on notice."
    )
    score = score_response(quoted, CLAUSE)
    assert score.groundedness == 1.0


def test_truncated_response_scores_low_length():
    truncated = "Issue: Does this cla"
    score = score_response(truncated, CLAUSE)
    # doesn't even parse into sections, so structure (and everything) is 0 --
    # covered by test_missing_section_scores_zero_structure. Test the length
    # component directly against a well-formed-but-short response instead.
    from src.eval.rubric import score_length
    assert score_length(truncated, min_words=40) < 0.2


def test_contradiction_detected():
    assert score_no_contradiction(
        "The clause is enforceable under state law.",
        "Therefore, the clause is unenforceable.",
    ) == 0.0


def test_no_contradiction_when_consistent():
    assert score_no_contradiction(
        "The clause is enforceable under state law.",
        "Therefore, the clause is enforceable and binding.",
    ) == 1.0


def test_no_contradiction_when_topic_absent():
    assert score_no_contradiction(
        "The clause requires thirty days notice.",
        "The clause is a standard mutual termination provision.",
    ) == 1.0
