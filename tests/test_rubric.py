from src.eval.rubric import score_groundedness, score_no_contradiction, score_response

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


def test_no_false_positive_when_conclusion_only_uses_negative_form():
    """Regression test: "enforceable" is a literal substring of
    "unenforceable". Application asserting "enforceable" must not be
    treated as also asserting "unenforceable" just because that word
    contains it -- caught on the real 7,620-row teacher-generated dataset,
    see LOG.md 2026-08-17."""
    assert score_no_contradiction(
        "The clause is enforceable under state law.",
        "Therefore, the clause is unenforceable.",
    ) == 0.0  # this IS a real contradiction -- opposite pair, both unhedged


def test_no_false_positive_on_hedged_risk_then_remedy():
    """Application flags a defect that *may* cause a problem; Conclusion
    recommends a fix *to ensure* the good outcome. Consistent reasoning
    (risk identified, remedy proposed), not a contradiction -- this exact
    pattern false-positived on the real dataset before the hedge-word
    guard was added."""
    assert score_no_contradiction(
        "The lack of these details may render the clause overly broad and potentially unenforceable.",
        "The clause should be clarified to ensure it is enforceable.",
    ) == 1.0


def test_no_false_positive_on_scope_limited_application():
    """"Applies to X ... does not extend to Y outside X" and "does not
    apply outside X" are the same claim about limited scope stated twice,
    not a contradiction. (applies/does not apply was dropped entirely from
    CONTRADICTION_PAIRS after this exact false positive on the real
    dataset -- kept as a regression test in case it's ever re-added.)"""
    assert score_no_contradiction(
        "The clause applies to interactions within Covered Regions and does not extend to "
        "solicitation activities outside these regions.",
        "The obligation is geographically limited and does not apply to solicitation "
        "activities outside these regions.",
    ) == 1.0


def test_condition_precedent_is_not_flagged_as_invented_authority():
    """"Condition precedent" is a standard contract-law term of art (a
    condition that must occur before a duty arises) -- nothing to do with
    case-law precedent. A bare "precedent" substring check flagged this as
    invented legal authority on the real 7,620-row dataset; fixed by
    requiring qualified case-law phrases instead. See LOG.md 2026-08-17."""
    application = "The Publishers must exercise their renewal right each year to remain eligible."
    rule = (
        "A conditional right to renew typically requires the fulfillment of a specific "
        "condition precedent before the renewal right can be exercised."
    )
    clause = "The Publishers exercise their renewal right each year to remain eligible."
    assert score_groundedness(application, rule, clause) > 0.0


def test_genuine_case_law_citation_still_flagged():
    application = "The clause requires notice before termination."
    rule = "This reading is supported by binding precedent in a closely analogous dispute."
    clause = "The clause requires notice before termination."
    assert score_groundedness(application, rule, clause) == 0.0


def test_self_hedging_conclusion_flagged():
    """Real case from manual review of the full dataset: Conclusion
    confidently asserts something, then hedges the same point in the same
    sentence -- self-undermining, and invisible to the term-pair check
    (no shared vocabulary). See LOG.md 2026-08-17."""
    application = (
        "The clause limits the number of audits to one for any fiscal year. It does not "
        'specify whether this applies to all Participants collectively or to each '
        "Participant individually."
    )
    conclusion = (
        "The clause effectively caps the total number of audits to one per fiscal year "
        "for all Participants combined, but it is unclear if this applies to each "
        "Participant individually or collectively."
    )
    assert score_no_contradiction(application, conclusion) == 0.0


def test_honest_uncertainty_without_prior_claim_not_flagged():
    """A Conclusion that honestly states its finding is uncertain, without
    first confidently asserting the opposite, is a legitimate conclusion
    -- not every mention of "unclear" is self-contradictory."""
    application = "The clause does not specify a notice period for termination."
    conclusion = "It is unclear whether notice is required at all under this clause."
    assert score_no_contradiction(application, conclusion) == 1.0
