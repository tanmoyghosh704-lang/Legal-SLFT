from src.eval.llm_judge import parse_judge_response

WELL_FORMED = """GROUNDEDNESS: 5
REASONING_QUALITY: 4
OVERALL: 4
RATIONALE: The analysis stays faithful to the clause text and the reasoning is sound, though the Rule section is somewhat generic."""


def test_parses_well_formed_response():
    result = parse_judge_response(WELL_FORMED)
    assert result == {
        "groundedness": 5,
        "reasoning_quality": 4,
        "overall": 4,
        "rationale": "The analysis stays faithful to the clause text and the reasoning is sound, "
                     "though the Rule section is somewhat generic.",
    }


def test_parses_response_with_leading_preamble():
    """Judge models sometimes preface the required format with a stray
    sentence despite the prompt's "EXACTLY this format, nothing else"
    instruction -- the regex searches, not anchors, so this should still
    parse."""
    raw = "Here is my evaluation:\n\n" + WELL_FORMED
    result = parse_judge_response(raw)
    assert result is not None
    assert result["overall"] == 4


def test_returns_none_for_malformed_response():
    assert parse_judge_response("I think this analysis is pretty good overall.") is None


def test_returns_none_for_out_of_range_score():
    malformed = WELL_FORMED.replace("OVERALL: 4", "OVERALL: 7")
    assert parse_judge_response(malformed) is None


def test_rationale_captures_multiline_text():
    raw = """GROUNDEDNESS: 3
REASONING_QUALITY: 2
OVERALL: 2
RATIONALE: The Application section invents a consent requirement not present
in the clause text, and the Conclusion contradicts the Application's own
finding."""
    result = parse_judge_response(raw)
    assert result["groundedness"] == 3
    assert "invents a consent requirement" in result["rationale"]
