from src.data.build_irac import CANONICAL_PROMPT_TEMPLATE, is_substantive_clause

# Real fragments found via manual review of the generated dataset
# (data/review/sft_review_sample.csv) -- see LOG.md 2026-08-17.
REAL_FRAGMENTS = [
    "This",
    "transferable or assignable.",
    "If the Minimum Efficiency Level has",
]

REAL_SUBSTANTIVE_CLAUSES = [
    "Either party may terminate this Agreement upon thirty (30) days written notice.",
    "This Agreement shall be governed by the laws of the State of Delaware.",
]


def test_fragments_rejected():
    for fragment in REAL_FRAGMENTS:
        assert not is_substantive_clause(fragment), f"should reject: {fragment!r}"


def test_substantive_clauses_accepted():
    for clause in REAL_SUBSTANTIVE_CLAUSES:
        assert is_substantive_clause(clause), f"should accept: {clause!r}"


def test_known_residual_case_still_passes():
    """Documents a real limitation, doesn't assert a fix that doesn't
    exist: an 11-word syntactically-incomplete trailing fragment from the
    real dataset is *not* caught by the word-count floor alone (8 words is
    a conservative floor to avoid rejecting genuine short clauses). This
    test exists so a future change to the threshold doesn't silently start
    passing this case without someone noticing and updating the note."""
    fragment = "described and mutually agreed in writing as amendments to this Agreement."
    assert is_substantive_clause(fragment)


def test_prompt_template_ends_with_separator_before_completion():
    """Regression test for a real bug found during the local smoke test:
    with no separator between prompt and completion, trl's SFTTrainer (and
    later DPOTrainer) compute the completion-only loss mask by tokenizing
    the prompt alone and slicing the jointly-tokenized prompt+completion at
    that length. Every target starts with "Issue:", and the prompt used to
    end with "...section." (no trailing whitespace) -- BPE merged "." and
    "I" across the boundary into one token, which landed on the masked
    (loss=0) side, silently dropping the completion's first token from loss
    on every single training example. See LOG.md 2026-08-17."""
    formatted = CANONICAL_PROMPT_TEMPLATE.format(category="X", clause_text="Y")
    assert formatted.endswith("\n\n"), (
        "prompt must end with whitespace so BPE can't merge its last "
        "character with the completion's first character across the "
        "trl completion-mask boundary"
    )
