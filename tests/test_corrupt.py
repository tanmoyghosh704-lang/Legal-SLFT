import random

import pytest

from src.data.build_irac import IRAC_SECTIONS, parse_irac
from src.data.corrupt import CORRUPTION_TYPES, CORRUPTORS, make_rejected
from src.eval.rubric import score_response

CLAUSE = (
    "Either party may terminate this Agreement upon thirty (30) days "
    "written notice to the other party."
)

CHOSEN = (
    "Issue: Does this clause permit unilateral termination, and under what conditions?\n\n"
    "Rule: A termination-for-convenience clause is enforceable if it clearly states "
    "the notice period and applies equally to both parties.\n\n"
    "Application: The clause permits either party to terminate upon thirty days written "
    "notice to the other party, without requiring cause. The right is mutual.\n\n"
    "Conclusion: The clause is a valid mutual termination-for-convenience provision "
    "conditioned on thirty days' written notice."
)


def test_make_rejected_requires_well_formed_input():
    with pytest.raises(ValueError):
        make_rejected("Issue: incomplete, no other sections", random.Random(0))


@pytest.mark.parametrize("corruption_type", CORRUPTION_TYPES)
def test_each_corruption_type_changes_the_text(corruption_type):
    sections = parse_irac(CHOSEN)
    rejected = CORRUPTORS[corruption_type](sections, random.Random(1))
    assert rejected != CHOSEN
    assert len(rejected) > 0


def test_drop_section_actually_removes_a_section():
    sections = parse_irac(CHOSEN)
    rejected = CORRUPTORS["drop_section"](sections, random.Random(2))
    present = [name for name in IRAC_SECTIONS if f"{name}:" in rejected]
    assert len(present) == 3


def test_reorder_sections_changes_order():
    sections = parse_irac(CHOSEN)
    rejected = CORRUPTORS["reorder_sections"](sections, random.Random(3))
    positions = [rejected.index(f"{name}:") for name in IRAC_SECTIONS]
    assert positions != sorted(positions)


def test_wrong_citation_inserts_unfounded_quote():
    sections = parse_irac(CHOSEN)
    rejected = CORRUPTORS["wrong_citation"](sections, random.Random(4))
    assert '"' in rejected


def test_truncate_is_shorter():
    sections = parse_irac(CHOSEN)
    full = "\n\n".join(f"{name}: {sections[name]}" for name in IRAC_SECTIONS)
    rejected = CORRUPTORS["truncate"](sections, random.Random(5))
    assert len(rejected) < len(full)


def test_deterministic_given_same_seed():
    rejected_a, type_a = make_rejected(CHOSEN, random.Random(42))
    rejected_b, type_b = make_rejected(CHOSEN, random.Random(42))
    assert rejected_a == rejected_b
    assert type_a == type_b


def test_rejected_scores_lower_than_chosen_on_rubric():
    """The integration check that actually matters: every corruption type
    should make the rubric score go down relative to the chosen response it
    was derived from, since that's the entire premise of the DPO pair."""
    chosen_score = score_response(CHOSEN, CLAUSE).aggregate
    for corruption_type in CORRUPTION_TYPES:
        rng = random.Random(hash(corruption_type) % 1000)
        rejected_text, actual_type = make_rejected(CHOSEN, rng)
        rejected_score = score_response(rejected_text, CLAUSE).aggregate
        assert rejected_score < chosen_score, (
            f"{actual_type} produced a rejected sample scoring "
            f"{rejected_score:.2f} >= chosen's {chosen_score:.2f}"
        )


def test_corruption_type_distribution_over_many_seeds():
    """Sanity check that all six corruption types actually get used when
    sampled repeatedly, not just the first one alphabetically/first-in-list."""
    seen = set()
    for i in range(200):
        _, corruption_type = make_rejected(CHOSEN, random.Random(i))
        seen.add(corruption_type)
    assert seen == set(CORRUPTION_TYPES)
