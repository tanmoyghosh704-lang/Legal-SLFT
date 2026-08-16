"""Deterministic corruption functions: turn a clean IRAC target (the DPO
"chosen" response) into a deliberately degraded "rejected" response.

Per project spec section 3.2 — five corruption types, applied
programmatically. Each corrupted sample records which corruption type
produced it, so the results can report a per-corruption-type breakdown
later (spec: "a genuinely interesting analysis almost nobody does").

Honesty note carried from the spec: these are programmatic corruptions of
already-correct answers, not real human preference judgments. They teach a
narrower, more mechanical objective (prefer well-formed/grounded IRAC over
malformed/unsupported IRAC) than genuine RLHF preference data would. State
this in the write-up; don't oversell it.
"""
import random

from src.data.build_irac import IRAC_SECTIONS, parse_irac

CORRUPTION_TYPES = [
    "drop_section",
    "reorder_sections",
    "wrong_citation",
    "unsupported_claim",
    "contradict_conclusion",
    "truncate",
]

# Fabricated citations, inserted as if quoting the clause. Written to sound
# plausible in isolation but are never a substring of any real clause text,
# so score_groundedness (rubric.py) reliably catches them.
FAKE_QUOTES = [
    "either party may terminate immediately without notice upon a change of control",
    "all disputes shall be resolved exclusively by binding arbitration in Delaware",
    "the indemnification obligations survive termination for a period of ten years",
    "liability under this section is uncapped for any breach of confidentiality",
    "this provision automatically terminates upon the filing of bankruptcy by either party",
]

# Plausible-sounding but unsupported legal claims -- invented case law /
# statutory hooks not implied by anything in the clause.
UNSUPPORTED_CLAIMS = [
    " This reading is further supported by the Delaware Court of Chancery's "
    "holding in a closely analogous dispute.",
    " Under the Uniform Commercial Code, this provision would also govern "
    "any related equipment transactions between the parties.",
    " Courts in this jurisdiction have consistently upheld the stricter of "
    "the two available readings in comparable cases.",
    " Federal preemption doctrine would override any conflicting state-law "
    "interpretation of this language.",
    " This type of provision has previously been held void as against "
    "public policy in similar commercial contexts.",
]

CONTRADICTION_TEMPLATES = [
    "Therefore, contrary to the analysis above, this clause imposes no "
    "enforceable obligation on either party and may be disregarded.",
    "Accordingly, despite the foregoing, the clause should be read as "
    "granting the opposite right from the one just described.",
    "In conclusion, the clause is unenforceable as drafted, notwithstanding "
    "the application above showing it meets the applicable standard.",
]


def _text(sections: dict, order: list) -> str:
    return "\n\n".join(f"{name}: {sections[name]}" for name in order if sections.get(name))


def corrupt_drop_section(sections: dict, rng: random.Random) -> str:
    victim = rng.choice(IRAC_SECTIONS)
    remaining = [s for s in IRAC_SECTIONS if s != victim]
    return _text(sections, remaining)


def corrupt_reorder_sections(sections: dict, rng: random.Random) -> str:
    order = list(IRAC_SECTIONS)
    while order == IRAC_SECTIONS:
        rng.shuffle(order)
    return _text(sections, order)


def corrupt_wrong_citation(sections: dict, rng: random.Random) -> str:
    fake = rng.choice(FAKE_QUOTES)
    corrupted = dict(sections)
    corrupted["Application"] = (
        corrupted["Application"].rstrip(". ") + f'. The clause further states that "{fake}."'
    )
    return _text(corrupted, IRAC_SECTIONS)


def corrupt_unsupported_claim(sections: dict, rng: random.Random) -> str:
    claim = rng.choice(UNSUPPORTED_CLAIMS)
    corrupted = dict(sections)
    corrupted["Rule"] = corrupted["Rule"].rstrip(". ") + "." + claim
    return _text(corrupted, IRAC_SECTIONS)


def corrupt_contradict_conclusion(sections: dict, rng: random.Random) -> str:
    corrupted = dict(sections)
    corrupted["Conclusion"] = rng.choice(CONTRADICTION_TEMPLATES)
    return _text(corrupted, IRAC_SECTIONS)


def corrupt_truncate(sections: dict, rng: random.Random) -> str:
    full = _text(sections, IRAC_SECTIONS)
    # Cut somewhere in the back half so the truncation is mid-Application or
    # mid-Conclusion, not just a missing Conclusion (that's drop_section's job).
    cut_point = rng.randint(int(len(full) * 0.5), int(len(full) * 0.85))
    return full[:cut_point].rstrip()


CORRUPTORS = {
    "drop_section": corrupt_drop_section,
    "reorder_sections": corrupt_reorder_sections,
    "wrong_citation": corrupt_wrong_citation,
    "unsupported_claim": corrupt_unsupported_claim,
    "contradict_conclusion": corrupt_contradict_conclusion,
    "truncate": corrupt_truncate,
}


def make_rejected(chosen_text: str, rng: random.Random) -> tuple[str, str]:
    """Return (rejected_text, corruption_type) for a clean IRAC `chosen_text`.

    Raises ValueError if `chosen_text` doesn't parse -- corruption assumes a
    well-formed starting point (this should only ever be called on
    already-validated `parse_ok=True` teacher targets).
    """
    sections = parse_irac(chosen_text)
    if sections is None:
        raise ValueError("make_rejected requires a well-formed IRAC chosen_text")
    corruption_type = rng.choice(CORRUPTION_TYPES)
    rejected_text = CORRUPTORS[corruption_type](sections, rng)
    return rejected_text, corruption_type
