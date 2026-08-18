"""Deterministic, non-LLM scoring rubric for IRAC outputs.

Built before any training run, per the project's own build order — this is
what "good" means, and it has to be fixed before it's used to judge model
outputs, not adjusted after seeing results. Component scores are reported
separately as well as an aggregate: a single blended number hides which
capability actually improved (spec section 5.1).
"""
import re
from dataclasses import asdict, dataclass

from src.data.build_irac import parse_irac

STOPWORDS = {
    "this", "that", "these", "those", "with", "from", "have", "shall",
    "will", "would", "could", "should", "which", "their", "there", "under",
    "into", "such", "than", "then", "also", "been", "being", "were", "each",
    "other", "some", "more", "most", "only", "both", "same", "very", "does",
    "upon", "when", "where", "what", "while", "must", "party", "parties",
    "clause", "provision", "agreement", "contract",
}

QUOTE_RE = re.compile(r'"([^"]{4,})"')

# (assert_positive, assert_negative) term pairs. If Application asserts one
# polarity and Conclusion asserts the other for the same underlying claim,
# that's a direct contradiction. Heuristic, not an NLI model — see spec
# section 5.1, "no contradiction markers ... (heuristic)".
#
# ("applies", "does not apply") and ("renews", "does not renew") were
# dropped after running this on the real 7,620-row teacher-generated
# dataset: both are scope/conditional pairs that legitimate IRAC analysis
# routinely uses together in one consistent claim ("applies to X ... does
# not extend to Y outside X") -- a real false positive, not a contradiction.
# See LOG.md 2026-08-17.
CONTRADICTION_PAIRS = [
    ("enforceable", "unenforceable"),
    ("enforceable", "not enforceable"),
    ("valid", "invalid"),
    ("permitted", "prohibited"),
    ("permitted", "not permitted"),
    ("obligated", "not obligated"),
    ("liable", "not liable"),
    ("required", "not required"),
    ("ambiguous", "unambiguous"),
]


@dataclass
class RubricScore:
    structure_ok: float
    groundedness: float
    length_ok: float
    no_contradiction: float

    @property
    def aggregate(self) -> float:
        return (self.structure_ok + self.groundedness + self.length_ok + self.no_contradiction) / 4

    def to_dict(self) -> dict:
        d = asdict(self)
        d["aggregate"] = self.aggregate
        return d


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-zA-Z]{4,}", text.lower()) if w not in STOPWORDS}


def score_structure(sections: dict | None) -> float:
    """All four IRAC sections present, in order, non-empty. `parse_irac`
    already enforces order (it's a single anchored regex) and non-emptiness,
    so this is just "did it parse at all"."""
    return 1.0 if sections is not None else 0.0


# The teacher prompt (build_irac.py TEACHER_GENERATION_TEMPLATE) explicitly
# instructs "do not invent case citations" for the Rule section. A Rule
# invoking a specific court, named statute, or claimed precedent is
# precisely the failure mode that instruction exists to prevent -- a
# legitimate "general, well-established contract law principle" doesn't
# need to name one. Added after test_corrupt.py's
# test_rejected_scores_lower_than_chosen_on_rubric caught unsupported_claim
# corruptions (inserted into Rule) being invisible to groundedness, since
# that check only ever looked at Application. See LOG.md 2026-08-17.
# Note: bare "precedent" is deliberately excluded. "Condition precedent"
# (a condition that must occur before a duty arises) is a completely
# standard, legitimate contract-law term of art with nothing to do with
# case law -- flagging it as invented authority was a real false positive
# found on the real 7,620-row dataset (Rule text: "...requires the
# fulfillment of a specific condition precedent before the renewal right
# can be exercised"). Only qualified phrases that actually mean case-law
# precedent are listed here. See LOG.md 2026-08-17.
AUTHORITY_RED_FLAGS = [
    "court of chancery", "uniform commercial code", "federal preemption",
    "held that", "ruling", "case precedent", "legal precedent",
    "binding precedent", "sets a precedent", "sets precedent",
    "as a precedent", "case law", "prior litigation",
    "consistently upheld", "void as against public policy",
]


def score_groundedness(application_text: str, rule_text: str, clause_text: str) -> float:
    """Hallucination guard — the most important single check per spec.

    Two independent signals, combined with min() (either failing sinks the
    score, since this check is deliberately conservative):

    1. Application vs. clause text. Strong check: any quoted span in
       Application must be an actual substring of the source clause text.
       Falls back to a word-overlap heuristic when the model paraphrases
       instead of quoting directly — the prompt template asks the model to
       stay grounded in the clause but doesn't force literal quotation, so
       this fallback is a real, documented limitation, not an oversight: a
       well-grounded paraphrase and a fluent hallucination can produce
       similar word overlap. The quote-based path is exact; the fallback
       is a weaker proxy.
    2. Rule vs. invented legal authority — see AUTHORITY_RED_FLAGS above.
    """
    clause_norm = _normalize(clause_text)
    quotes = QUOTE_RE.findall(application_text)
    if quotes:
        clause_score = sum(1 for q in quotes if _normalize(q) in clause_norm) / len(quotes)
    else:
        app_words = _content_words(application_text)
        clause_words = _content_words(clause_text)
        if not app_words:
            clause_score = 0.0
        else:
            overlap = app_words & clause_words
            # A grounded paraphrase of a short clause typically reuses a
            # meaningful fraction of its content words (parties, defined
            # terms, key verbs, numbers). 30% is a deliberately loose bar
            # for the fallback proxy -- it should catch wholesale
            # invention, not penalize normal paraphrase.
            clause_score = min(1.0, len(overlap) / max(1, round(len(app_words) * 0.3)))

    rule_lower = rule_text.lower()
    authority_score = 0.0 if any(flag in rule_lower for flag in AUTHORITY_RED_FLAGS) else 1.0

    return min(clause_score, authority_score)


def score_length(full_text: str, min_words: int = 40, max_words: int = 400) -> float:
    """Catches degenerate/truncated output (too short) and rambling/
    repetition-loop output (too long). Graded, not a hard cutoff, so a
    response 10% under the floor scores 0.9 rather than 0."""
    n = len(re.findall(r"[a-zA-Z]+", full_text))
    if min_words <= n <= max_words:
        return 1.0
    if n < min_words:
        return max(0.0, n / min_words)
    return max(0.0, max_words / n)


# Explicit self-reversal cues -- a Conclusion that flags its own departure
# from the Application ("contrary to", "notwithstanding the application")
# is a contradiction regardless of whether the specific vocabulary matches
# a CONTRADICTION_PAIRS entry. Added after test_corrupt.py's
# test_rejected_scores_lower_than_chosen_on_rubric caught the term-pair
# check alone missing a corrupted Conclusion that reversed the Application
# using vocabulary neither pair covered ("contrary to the analysis above,
# this clause imposes no enforceable obligation") -- see LOG.md 2026-08-17.
REVERSAL_CUES = [
    "contrary to the", "notwithstanding the application", "notwithstanding the analysis",
    "despite the foregoing", "despite the application", "opposite right",
    "opposite conclusion", "should be disregarded",
]

# A sentence containing one of these is describing a risk, condition, or
# recommendation ("may render it unenforceable", "to ensure it is
# enforceable") rather than asserting the term as a present fact. Without
# this guard, real teacher output on the full 7,620-row dataset false-
# positived on exactly this pattern: Application flags a defect that
# *could* cause a problem, Conclusion recommends a fix to *ensure* the
# good outcome -- consistent reasoning, not a contradiction. See LOG.md
# 2026-08-17.
HEDGE_WORDS = [
    "may ", "may render", "might ", "could ", "potentially", "to ensure",
    "in order to", "need to", "needs to", "would need", "unless", "if the",
    "if it", "should be", "clarif",
]

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _asserted(text_lower: str, term: str) -> bool:
    """True if `term` appears, as a whole word/phrase, in some sentence of
    text_lower that isn't hedged -- a hedged sentence describes a risk/
    condition/recommendation, not a present-tense assertion of the term.

    Word-boundary match, not plain substring: "enforceable" is a literal
    substring of "unenforceable", so a plain `in` check would count
    "unenforceable" as asserting *both* "enforceable" and "unenforceable"
    simultaneously, defeating the pair's mutual exclusivity.
    """
    pattern = re.compile(rf"\b{re.escape(term)}\b")
    for sentence in SENTENCE_SPLIT_RE.split(text_lower):
        if pattern.search(sentence) and not any(h in sentence for h in HEDGE_WORDS):
            return True
    return False


# A Conclusion sentence that confidently asserts something and then
# immediately hedges the same point ("...effectively caps X for all
# Participants combined, but it is unclear if this applies to each
# Participant individually or collectively") is self-undermining
# regardless of subject matter -- IRAC's Conclusion is supposed to
# conclude, not equivocate in the same breath it just asserted something.
# Found via manual review of the real dataset (a case the term-pair check
# structurally can't catch -- no shared vocabulary between the confident
# claim and the hedge). Deliberately narrow: requires an uncertainty
# marker *and* a contrastive conjunction in the same sentence, so a
# Conclusion that honestly states "it is unclear whether X" as its actual
# finding (no preceding contradicted claim) isn't penalized -- see LOG.md
# 2026-08-17.
UNCERTAINTY_MARKERS = [
    "unclear if", "unclear whether", "not clear whether", "not clear if",
    "uncertain whether", "ambiguous whether", "it is unclear",
]
CONTRAST_CUES = [" but ", ", but", " however", "; however", ", yet "]


def _self_hedging_conclusion(conclusion_lower: str) -> bool:
    for sentence in SENTENCE_SPLIT_RE.split(conclusion_lower):
        if any(m in sentence for m in UNCERTAINTY_MARKERS) and \
           any(c in sentence for c in CONTRAST_CUES):
            return True
    return False


def score_no_contradiction(application_text: str, conclusion_text: str) -> float:
    application = application_text.lower()
    conclusion = conclusion_text.lower()

    if any(cue in conclusion for cue in REVERSAL_CUES):
        return 0.0

    if _self_hedging_conclusion(conclusion):
        return 0.0

    for pos, neg in CONTRADICTION_PAIRS:
        app_pos, app_neg = _asserted(application, pos), _asserted(application, neg)
        concl_pos, concl_neg = _asserted(conclusion, pos), _asserted(conclusion, neg)
        if (app_pos and not app_neg and concl_neg and not concl_pos) or \
           (app_neg and not app_pos and concl_pos and not concl_neg):
            return 0.0
    return 1.0


def score_response(raw_text: str, clause_text: str) -> RubricScore:
    """Score a single model response against its source clause.

    If the response doesn't parse into four IRAC sections, every component
    scores 0 — a malformed response can't be meaningfully grounded,
    length-checked, or contradiction-checked in a way that's comparable to
    a well-formed one.
    """
    sections = parse_irac(raw_text)
    if sections is None:
        return RubricScore(structure_ok=0.0, groundedness=0.0, length_ok=0.0, no_contradiction=0.0)
    return RubricScore(
        structure_ok=1.0,
        groundedness=score_groundedness(sections["Application"], sections["Rule"], clause_text),
        length_ok=score_length(raw_text),
        no_contradiction=score_no_contradiction(sections["Application"], sections["Conclusion"]),
    )
