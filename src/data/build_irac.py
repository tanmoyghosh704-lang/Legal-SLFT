"""CUAD -> IRAC task definition: canonical prompt/target template, clause
filtering, and the teacher-generation queue.

This module is the single source of truth for the prompt template used
everywhere else in the project (SFT, DPO prompts, baseline zero-shot eval,
serving). Every other script imports CANONICAL_PROMPT_TEMPLATE and
parse_irac from here rather than redefining them, per the project rule
that template drift between runs invalidates the cross-run comparison.
"""
import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

# CUAD categories that are pure extraction ("what is the document name?")
# rather than substantive legal questions. IRAC has no "Issue" to raise for
# these, so they're excluded from the alignment task. See LOG.md
# (2026-08-16, Phase 1) for the reasoning.
EXCLUDED_CATEGORIES = {
    "Parties",
    "Document Name",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
}

IRAC_SECTIONS = ["Issue", "Rule", "Application", "Conclusion"]

CANONICAL_PROMPT_TEMPLATE = """You are a legal analyst reviewing a contract clause.

Clause category: {category}
Clause text: {clause_text}

Analyze this clause using IRAC format. Respond with exactly four sections, \
each starting on its own line with the section name followed by a colon, \
in this order: Issue, Rule, Application, Conclusion. Do not include any \
text before Issue: or after the Conclusion section."""

# Instruction to the teacher model (Kaggle-hosted Qwen2.5-7B-Instruct or
# local Ollama for smoke-testing) generating SFT targets. Adds groundedness
# constraints that the canonical student-facing prompt doesn't need to
# spell out, since the teacher's job is specifically to produce clean
# training targets, not just any answer.
TEACHER_GENERATION_TEMPLATE = CANONICAL_PROMPT_TEMPLATE + """

Additional instructions for this analysis:
- Base the Rule section on general, well-established contract law \
principles relevant to this clause category — do not invent case citations.
- The Application section must refer only to language actually present in \
the clause text above. Do not assume facts not stated in the clause.
- Keep each section to 2-4 sentences.
- The Conclusion must follow logically from the Application — do not \
contradict it."""

IRAC_PARSE_RE = re.compile(
    r"Issue:\s*(?P<Issue>.*?)\s*"
    r"Rule:\s*(?P<Rule>.*?)\s*"
    r"Application:\s*(?P<Application>.*?)\s*"
    r"Conclusion:\s*(?P<Conclusion>.*?)\s*$",
    re.DOTALL,
)


def parse_irac(text: str) -> dict | None:
    """Parse a model response into {Issue, Rule, Application, Conclusion}.

    Returns None if the four sections aren't present in order — used both
    to validate teacher-generated targets and later by the deterministic
    rubric to score student outputs.
    """
    match = IRAC_PARSE_RE.search(text.strip())
    if not match:
        return None
    sections = {k: v.strip() for k, v in match.groupdict().items()}
    if any(len(v) == 0 for v in sections.values()):
        return None
    return sections


def format_target(sections: dict) -> str:
    """Canonical serialization of parsed IRAC sections back to text.

    Used to normalize teacher output (strip extra whitespace/preamble)
    before saving as a training target, so all targets have identical
    formatting regardless of minor teacher generation quirks.
    """
    return "\n\n".join(f"{name}: {sections[name]}" for name in IRAC_SECTIONS)


CATEGORY_RE = re.compile(r'related to "([^"]+)"')


def load_cuad_filtered():
    """Load the canonical theatticusproject/cuad-qa via HF's auto-generated
    parquet branch (refs/convert/parquet), rather than the deprecated
    loading-script format the repo ships by default. See LOG.md 2026-08-16
    for the original script-loading failure, and 2026-08-17 for switching
    off the chenghao/cuad_qa community mirror onto this canonical source.

    This revision's `question` field is a full instruction sentence
    ('Highlight the parts...related to "Governing Law"...'), not a bare
    category label, so the category is extracted via regex. It also
    includes negative (unanswerable) rows the community mirror had already
    stripped, so the empty-answer filter below is load-bearing here in a
    way it wasn't before."""
    ds = load_dataset("theatticusproject/cuad-qa", split="train", revision="refs/convert/parquet")
    rows = []
    for ex in ds:
        if not ex["answers"]["text"]:
            continue
        match = CATEGORY_RE.search(ex["question"])
        if not match:
            continue
        category = match.group(1)
        if category in EXCLUDED_CATEGORIES:
            continue
        rows.append(
            {
                "id": ex["id"],
                "contract": ex["title"],
                "category": category,
                "clause_text": ex["answers"]["text"][0].strip(),
            }
        )
    return rows


def build_queue(output_path: str, cap_per_category: int | None = None, seed: int = 42):
    """Build the teacher-generation queue: one row per (contract, category)
    pair with the canonical prompt and the teacher-generation prompt
    already filled in, ready to hand to the Kaggle generation notebook.

    cap_per_category: if set, stratified-sample at most this many rows per
    category (deterministic given `seed`) instead of using every eligible
    row. Used to bound generation time; None means use everything.
    """
    import random

    rows = load_cuad_filtered()
    by_category: dict[str, list] = {}
    for r in rows:
        by_category.setdefault(r["category"], []).append(r)

    rng = random.Random(seed)
    selected = []
    for category, items in by_category.items():
        if cap_per_category is not None and len(items) > cap_per_category:
            items = rng.sample(items, cap_per_category)
        selected.extend(items)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in selected:
            record = {
                **r,
                "prompt": CANONICAL_PROMPT_TEMPLATE.format(
                    category=r["category"], clause_text=r["clause_text"]
                ),
                "teacher_prompt": TEACHER_GENERATION_TEMPLATE.format(
                    category=r["category"], clause_text=r["clause_text"]
                ),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} rows across {len(by_category)} categories to {out}")
    return len(selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/sft_generation_queue.jsonl")
    parser.add_argument("--cap-per-category", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_queue(args.output, args.cap_per_category, args.seed)
