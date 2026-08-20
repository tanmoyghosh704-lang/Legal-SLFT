"""LLM-as-judge scoring: complements the deterministic rubric (rubric.py)
with a holistic quality judgment, since heuristic checks can't catch
subtly wrong legal reasoning or unhelpful analysis that still passes the
rubric's structural/word-overlap checks. This is the second half of the
project's dual-metric evaluation design.

Same Ollama HTTP pattern as generate_teacher_targets.py (--host/--model/
--concurrency, resumable by id) -- reuse of the teacher-generation
pipeline's already-proven approach to running LLM calls at scale, not a
new mechanism. Run locally at small scale to validate the prompt/parser
first (this project's standing rule, see LOG.md throughout); run the real
scoring pass on Kaggle-hosted Ollama the same way teacher generation was.

Scores three dimensions per response, 1-5 each:
  - groundedness: faithfulness to the clause text specifically (distinct
    from rubric.py's word-overlap proxy -- an LLM can recognize a fluent
    paraphrase that subtly invents a condition, which pure word overlap
    cannot).
  - reasoning_quality: whether Rule -> Application -> Conclusion is
    actually coherent and legally sound, not just present.
  - overall: holistic usefulness/correctness judgment.
Deliberately narrow to three dimensions with a required one-line
rationale each, rather than a single 1-10 "quality" number -- a single
score collapses exactly the kind of distinction (well-grounded but
poorly-reasoned vs. well-reasoned but ungrounded) this second metric
exists to surface that the rubric's aggregate can't.
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

JUDGE_PROMPT_TEMPLATE = """You are an expert legal analyst evaluating an AI-generated IRAC analysis of a contract clause.

Clause category: {category}
Clause text: {clause_text}

AI-generated analysis:
{response}

Score this analysis on three dimensions, each from 1 (poor) to 5 (excellent):

1. GROUNDEDNESS: Does the analysis stay strictly faithful to the clause text? A high score means every claim in the Application section is directly supported by the clause; no invented facts, conditions, or requirements not actually present in the clause text. A low score means the analysis asserts things (e.g. "requires X's consent", "applies only if Y") that are not actually stated in the clause.

2. REASONING_QUALITY: Is the legal reasoning (Rule -> Application -> Conclusion) coherent, well-structured, and legally sound? A high score means the Rule states a real, relevant legal principle, the Application correctly applies it to the clause, and the Conclusion follows logically. A low score means the reasoning is generic, contradictory, or doesn't actually connect to the clause.

3. OVERALL: Your holistic judgment of how useful and correct this analysis would be to a lawyer reviewing this clause.

Respond with EXACTLY this format, nothing else:
GROUNDEDNESS: <1-5>
REASONING_QUALITY: <1-5>
OVERALL: <1-5>
RATIONALE: <one or two sentences explaining your scores, especially if any score is below 4>"""

SCORE_RE = re.compile(
    r"GROUNDEDNESS:\s*(?P<groundedness>[1-5])\s*"
    r"REASONING_QUALITY:\s*(?P<reasoning_quality>[1-5])\s*"
    r"OVERALL:\s*(?P<overall>[1-5])\s*"
    r"RATIONALE:\s*(?P<rationale>.+)",
    re.DOTALL,
)


def parse_judge_response(raw: str) -> dict | None:
    match = SCORE_RE.search(raw.strip())
    if not match:
        return None
    d = match.groupdict()
    return {
        "groundedness": int(d["groundedness"]),
        "reasoning_quality": int(d["reasoning_quality"]),
        "overall": int(d["overall"]),
        "rationale": d["rationale"].strip(),
    }


def already_done_ids(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    ids = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def judge_ollama(prompt: str, model: str, host: str) -> str:
    import requests

    r = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["response"]


def judge_one(row: dict, model: str, host: str) -> dict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        category=row["category"], clause_text=row["clause_text"], response=row["raw_response"],
    )
    raw = judge_ollama(prompt, model, host)
    parsed = parse_judge_response(raw)
    if parsed is None:
        return {"judge_parse_ok": False, "judge_raw": raw}
    return {"judge_parse_ok": True, **parsed}


def run(input_path: str, output_path: str, model: str, host: str,
        concurrency: int = 1, ids_file: str | None = None):
    with open(input_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if ids_file:
        with open(ids_file, encoding="utf-8") as f:
            keep_ids = set(json.load(f))
        rows = [r for r in rows if r["id"] in keep_ids]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_done_ids(out_path)
    pending = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} total, {len(done)} already done, {len(pending)} pending")
    if not pending:
        return

    n_done = 0
    n_parse_failures = 0
    t_start = time.time()
    write_lock = Lock()

    with out_path.open("a", encoding="utf-8") as out_f, \
            ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(judge_one, row, model, host): row for row in pending}
        for future in as_completed(futures):
            row = futures[future]
            result = future.result()
            if not result.get("judge_parse_ok", False):
                n_parse_failures += 1
            with write_lock:
                out_f.write(json.dumps({**row, **result}, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta_min = (len(pending) - n_done) / rate / 60 if rate > 0 else float("inf")
                print(f"[{n_done}/{len(pending)}] elapsed={elapsed:.0f}s eta={eta_min:.1f}min "
                      f"parse_failures_so_far={n_parse_failures}")

    print(f"Done. parse_failures={n_parse_failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="generate.py output (has id, category, clause_text, raw_response)")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model", default="qwen2.5:7b-instruct-q4_0")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--concurrency", type=int, default=1,
                         help="Parallel Ollama requests. Use 1 locally, 8+ on Kaggle "
                              "(also set OLLAMA_NUM_PARALLEL on the server to match).")
    parser.add_argument("--ids-file", default=None,
                         help="Optional JSON list of ids to restrict judging to (see sample_judge_ids.py) "
                              "-- for judging the same common subsample across every run's gen file.")
    args = parser.parse_args()
    run(args.input_file, args.output_file, args.model, args.host, args.concurrency, args.ids_file)
