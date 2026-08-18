"""Contract-level train/valid/test splitting.

Splits by contract, never by clause — clauses from the same contract
appearing in both train and test would be leakage (the model could learn
contract-specific phrasing rather than generalizing). Deterministic given
a seed, so every run trains on identical splits (spec section 3.3).
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def _company_key(contract_title: str) -> str:
    """Leading company-name portion of a CUAD contract title, normalized
    (uppercased, punctuation/whitespace stripped) for comparison. CUAD
    titles are filenames, and the same real document sometimes appears
    twice with different filename conventions for the date/accession
    portion (e.g. "ARMSTRONGFLOORING,INC_01_07_2019-EX-10.2-..." vs
    "ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_..."), but the
    company-name prefix is stable across both."""
    prefix = re.match(r"^[A-Za-z][A-Za-z,.\s]*", contract_title)
    key = prefix.group(0) if prefix else contract_title
    return re.sub(r"[^A-Z0-9]", "", key.upper())


def find_duplicate_documents(records: list, contract_key: str = "contract",
                              clause_key: str = "clause_text", min_shared_clauses: int = 3) -> dict:
    """Detect contract-title strings that are actually the same underlying
    document filed twice under different titles, and return a mapping from
    each duplicate title to one canonical title to use for splitting.

    Two signals combined, deliberately conservative (both required): (1)
    the two titles share at least `min_shared_clauses` identical
    clause_text values, and (2) their normalized company-name prefixes
    match. Signal (1) alone isn't enough — found on the real dataset that
    two genuinely different companies (different real contracts) shared 5
    identical clauses purely from reusing the same boilerplate agreement
    template, which would have wrongly merged them. See LOG.md 2026-08-17.
    """
    by_clause = defaultdict(set)
    for r in records:
        by_clause[r[clause_key]].add(r[contract_key])

    shared_counts = defaultdict(int)
    for contracts in by_clause.values():
        if len(contracts) > 1:
            names = sorted(contracts)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    shared_counts[(names[i], names[j])] += 1

    # Union-find over confirmed duplicate pairs, so a chain of 3+ near-
    # duplicate titles for the same document (not observed yet, but
    # possible) collapses to one canonical title rather than only merging
    # pairwise.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = min(ra, rb)  # deterministic: alphabetically-first wins as root
            parent[rb] = min(ra, rb)

    for (a, b), count in shared_counts.items():
        if count >= min_shared_clauses and _company_key(a) == _company_key(b):
            union(a, b)

    return {name: find(name) for name in parent}


def assign_splits(contracts: list, ratios: dict, seed: int) -> dict:
    """Return {contract_name: split_name}, one entry per unique contract.

    Shuffles once, deterministically, then cuts by the given ratios in a
    fixed order (train, valid, test) so results are reproducible across
    runs and independent of dict/set iteration order.
    """
    names = sorted(set(contracts))  # sorted first: dedupe with a stable, order-independent base
    rng = random.Random(seed)
    rng.shuffle(names)

    n = len(names)
    n_train = round(n * ratios["train"])
    n_valid = round(n * ratios["valid"])
    # test gets the remainder, so rounding never drops or duplicates a contract
    n_test = n - n_train - n_valid

    assignment = {}
    for name in names[:n_train]:
        assignment[name] = "train"
    for name in names[n_train:n_train + n_valid]:
        assignment[name] = "valid"
    for name in names[n_train + n_valid:n_train + n_valid + n_test]:
        assignment[name] = "test"
    return assignment


def split_records(records: list, ratios: dict, seed: int, contract_key: str = "contract",
                   dedupe_documents: bool = True) -> dict:
    """Partition records (SFT rows, DPO pairs, anything with a contract_key
    field) into {"train": [...], "valid": [...], "test": [...]} using a
    single contract->split assignment, so the same contract always lands
    in the same split regardless of which record list it's applied to
    (SFT and DPO data stay consistent with each other).

    dedupe_documents: canonicalize contract titles that are the same
    underlying document filed under two different names (see
    find_duplicate_documents) before assigning splits, so a duplicate
    filing can't land in a different split than its twin and leak.
    """
    canonical = find_duplicate_documents(records, contract_key) if dedupe_documents else {}
    contracts = [canonical.get(r[contract_key], r[contract_key]) for r in records]
    assignment = assign_splits(contracts, ratios, seed)

    out = {"train": [], "valid": [], "test": []}
    for r in records:
        canonical_name = canonical.get(r[contract_key], r[contract_key])
        out[assignment[canonical_name]].append(r)
    return out


def write_splits(records: list, output_dir: str, ratios: dict, seed: int,
                  contract_key: str = "contract", basename: str = "data"):
    splits = split_records(records, ratios, seed, contract_key)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = out_dir / f"{basename}_{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_contracts = len({r[contract_key] for r in rows})
        print(f"{name}: {len(rows)} rows, {n_contracts} contracts -> {path}")
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with a 'contract' field per row")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--basename", default="sft")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    ratios = {"train": args.train_ratio, "valid": args.valid_ratio, "test": args.test_ratio}
    write_splits(records, args.output_dir, ratios, args.seed, basename=args.basename)
