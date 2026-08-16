"""Contract-level train/valid/test splitting.

Splits by contract, never by clause — clauses from the same contract
appearing in both train and test would be leakage (the model could learn
contract-specific phrasing rather than generalizing). Deterministic given
a seed, so every run trains on identical splits (spec section 3.3).
"""
import argparse
import json
import random
from pathlib import Path


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


def split_records(records: list, ratios: dict, seed: int, contract_key: str = "contract") -> dict:
    """Partition records (SFT rows, DPO pairs, anything with a contract_key
    field) into {"train": [...], "valid": [...], "test": [...]} using a
    single contract->split assignment, so the same contract always lands
    in the same split regardless of which record list it's applied to
    (SFT and DPO data stay consistent with each other)."""
    contracts = [r[contract_key] for r in records]
    assignment = assign_splits(contracts, ratios, seed)

    out = {"train": [], "valid": [], "test": []}
    for r in records:
        out[assignment[r[contract_key]]].append(r)
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
