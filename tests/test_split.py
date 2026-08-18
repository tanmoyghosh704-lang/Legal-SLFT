from src.data.split import assign_splits, find_duplicate_documents, split_records

RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}


def make_records(n_contracts: int, clauses_per_contract: int = 3) -> list:
    records = []
    for c in range(n_contracts):
        for i in range(clauses_per_contract):
            records.append({"contract": f"contract_{c}", "clause_text": f"clause {c}-{i}"})
    return records


def test_every_contract_assigned_exactly_once():
    contracts = [f"contract_{i}" for i in range(100)] * 3  # simulate repeats (multiple clauses/contract)
    assignment = assign_splits(contracts, RATIOS, seed=1)
    assert set(assignment.keys()) == {f"contract_{i}" for i in range(100)}
    assert len(assignment) == 100


def test_split_ratios_approximately_correct():
    contracts = [f"contract_{i}" for i in range(1000)]
    assignment = assign_splits(contracts, RATIOS, seed=1)
    counts = {"train": 0, "valid": 0, "test": 0}
    for split in assignment.values():
        counts[split] += 1
    assert 780 <= counts["train"] <= 820
    assert 80 <= counts["valid"] <= 120
    assert 80 <= counts["test"] <= 120
    assert sum(counts.values()) == 1000


def test_deterministic_given_same_seed():
    contracts = [f"contract_{i}" for i in range(50)]
    a = assign_splits(contracts, RATIOS, seed=7)
    b = assign_splits(contracts, RATIOS, seed=7)
    assert a == b


def test_different_seeds_give_different_splits():
    contracts = [f"contract_{i}" for i in range(50)]
    a = assign_splits(contracts, RATIOS, seed=1)
    b = assign_splits(contracts, RATIOS, seed=2)
    assert a != b


def test_no_contract_leaks_across_splits():
    """The check that actually matters: a contract's clauses must never be
    split across train/valid/test, since that's leakage."""
    records = make_records(n_contracts=200, clauses_per_contract=5)
    result = split_records(records, RATIOS, seed=3)

    contract_to_splits = {}
    for split_name, rows in result.items():
        for row in rows:
            contract_to_splits.setdefault(row["contract"], set()).add(split_name)

    leaked = {c: s for c, s in contract_to_splits.items() if len(s) > 1}
    assert leaked == {}, f"contracts appearing in multiple splits: {leaked}"


def test_all_records_preserved():
    records = make_records(n_contracts=50, clauses_per_contract=4)
    result = split_records(records, RATIOS, seed=5)
    total = sum(len(rows) for rows in result.values())
    assert total == len(records)


def test_small_n_no_split_starves_via_rounding():
    """With few contracts, rounding shouldn't silently zero out a split
    when the ratio would round to <1 contract for a nonzero ratio -- not
    asserting nonzero here (it's mathematically possible with too few
    contracts), just that the three splits sum exactly to n with no
    contract dropped or double-counted."""
    contracts = [f"contract_{i}" for i in range(7)]
    assignment = assign_splits(contracts, RATIOS, seed=1)
    assert len(assignment) == 7
    assert set(assignment.values()) <= {"train", "valid", "test"}


def test_duplicate_document_under_different_title_detected():
    """Real case from the full dataset: the same contract filed twice
    under different title strings shares many identical clauses and has
    matching normalized company names. See LOG.md 2026-08-17."""
    shared_clause = lambda i: {"clause_text": f"shared clause {i}"}
    records = (
        [{"contract": "ARMSTRONGFLOORING,INC_01_07_2019-EX-10.2-INTELLECTUAL PROPERTY AGREEMENT",
          **shared_clause(i)} for i in range(4)]
        + [{"contract": "ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual Property Agreement",
            **shared_clause(i)} for i in range(4)]
    )
    canonical = find_duplicate_documents(records)
    a = "ARMSTRONGFLOORING,INC_01_07_2019-EX-10.2-INTELLECTUAL PROPERTY AGREEMENT"
    b = "ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual Property Agreement"
    assert canonical[a] == canonical[b]


def test_different_companies_sharing_boilerplate_not_merged():
    """Real false-positive case avoided: two genuinely different companies
    (different real contracts) shared 5 identical clauses purely from
    reusing the same boilerplate agreement template. Shared-clause count
    alone would wrongly merge them; the company-name-prefix check must
    also match. See LOG.md 2026-08-17."""
    shared_clause = lambda i: {"clause_text": f"boilerplate clause {i}"}
    records = (
        [{"contract": "INTELLIGENTHIGHWAYSOLUTIONS,INC_01_18_2018-EX-10.1-Strategic Alliance Agreement",
          **shared_clause(i)} for i in range(5)]
        + [{"contract": "SIBANNAC,INC_12_04_2017-EX-2.1-Strategic Alliance Agreement",
            **shared_clause(i)} for i in range(5)]
    )
    canonical = find_duplicate_documents(records)
    assert canonical == {}


def test_no_leakage_after_document_dedup():
    """The actual guarantee that matters: once two titles are recognized
    as the same document, their combined clauses always land in the same
    split -- never partially in train and partially in test."""
    shared_clause = lambda i: {"clause_text": f"shared clause {i}"}
    records = (
        [{"contract": "ARMSTRONGFLOORING,INC_01_07_2019-EX-10.2-INTELLECTUAL PROPERTY AGREEMENT",
          **shared_clause(i)} for i in range(4)]
        + [{"contract": "ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795_EX-10.2_Intellectual Property Agreement",
            **shared_clause(i)} for i in range(4)]
        + make_records(n_contracts=50, clauses_per_contract=3)
    )
    result = split_records(records, RATIOS, seed=1)
    splits_seen = set()
    for split_name, rows in result.items():
        for r in rows:
            if "Armstrong" in r["contract"] or "ARMSTRONG" in r["contract"]:
                splits_seen.add(split_name)
    assert len(splits_seen) == 1
