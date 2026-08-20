"""load_dpo_dataset is importable without trl.DPOTrainer -- the trl import
was deliberately moved inside main() so this stays testable even though
trl.DPOTrainer itself fails to import on this dev machine (needs
torch.distributed.fsdp.FSDPModule, added in a PyTorch release newer than
the locally-pinned 2.5.1). See LOG.md 2026-08-19."""
import json

from src.train.train_dpo import load_dpo_dataset


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_dpo_dataset_keeps_only_trl_expected_columns(tmp_path):
    path = tmp_path / "dpo.jsonl"
    _write_jsonl(path, [
        {"id": "a", "contract": "C", "category": "Cat", "clause_text": "text",
         "prompt": "P1", "chosen": "good", "rejected": "bad", "corruption_type": "truncate"},
    ])

    ds = load_dpo_dataset(str(path))

    assert set(ds.column_names) == {"prompt", "chosen", "rejected"}
    assert ds[0] == {"prompt": "P1", "chosen": "good", "rejected": "bad"}


def test_load_dpo_dataset_against_real_split():
    ds = load_dpo_dataset("data/splits/dpo_valid.jsonl")
    assert set(ds.column_names) == {"prompt", "chosen", "rejected"}
    assert len(ds) == 726
