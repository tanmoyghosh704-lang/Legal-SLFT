"""DPO training via trl.DPOTrainer + peft LoRA/QLoRA.

Same parameterized-script philosophy as train_sft.py: one script for both
`dpo_from_sft` (warm-started from an already-trained SFT adapter, via
--adapter) and `dpo_from_base` (fresh LoRA on the base model, --adapter
omitted) -- not two scripts to keep in sync.

Expects a JSONL train/eval file with `prompt`, `chosen`, `rejected` fields
(the schema produced by build_dpo.py) -- trl's DPOTrainer standard column
names, no renaming needed.

Reuses every fix already found and verified for train_sft.py on 2026-08-18
(see LOG.md): single-GPU device pin, gradient checkpointing, capability-
aware bf16/fp16 (DPOConfig defaults bf16=True unconditionally per its own
docstring -- the exact same bug already found in SFTConfig, so this is not
optional here), and the --use-unsloth opt-in accelerated path.

Warm-start (--adapter): loads the existing PEFT adapter with
is_trainable=True and continues training those same LoRA weights, rather
than merging into the base model first -- simpler, and avoids merging a
LoRA delta into a 4-bit-quantized base (not straightforward with QLoRA).
When the model is already a PeftModel, DPOTrainer uses it as its own
implicit reference model (adapters disabled internally) when ref_model is
left as None -- confirmed by reading trl's own dpo_trainer.py source
locally, since `trl.DPOTrainer` itself couldn't even be imported locally to
test this (see module comment below on why).

IMPORTANT caveat: unlike train_sft.py, this script's local verification is
partial. `trl.DPOTrainer` fails to import on this dev machine's pinned
torch==2.5.1 (needs `torch.distributed.fsdp.FSDPModule`, added in a later
PyTorch release) -- so only the dataset-loading logic could be verified
locally (see tests/test_train_dpo.py); the actual DPOTrainer construction
and training loop are unverified until run on Kaggle, where a newer
PyTorch is expected. Lean on the Kaggle notebook's dry-run cell more than
usual for this one. See LOG.md 2026-08-19.
"""
import sys

# Unsloth must be imported before torch/transformers/etc. for its
# monkey-patches to take effect (a documented Unsloth requirement) -- so
# this checks the raw argv for --use-unsloth before any other import runs,
# since argparse itself needs those other imports loaded first. Same
# pattern as train_sft.py.
_USE_UNSLOTH = "--use-unsloth" in sys.argv
if _USE_UNSLOTH:
    from unsloth import FastLanguageModel

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
# trl imported lazily inside main(), not here -- trl.DPOTrainer fails to
# import at all on the local dev machine (needs torch.distributed.fsdp.
# FSDPModule, added in a PyTorch release newer than the locally-pinned
# 2.5.1). Keeping it out of the module-level imports means
# load_dpo_dataset can still be imported and unit-tested locally even
# though DPOTrainer itself can't be. No behavior change on Kaggle, where
# a newer PyTorch is expected to have this. See LOG.md 2026-08-19.


def load_dpo_dataset(path: str):
    ds = load_dataset("json", data_files=path, split="train")
    keep = ["prompt", "chosen", "rejected"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    return ds


def main():
    from trl import DPOConfig, DPOTrainer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default=None,
                         help="Warm-start from an existing PEFT adapter dir (dpo_from_sft). "
                              "Omit for dpo_from_base (fresh LoRA on the base model).")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-epochs", type=float, default=1)
    parser.add_argument("--max-steps", type=int, default=-1, help="Overrides num-epochs if set (>0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--use-unsloth", action="store_true",
                         help="Opt-in accelerated QLoRA path via Unsloth. See module docstring.")
    args = parser.parse_args()

    train_ds = load_dpo_dataset(args.train_file)
    eval_ds = load_dpo_dataset(args.eval_file) if args.eval_file else None
    print(f"train: {len(train_ds)} examples" + (f", eval: {len(eval_ds)} examples" if eval_ds else ""))

    # See train_sft.py 2026-08-18 for the full story: T4/Turing (compute
    # capability 7.5) lacks Tensor Core bf16 acceleration, only Ampere+
    # (8.0+) has it. DPOConfig defaults bf16=True unconditionally (per its
    # own docstring) if this isn't overridden explicitly -- same bug,
    # reintroduced by default if not handled here too.
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0)
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)

    if args.use_unsloth:
        # If --adapter is given, load the saved LoRA checkpoint directly --
        # Unsloth's from_pretrained supports a checkpoint dir the same way
        # it supports a base model id (auto-detects the adapter_config.json
        # and loads the referenced base model underneath it). Unverified
        # locally, see module docstring.
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.adapter or args.model, max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit, dtype=dtype,
        )
        if args.adapter:
            FastLanguageModel.for_training(model)
            peft_config = None
        else:
            model = FastLanguageModel.get_peft_model(
                model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                target_modules=args.target_modules, bias="none",
                use_gradient_checkpointing="unsloth" if not args.disable_gradient_checkpointing else False,
                random_state=args.seed,
            )
            peft_config = None
    else:
        quantization_config = None
        if args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            )
        device_map = {"": 0} if torch.cuda.is_available() else None
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quantization_config, device_map=device_map, dtype=dtype,
        )
        if args.adapter:
            model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=True)
            peft_config = None
        else:
            model = base_model
            peft_config = LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                target_modules=args.target_modules, task_type="CAUSAL_LM",
            )

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        report_to=[],
        seed=args.seed,
        bf16=use_bf16 if args.use_unsloth else torch.cuda.is_available(),
        fp16=(torch.cuda.is_available() and not use_bf16) if args.use_unsloth else False,
        gradient_checkpointing=not args.disable_gradient_checkpointing and not args.use_unsloth,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ref_model intentionally left as None: since `model` is a PeftModel in
    # every branch above (either freshly LoRA-wrapped or warm-started),
    # DPOTrainer uses the model itself as its own implicit reference
    # (adapters disabled internally for the reference pass) -- confirmed by
    # reading trl/trainer/dpo_trainer.py directly (line ~919: `if ref_model
    # is None: if is_peft_model(self.model)... self.ref_model = None`).
    # Avoids loading a second full model copy just for the reference.
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
