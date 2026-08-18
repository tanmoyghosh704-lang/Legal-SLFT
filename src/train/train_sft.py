"""SFT training via trl.SFTTrainer + peft LoRA/QLoRA.

Parameterized (model name, paths, batch size, LoRA rank, 4-bit on/off all
CLI args) so the exact same script runs locally for the 0.5B smoke test
and on Kaggle for the real 3B ablation runs -- no separate "local version"
and "Kaggle version" to keep in sync (spec section 0B: "do not maintain
two versions").

Expects a JSONL train/eval file with `prompt` and `target` fields (the
schema produced by build_irac.py / clean_sft.py). Renamed to `prompt`/
`completion` for trl's prompt-completion dataset format, which enables
completion-only loss automatically -- the model is trained to predict the
IRAC target, not to reproduce the prompt.

`--use-unsloth`: opt-in accelerated path (custom Triton kernels, ~2-5x
faster QLoRA per Unsloth's own benchmarks). Added after the vanilla
trl/bitsandbytes path measured ~15-20x slower than expected on a
single-GPU-pinned Kaggle T4 (117-125s/step against an expected 2-8s/step)
with no root cause found -- see LOG.md 2026-08-18. Kept as an opt-in flag
rather than a replacement: the vanilla path is the one actually validated
end-to-end (local smoke test, adapter reload, generation), and this
project's own rule is not to trust an unverified path with real GPU-hours
without a cheap dry run first, which now has to happen on Kaggle directly
since Unsloth couldn't be validated on this local dev machine (dependency
conflicts with the older torch pin this GPU's CUDA setup needs -- see
LOG.md 2026-08-18 for the full trail).
"""
import sys

# Unsloth must be imported before torch/transformers/etc. for its
# monkey-patches to take effect (a documented Unsloth requirement) -- so
# this checks the raw argv for --use-unsloth before any other import runs,
# since argparse itself needs those other imports loaded first.
_USE_UNSLOTH = "--use-unsloth" in sys.argv
if _USE_UNSLOTH:
    from unsloth import FastLanguageModel

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_prompt_completion_dataset(path: str):
    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.rename_column("target", "completion")
    keep = ["prompt", "completion"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-epochs", type=float, default=3)
    parser.add_argument("--max-steps", type=int, default=-1, help="Overrides num-epochs if set (>0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-gradient-checkpointing", action="store_true",
                         help="Diagnostic lever: isolate whether checkpointing's recompute is the dominant "
                              "per-step cost. See LOG.md 2026-08-18.")
    parser.add_argument("--use-unsloth", action="store_true",
                         help="Opt-in accelerated QLoRA path via Unsloth. See module docstring.")
    args = parser.parse_args()

    train_ds = load_prompt_completion_dataset(args.train_file)
    eval_ds = load_prompt_completion_dataset(args.eval_file) if args.eval_file else None
    print(f"train: {len(train_ds)} examples" + (f", eval: {len(eval_ds)} examples" if eval_ds else ""))

    # bf16 requires Ampere+ (compute capability 8.0+); T4/Turing (7.5) is
    # below that line. The vanilla path below hardcodes bf16=True and it
    # "works" (trains without crashing), just very slowly (117-125s/step on
    # a Kaggle T4, vs. an expected 2-8s/step) -- transformers' own bf16
    # eligibility check is apparently stricter under Unsloth's patched
    # SFTConfig than under vanilla trl, since only the Unsloth path raised
    # "Your setup doesn't support bf16/gpu... You need Ampere+ GPU" outright
    # rather than silently proceeding unaccelerated. That's a strong signal
    # the vanilla path's mystery slowness was bf16 running without Tensor
    # Core acceleration the whole time. Scoped this fix to the Unsloth
    # branch only, not the vanilla one: an earlier attempt to make bf16
    # conditional in the vanilla path crashed with a dtype-mismatch error
    # ("_amp_foreach_non_finite_check_and_unscale_cuda not implemented for
    # 'BFloat16'") that was never root-caused, and that path currently
    # works end-to-end, just slowly -- not worth the regression risk of
    # touching it again without being able to test locally. See LOG.md
    # 2026-08-18.
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0)

    if args.use_unsloth:
        # Unsloth handles quantization and device placement internally via
        # load_in_4bit -- no separate BitsAndBytesConfig/device_map needed,
        # and it's single-GPU by design (doesn't do the naive multi-GPU
        # auto-placement that caused the pipeline-parallelism bug in the
        # vanilla path below). get_peft_model applies LoRA directly, so
        # peft_config stays None for SFTTrainer -- passing both would
        # double-apply LoRA. use_gradient_checkpointing="unsloth" is
        # Unsloth's own async-offloading implementation, not the vanilla
        # boolean flag. See module docstring for why this path exists and
        # LOG.md 2026-08-18 for what it's replacing.
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model, max_seq_length=args.max_seq_length, load_in_4bit=args.load_in_4bit,
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=args.target_modules, bias="none",
            use_gradient_checkpointing="unsloth" if not args.disable_gradient_checkpointing else False,
            random_state=args.seed,
        )
        peft_config = None
    else:
        # Tried making this conditional on GPU generation (fp16 on pre-Ampere,
        # reasoning Turing/T4 lacks Tensor Core bf16 support) and reverted it --
        # it crashed the smoke test with "_amp_foreach_non_finite_check_and_
        # unscale_cuda not implemented for 'BFloat16'" (some component ends up
        # bf16 regardless of the requested dtype, which breaks fp16's GradScaler
        # assumptions), and a local matmul benchmark actually showed bf16
        # *faster* than fp16 on this Turing GPU anyway, contradicting the
        # hypothesis it was meant to fix. bf16 unconditionally is simpler and
        # was never actually shown to be the problem -- see LOG.md 2026-08-18.
        quantization_config = None
        if args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            )

        # Explicitly pin to a single GPU rather than passing a bare model-id
        # string to SFTTrainer and letting transformers/accelerate auto-place
        # it. With >1 GPU visible (e.g. Kaggle's "T4 x2" accelerator), the
        # default placement can split the model pipeline-parallel across both
        # devices -- the exact bug already found and fixed in generate.py's
        # docstring for eval generation (measured ~15.6s/row vs. one pinned
        # GPU during teacher generation). Even after pinning, a Kaggle T4 run
        # still measured ~117-125s/step against an expected 2-8s/step, with
        # no further root cause found -- see the --use-unsloth path above,
        # added specifically because this path's remaining slowness was
        # never explained. See LOG.md 2026-08-18.
        device_map = {"": 0} if torch.cuda.is_available() else None
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quantization_config, device_map=device_map, dtype=dtype,
        )
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=args.target_modules, task_type="CAUSAL_LM",
        )

    # Direct confirmation of GPU placement, not just an assumption -- print
    # per-GPU memory right after load so a multi-GPU session (e.g. Kaggle's
    # "T4 x2") shows allocation on device 0 only. If a second visible GPU
    # shows meaningful memory too, the model is still being split, and that
    # needs investigating before trusting any step-timing number. Applies
    # to both paths above. See LOG.md 2026-08-18.
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        max_length=args.max_seq_length,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        report_to=[],
        seed=args.seed,
        # use_bf16 (Ampere+ only) for the Unsloth path -- see the comment
        # above where it's computed for why this is scoped to Unsloth only,
        # not the vanilla path below. See LOG.md 2026-08-18.
        bf16=use_bf16 if args.use_unsloth else torch.cuda.is_available(),
        fp16=(torch.cuda.is_available() and not use_bf16) if args.use_unsloth else False,
        # Without this, activations for every layer are kept in memory for
        # the whole forward pass so backward has them -- fine for the 0.5B
        # smoke-test model, but OOM'd a 3B model at batch_size=4/seq_len=2048
        # on a 16GB Kaggle T4 (14.56GB total, <100MB free at the point of
        # failure). Gradient checkpointing recomputes activations during
        # backward instead of storing them all -- standard QLoRA technique,
        # trades compute for memory. use_reentrant=False is the currently
        # recommended mode (avoids known issues with the old reentrant
        # autograd implementation under peft). Left False when using Unsloth
        # -- it already applied its own checkpointing implementation via
        # use_gradient_checkpointing="unsloth" in get_peft_model above;
        # enabling it again here would be redundant at best, conflicting at
        # worst. See LOG.md 2026-08-18.
        gradient_checkpointing=not args.disable_gradient_checkpointing and not args.use_unsloth,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
