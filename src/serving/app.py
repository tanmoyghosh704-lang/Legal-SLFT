"""FastAPI serving for the best-performing adapter (dpo_from_sft by default).

Reuses CANONICAL_PROMPT_TEMPLATE and parse_irac from build_irac.py -- the
same single source of truth every other stage (SFT, DPO, eval generation)
already imports, per this project's own "never redefine the template
inline" rule (see config.yaml's irac_template comment). Model loading
mirrors src/eval/generate.py's device_map={"":0} single-GPU pin, the same
fix already found necessary for every other model-loading code path in
this project.

Adapter is configurable via the ADAPTER_NAME env var (defaults to
dpo_from_sft -- the best result on both the deterministic rubric and,
with the caveat documented in WRITEUP.md section 6.3, the LLM judge). Set
it to any of sft_qlora / sft_lora_fp / dpo_from_base / dpo_from_sft to
compare adapters live, or unset ADAPTER_NAME entirely for zero-shot
baseline.

Run: uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
"""
import os
from contextlib import asynccontextmanager

import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data.build_irac import CANONICAL_PROMPT_TEMPLATE, is_substantive_clause, parse_irac

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")
ADAPTER_NAME = os.environ.get("ADAPTER_NAME", "dpo_from_sft")
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "1") != "0"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "350"))

_state: dict = {"model": None, "tokenizer": None, "model_id": None, "adapter_name": None}


def _load():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_id = cfg["model"]["candidates"][cfg["model"]["active"]]["hf_id"]
    adapters_dir = cfg["paths"]["adapters_dir"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if LOAD_IN_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
    device_map = {"": 0} if torch.cuda.is_available() else None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quantization_config, device_map=device_map, dtype=dtype,
    )

    if ADAPTER_NAME:
        from peft import PeftModel
        adapter_path = os.path.join(adapters_dir, ADAPTER_NAME)
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(
                f"ADAPTER_NAME={ADAPTER_NAME!r} but {adapter_path} doesn't exist. "
                f"Available: {[d for d in os.listdir(adapters_dir) if os.path.isdir(os.path.join(adapters_dir, d))] if os.path.isdir(adapters_dir) else '(adapters_dir missing)'}"
            )
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["model_id"] = model_id
    _state["adapter_name"] = ADAPTER_NAME


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield
    _state["model"] = None
    _state["tokenizer"] = None


app = FastAPI(
    title="Legal Clause IRAC Analyzer",
    description="SFT+DPO-aligned Qwen2.5-3B-Instruct, IRAC-format contract clause analysis.",
    lifespan=lifespan,
)


class AnalyzeRequest(BaseModel):
    category: str = Field(..., examples=["Non-Compete"])
    clause_text: str = Field(..., min_length=1, examples=[
        "Either party may terminate this Agreement upon thirty (30) days written notice to the other party."
    ])


class AnalyzeResponse(BaseModel):
    issue: str | None
    rule: str | None
    application: str | None
    conclusion: str | None
    parse_ok: bool
    raw_response: str
    model_id: str
    adapter_name: str | None


@app.get("/health")
def health():
    return {
        "status": "ok" if _state["model"] is not None else "loading",
        "model_id": _state["model_id"],
        "adapter_name": _state["adapter_name"],
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Same substantiveness filter used to clean the training data itself
    # (build_irac.py) -- reject the same kind of unusable fragment
    # ("This", a mid-sentence truncation) at inference time rather than
    # silently generating a full but meaningless analysis for it.
    if not is_substantive_clause(req.clause_text):
        raise HTTPException(
            status_code=422,
            detail="clause_text doesn't look like a substantive clause (too short/fragmentary) "
                   "to analyze meaningfully.",
        )

    model, tokenizer = _state["model"], _state["tokenizer"]
    prompt = CANONICAL_PROMPT_TEMPLATE.format(category=req.category, clause_text=req.clause_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    raw_response = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

    sections = parse_irac(raw_response)
    return AnalyzeResponse(
        issue=sections["Issue"] if sections else None,
        rule=sections["Rule"] if sections else None,
        application=sections["Application"] if sections else None,
        conclusion=sections["Conclusion"] if sections else None,
        parse_ok=sections is not None,
        raw_response=raw_response,
        model_id=_state["model_id"],
        adapter_name=_state["adapter_name"],
    )
