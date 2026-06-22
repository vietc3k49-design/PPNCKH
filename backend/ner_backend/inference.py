"""LoRA adapter loading and batch inference."""

from __future__ import annotations

import json
from pathlib import Path

from .config import BASE_MODEL_NAME, LABELS, ensure_dataset_name
from .io_utils import iter_jsonl
from .prompts import build_prompt_text


def _torch_dtype(torch, dtype: str):
    if dtype == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, dtype)


def _model_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def load_lora_model(
    adapter_path: Path | str,
    base_model_name: str = BASE_MODEL_NAME,
    dtype: str = "auto",
    load_in_4bit: bool = False,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = _torch_dtype(torch, dtype)

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    return model, tokenizer


def generate_one(
    model,
    tokenizer,
    dataset_name: str,
    text: str,
    max_new_tokens: int = 256,
) -> str:
    import torch

    dataset_name = ensure_dataset_name(dataset_name)
    prompt = build_prompt_text(tokenizer, dataset_name, LABELS[dataset_name], text)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = _model_input_device(model)
    if device is not None:
        inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    input_length = inputs.input_ids.shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)


def run_inference_file(
    dataset_name: str,
    input_file: Path | str,
    output_file: Path | str,
    adapter_path: Path | str,
    base_model_name: str = BASE_MODEL_NAME,
    max_new_tokens: int = 256,
    limit: int | None = None,
    load_in_4bit: bool = False,
) -> Path:
    from tqdm.auto import tqdm

    dataset_name = ensure_dataset_name(dataset_name)
    model, tokenizer = load_lora_model(
        adapter_path,
        base_model_name=base_model_name,
        load_in_4bit=load_in_4bit,
    )

    rows = list(iter_jsonl(input_file))
    if limit is not None:
        rows = rows[:limit]

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f_out:
        for row in tqdm(rows, desc=f"Inference {dataset_name}"):
            text = row.get("vi_sentence_str") or row.get("vi_sentence") or row.get("input_text") or ""
            if not text:
                continue

            response = generate_one(
                model,
                tokenizer,
                dataset_name=dataset_name,
                text=text,
                max_new_tokens=max_new_tokens,
            )
            f_out.write(
                json.dumps(
                    {
                        "input_text": text,
                        "predicted_json": response,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return out_path
