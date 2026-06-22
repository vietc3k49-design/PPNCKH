"""LoRA SFT training utilities."""

from __future__ import annotations

import gc
import inspect
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import (
    BASE_MODEL_NAME,
    LABELS,
    TARGET_MODULES,
    TRAIN_CONFIG,
    TrainingConfig,
    adapter_output_dir,
    ensure_dataset_name,
)
from .data import load_records_for_dataset


def load_tokenizer(base_model_name: str = BASE_MODEL_NAME):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


class NerSftDataset:
    def __init__(self, records: list[dict], tokenizer, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _to_ids(value):
        if hasattr(value, "input_ids"):
            return value.input_ids
        return value

    def __getitem__(self, index: int):
        record = self.records[index]

        prompt_ids = self._to_ids(
            self.tokenizer.apply_chat_template(
                record["prompt"],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        response_ids = self.tokenizer(record["response"], add_special_tokens=False).input_ids

        if self.tokenizer.eos_token_id is not None:
            response_ids = response_ids + [self.tokenizer.eos_token_id]

        if len(prompt_ids) + len(response_ids) > self.max_length:
            return None

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


def materialize_tokenized_dataset(dataset: NerSftDataset) -> list[dict]:
    items: list[dict] = []
    for index in range(len(dataset)):
        item = dataset[index]
        if item is not None:
            items.append(item)
    return items


@dataclass
class NerDataCollator:
    tokenizer: object

    def __call__(self, features: list[dict]):
        import torch

        features = [item for item in features if item is not None]
        if not features:
            return {}

        max_len = max(len(item["input_ids"]) for item in features)
        input_ids, attention_mask, labels = [], [], []

        for item in features:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_bnb_config():
    import torch
    from transformers import BitsAndBytesConfig

    compute_dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_training_args(output_dir: Path | str, train_config: TrainingConfig = TRAIN_CONFIG):
    import torch
    from transformers import TrainingArguments

    params = dict(
        output_dir=str(output_dir),
        num_train_epochs=train_config.epochs,
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=train_config.eval_batch_size,
        gradient_accumulation_steps=train_config.grad_accum,
        learning_rate=train_config.lr,
        warmup_ratio=train_config.warmup_ratio,
        weight_decay=train_config.weight_decay,
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        eval_steps=train_config.eval_steps,
        save_total_limit=train_config.save_total_limit,
        optim=train_config.optim,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        params["eval_strategy"] = "steps"
    else:
        params["evaluation_strategy"] = "steps"
    if "save_strategy" in sig.parameters:
        params["save_strategy"] = "steps"
    if "logging_strategy" in sig.parameters:
        params["logging_strategy"] = "steps"

    return TrainingArguments(**params)


def load_base_model(base_model_name: str = BASE_MODEL_NAME):
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=build_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return prepare_model_for_kbit_training(model)


def make_lora_model(train_config: TrainingConfig = TRAIN_CONFIG, base_model_name: str = BASE_MODEL_NAME):
    from peft import LoraConfig, get_peft_model

    base = load_base_model(base_model_name)
    lora_config = LoraConfig(
        r=train_config.lora_r,
        lora_alpha=train_config.lora_alpha,
        lora_dropout=train_config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(TARGET_MODULES),
    )
    model = get_peft_model(base, lora_config)
    model.print_trainable_parameters()
    return model


def copy_checkpoint_for_resume(input_checkpoint_dir: Path | str, dataset_name: str, output_root: Path | str | None = None) -> Path:
    src = Path(input_checkpoint_dir)
    dst = adapter_output_dir(dataset_name, output_root) / src.name
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint not found: {src}")
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
    return dst


def train_one_adapter(
    dataset_name: str,
    data_root: Path | str,
    output_root: Path | str | None = None,
    base_model_name: str = BASE_MODEL_NAME,
    train_config: TrainingConfig = TRAIN_CONFIG,
    resume_from_checkpoint: str | Path | None = "auto",
) -> Path:
    import torch
    from transformers import Trainer
    from transformers.trainer_utils import get_last_checkpoint

    dataset_name = ensure_dataset_name(dataset_name)
    output_dir = adapter_output_dir(dataset_name, output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{dataset_name}] 1/5 Loading tokenizer and model...")
    tokenizer = load_tokenizer(base_model_name)
    model = make_lora_model(train_config, base_model_name)

    print(f"[{dataset_name}] 2/5 Building datasets...")
    train_records, dev_records = load_records_for_dataset(dataset_name, data_root)
    train_dataset = materialize_tokenized_dataset(NerSftDataset(train_records, tokenizer, train_config.max_length))
    eval_dataset = materialize_tokenized_dataset(NerSftDataset(dev_records, tokenizer, train_config.max_length))
    print(f"[{dataset_name}] usable train={len(train_dataset)} usable dev={len(eval_dataset)}")

    print(f"[{dataset_name}] 3/5 Preparing Trainer...")
    trainer = Trainer(
        model=model,
        args=build_training_args(output_dir, train_config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=NerDataCollator(tokenizer),
    )

    checkpoint = None
    if resume_from_checkpoint == "auto":
        checkpoint = get_last_checkpoint(output_dir)
    elif resume_from_checkpoint:
        checkpoint = str(resume_from_checkpoint)

    if checkpoint:
        print(f"[{dataset_name}] Resuming from checkpoint: {checkpoint}")

    print(f"[{dataset_name}] 4/5 Training...")
    trainer.train(resume_from_checkpoint=checkpoint)

    print(f"[{dataset_name}] 5/5 Saving final model and metadata...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": base_model_name,
        "dataset": dataset_name,
        "labels": list(LABELS[dataset_name]),
        "language": "vi",
        "max_length": train_config.max_length,
        "format": {
            "response": {
                "entities": [
                    {"label": "LABEL", "text": "text", "start": 0, "end": 0},
                ]
            }
        },
    }
    with (output_dir / "ner_adapter_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    del model
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[{dataset_name}] Done. Saved to: {output_dir}")
    return output_dir
