"""Project configuration shared by CLI commands and modules."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(os.getenv("NER_DATA_ROOT", REPO_ROOT))
DEFAULT_WORK_DIR = Path(os.getenv("NER_WORK_DIR", REPO_ROOT / "output"))

BASE_MODEL_NAME = os.getenv("NER_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
JUDGE_MODEL_NAME = os.getenv("NER_JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

LABELS = {
    "vimedner": ("DISEASE", "SYMPTOM", "TREATMENT", "CAUSE", "DIAGNOSTIC"),
    "vimq": ("SYMPTOM_AND_DISEASE", "MEDICAL_PROCEDURE", "DRUG"),
}

DATASET_DESCRIPTIONS = {
    "vimedner": "Van ban y te tong quat",
    "vimq": "Cau hoi y te",
}

SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class TrainingConfig:
    max_length: int = 384
    epochs: float = 1.0
    batch_size: int = 1
    eval_batch_size: int = 1
    grad_accum: int = 8
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    eval_steps: int = 150
    save_steps: int = 150
    save_total_limit: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    optim: str = "paged_adamw_8bit"


TRAIN_CONFIG = TrainingConfig()


def get_dataset_paths(data_root: Path | str = DEFAULT_DATA_ROOT) -> dict[str, dict]:
    """Return train/dev/test paths and labels for each supported dataset."""

    root = Path(data_root)
    return {
        name: {
            "train": root / name / "train_fixed_verified.jsonl",
            "dev": root / name / "dev_fixed_verified.jsonl",
            "test": root / name / "test_fixed_verified.jsonl",
            "labels": labels,
            "description": DATASET_DESCRIPTIONS.get(name, ""),
        }
        for name, labels in LABELS.items()
    }


def adapter_output_dir(dataset_name: str, output_root: Path | str | None = None) -> Path:
    root = Path(output_root) if output_root is not None else DEFAULT_WORK_DIR / "ner_adapters"
    return root / f"qwen2p5_3b_{dataset_name}_ner_lora"


def ensure_dataset_name(dataset_name: str) -> str:
    name = dataset_name.lower().strip()
    if name not in LABELS:
        valid = ", ".join(sorted(LABELS))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Valid values: {valid}")
    return name
