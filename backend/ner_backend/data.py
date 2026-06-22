"""Dataset loading and SFT record construction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import LABELS, get_dataset_paths
from .io_utils import read_jsonl
from .prompts import build_messages, format_response


def count_tokens(text: str | None) -> int:
    return len(str(text or "").split())


def normalize_gold_entities(row: dict, allowed_labels: Iterable[str] | None = None) -> list[dict]:
    allowed = set(allowed_labels) if allowed_labels is not None else None
    entities: list[dict] = []

    for entity in row.get("entities", []) or []:
        label = entity.get("label")
        term = entity.get("vi_term")
        start = entity.get("vi_start_token_idx")
        end = entity.get("vi_end_token_idx")

        if label is None or term is None or start is None or end is None:
            continue
        if allowed is not None and label not in allowed:
            continue

        try:
            start_i = int(start)
            end_i = int(end)
        except (TypeError, ValueError):
            continue

        term_s = str(term).strip()
        if not term_s or start_i < 0 or end_i < start_i:
            continue

        entities.append(
            {
                "label": str(label),
                "term": term_s,
                "start_token_idx": start_i,
                "end_token_idx": end_i,
            }
        )

    entities.sort(key=lambda x: (x["start_token_idx"], x["end_token_idx"], x["label"], x["term"]))
    return entities


def build_sft_records(rows: Iterable[dict], dataset_name: str, labels: Iterable[str]) -> list[dict]:
    records: list[dict] = []
    for idx, row in enumerate(rows):
        text = row.get("vi_sentence_str")
        if not text:
            continue

        entities = normalize_gold_entities(row, labels)
        records.append(
            {
                "id": f"{dataset_name}-{idx}-vi",
                "dataset": dataset_name,
                "language": "vi",
                "text": text,
                "prompt": build_messages(dataset_name, labels, text),
                "response": format_response(entities),
                "entities": entities,
            }
        )
    return records


def load_records_for_dataset(dataset_name: str, data_root: Path | str) -> tuple[list[dict], list[dict]]:
    paths = get_dataset_paths(data_root)
    labels = LABELS[dataset_name]
    train_rows = read_jsonl(paths[dataset_name]["train"])
    dev_rows = read_jsonl(paths[dataset_name]["dev"])
    return (
        build_sft_records(train_rows, dataset_name, labels),
        build_sft_records(dev_rows, dataset_name, labels),
    )


def validate_dataset_files(data_root: Path | str) -> dict[str, dict[str, Path]]:
    paths = get_dataset_paths(data_root)
    missing: list[str] = []
    for dataset_name, cfg in paths.items():
        for split in ("train", "dev", "test"):
            if not cfg[split].exists():
                missing.append(f"{dataset_name}/{split}: {cfg[split]}")

    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Missing dataset files:\n{joined}")

    return paths
