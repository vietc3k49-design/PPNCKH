"""Dataset summary and optional plotting utilities."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .config import SPLITS, get_dataset_paths
from .data import count_tokens
from .io_utils import read_jsonl, write_json


def build_dataset_summary(data_root: Path | str) -> dict:
    datasets = get_dataset_paths(data_root)
    path_rows = []
    summary_rows = []
    label_counts = Counter()
    top_terms = Counter()

    for dataset_name, cfg in datasets.items():
        for split in SPLITS:
            path = cfg[split]
            path_rows.append(
                {
                    "dataset": dataset_name,
                    "split": split,
                    "path": str(path),
                    "exists": path.exists(),
                    "size_mb": round(path.stat().st_size / 1024 / 1024, 2) if path.exists() else None,
                }
            )
            if not path.exists():
                continue

            rows = read_jsonl(path)
            total_entities = 0
            empty_samples = 0
            vi_lengths = []
            en_lengths = []

            for row in rows:
                entities = row.get("entities", []) or []
                total_entities += len(entities)
                empty_samples += int(len(entities) == 0)
                vi_lengths.append(count_tokens(row.get("vi_sentence_str")))
                en_lengths.append(count_tokens(row.get("en_sentence_str")))

                for entity in entities:
                    label = entity.get("label", "UNKNOWN")
                    label_counts[(dataset_name, split, label)] += 1
                    term = str(entity.get("vi_term") or "").strip()
                    if term:
                        top_terms[(dataset_name, label, term)] += 1

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "split": split,
                    "samples": len(rows),
                    "entities": total_entities,
                    "empty_samples": empty_samples,
                    "avg_entities_per_sample": total_entities / len(rows) if rows else 0,
                    "avg_vi_tokens": sum(vi_lengths) / len(vi_lengths) if vi_lengths else 0,
                    "avg_en_tokens": sum(en_lengths) / len(en_lengths) if en_lengths else 0,
                }
            )

    label_rows = [
        {"dataset": ds, "split": split, "label": label, "count": count}
        for (ds, split, label), count in sorted(label_counts.items())
    ]
    top_term_rows = [
        {"dataset": ds, "label": label, "term": term, "count": count}
        for (ds, label, term), count in top_terms.most_common(100)
    ]
    return {
        "paths": path_rows,
        "summary": summary_rows,
        "labels": label_rows,
        "top_terms": top_term_rows,
    }


def save_dataset_summary(data_root: Path | str, output_path: Path | str) -> dict:
    summary = build_dataset_summary(data_root)
    write_json(summary, output_path)
    return summary


def save_summary_csv(summary: dict, output_dir: Path | str) -> None:
    import pandas as pd

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in ("paths", "summary", "labels", "top_terms"):
        pd.DataFrame(summary[key]).to_csv(out_dir / f"{key}.csv", index=False, encoding="utf-8-sig")


def describe_datasets(data_root: Path | str, output_dir: Path | str | None = None, csv: bool = False) -> dict:
    summary = build_dataset_summary(data_root)
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(summary, out_dir / "dataset_summary.json")
        if csv:
            save_summary_csv(summary, out_dir)

    return summary
