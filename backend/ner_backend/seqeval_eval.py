"""Seqeval-based BIO evaluation and CoNLL export."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .io_utils import read_jsonl
from .schema import extract_entities, get_sentence


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def tokenize_vi(text: str) -> list[str]:
    text = normalize_spaces(text)
    return text.split(" ") if text else []


def normalize_label(label: str) -> str:
    return label.strip().upper().replace(" ", "_").replace("-", "_")


def build_bio_tags(tokens: list[str], entities: Iterable[dict]):
    tags = ["O"] * len(tokens)
    occupied = [False] * len(tokens)
    overlaps = []
    out_of_range = []

    sorted_entities = sorted(entities, key=lambda e: (e.get("start", 10**9), e.get("end", 10**9)))
    for entity in sorted_entities:
        start = entity.get("start")
        end = entity.get("end")
        label = normalize_label(entity.get("label", "ENT"))

        if start is None or end is None or start < 0 or end < start or end >= len(tokens):
            out_of_range.append(entity)
            continue

        if any(occupied[i] for i in range(start, end + 1)):
            overlaps.append(entity)
            continue

        tags[start] = f"B-{label}"
        for i in range(start + 1, end + 1):
            tags[i] = f"I-{label}"
        for i in range(start, end + 1):
            occupied[i] = True

    return tags, overlaps, out_of_range


def _extract_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    spans = set()
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag == "O" or "-" not in tag:
            i += 1
            continue

        prefix, label = tag.split("-", 1)
        if prefix not in ("B", "I"):
            i += 1
            continue

        start = i
        i += 1
        while i < len(tags) and tags[i] == f"I-{label}":
            i += 1
        spans.add((start, i - 1, label))
    return spans


def compute_entity_metrics(gold_tags_all: list[list[str]], pred_tags_all: list[list[str]]) -> dict:
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    report = classification_report(gold_tags_all, pred_tags_all, output_dict=True, zero_division=0)

    per_type = {}
    for label, values in report.items():
        if label in {"micro avg", "macro avg", "weighted avg"} or not isinstance(values, dict):
            continue
        per_type[label] = {
            "support_gold": int(values.get("support", 0)),
            "support_pred": 0,
            "precision": values.get("precision", 0.0),
            "recall": values.get("recall", 0.0),
            "f1": values.get("f1-score", 0.0),
        }

    gold_by_type = Counter()
    pred_by_type = Counter()
    tp_by_type = Counter()

    for gold_tags, pred_tags in zip(gold_tags_all, pred_tags_all):
        gold_spans = _extract_spans(gold_tags)
        pred_spans = _extract_spans(pred_tags)
        matched = gold_spans & pred_spans

        for _, _, label in gold_spans:
            gold_by_type[label] += 1
        for _, _, label in pred_spans:
            pred_by_type[label] += 1
        for _, _, label in matched:
            tp_by_type[label] += 1

    labels = sorted(set(gold_by_type) | set(pred_by_type) | set(per_type))
    for label in labels:
        per_type.setdefault(
            label,
            {
                "support_gold": 0,
                "support_pred": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
        )
        per_type[label]["support_gold"] = gold_by_type[label]
        per_type[label]["support_pred"] = pred_by_type[label]

    tp = sum(tp_by_type.values())
    fp = sum(pred_by_type.values()) - tp
    fn = sum(gold_by_type.values()) - tp

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision_score(gold_tags_all, pred_tags_all),
        "recall": recall_score(gold_tags_all, pred_tags_all),
        "f1": f1_score(gold_tags_all, pred_tags_all),
        "per_type": per_type,
    }


def run_seqeval(gold_path: Path | str, pred_path: Path | str, output_dir: Path | str, name: str) -> dict:
    gold_rows = read_jsonl(gold_path)
    pred_rows = read_jsonl(pred_path)
    used_n = min(len(gold_rows), len(pred_rows))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conll_path = out_dir / f"{name}.conll"
    gold_bio_path = out_dir / f"{name}.gold.bio"
    pred_bio_path = out_dir / f"{name}.pred.bio"
    summary_path = out_dir / f"{name}.summary.json"

    gold_tags_all = []
    pred_tags_all = []
    mismatched_sentences = 0
    gold_overlap_skipped = pred_overlap_skipped = 0
    gold_out_of_range_skipped = pred_out_of_range_skipped = 0

    with conll_path.open("w", encoding="utf-8") as f_conll, gold_bio_path.open(
        "w", encoding="utf-8"
    ) as f_gold, pred_bio_path.open("w", encoding="utf-8") as f_pred:
        for idx in range(used_n):
            gold_row = gold_rows[idx]
            pred_row = pred_rows[idx]

            gold_sentence = get_sentence(gold_row)
            pred_sentence = get_sentence(pred_row)
            if pred_sentence and normalize_spaces(gold_sentence) != normalize_spaces(pred_sentence):
                mismatched_sentences += 1

            tokens = tokenize_vi(gold_sentence)
            gold_tags, gold_overlap, gold_oor = build_bio_tags(tokens, extract_entities(gold_row, role="gold"))
            pred_tags, pred_overlap, pred_oor = build_bio_tags(tokens, extract_entities(pred_row, role="pred"))

            gold_overlap_skipped += len(gold_overlap)
            pred_overlap_skipped += len(pred_overlap)
            gold_out_of_range_skipped += len(gold_oor)
            pred_out_of_range_skipped += len(pred_oor)

            gold_tags_all.append(gold_tags)
            pred_tags_all.append(pred_tags)

            for token, gold_tag, pred_tag in zip(tokens, gold_tags, pred_tags):
                f_conll.write(f"{token}\t{gold_tag}\t{pred_tag}\n")
                f_gold.write(f"{token}\t{gold_tag}\n")
                f_pred.write(f"{token}\t{pred_tag}\n")

            f_conll.write("\n")
            f_gold.write("\n")
            f_pred.write("\n")

    metrics = compute_entity_metrics(gold_tags_all, pred_tags_all)
    summary = {
        "dataset": name,
        "gold_file": str(gold_path),
        "pred_file": str(pred_path),
        "sentences_gold": len(gold_rows),
        "sentences_pred": len(pred_rows),
        "sentences_used": used_n,
        "mismatched_sentences": mismatched_sentences,
        "gold_overlap_skipped": gold_overlap_skipped,
        "pred_overlap_skipped": pred_overlap_skipped,
        "gold_out_of_range_skipped": gold_out_of_range_skipped,
        "pred_out_of_range_skipped": pred_out_of_range_skipped,
        "metrics": metrics,
        "output_files": {
            "conll": str(conll_path),
            "gold_bio": str(gold_bio_path),
            "pred_bio": str(pred_bio_path),
            "summary": str(summary_path),
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary
