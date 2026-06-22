"""Clean raw model predictions and compute exact/IoU metrics."""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Iterable

from .config import LABELS, ensure_dataset_name
from .io_utils import read_jsonl, write_jsonl
from .schema import extract_entities, get_sentence


VI_PUNCT = string.punctuation + "”“\"'…"


def strip_code_fence(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```json"):
        return cleaned[7:-3].strip()
    if cleaned.startswith("```"):
        return cleaned[3:-3].strip()
    return cleaned


def parse_prediction_payload(row: dict) -> dict:
    if isinstance(row.get("predicted_json"), str):
        raw = strip_code_fence(row["predicted_json"])
        return json.loads(raw)
    if isinstance(row.get("entities"), list):
        return {"entities": row["entities"]}
    if isinstance(row.get("pred"), list):
        return {"entities": row["pred"]}
    return {"entities": []}


def find_real_token_indices(sentence: str, term: str):
    if not term or not sentence:
        return None, None, None

    sentence_tokens = sentence.split()
    term_cleaned = str(term).strip(VI_PUNCT + " ")
    if not term_cleaned:
        return None, None, None

    term_tokens = term_cleaned.split()
    n_sentence, n_term = len(sentence_tokens), len(term_tokens)
    sentence_lower = [token.lower() for token in sentence_tokens]
    term_lower = [token.lower() for token in term_tokens]

    for start in range(n_sentence - n_term + 1):
        if sentence_lower[start : start + n_term] == term_lower:
            return start, start + n_term - 1, term_cleaned

    return None, None, None


def calculate_iou(pred_start: int, pred_end: int, gold_start: int, gold_end: int) -> float:
    intersection_start = max(pred_start, gold_start)
    intersection_end = min(pred_end, gold_end)
    if intersection_start > intersection_end:
        return 0.0

    intersection_length = intersection_end - intersection_start + 1
    pred_length = pred_end - pred_start + 1
    gold_length = gold_end - gold_start + 1
    union_length = pred_length + gold_length - intersection_length
    return intersection_length / union_length


def calculate_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def clean_prediction_row(
    gold_row: dict,
    pred_row: dict,
    allowed_labels: Iterable[str] | None = None,
) -> dict:
    allowed = set(allowed_labels) if allowed_labels is not None else None
    sentence = get_sentence(gold_row) or get_sentence(pred_row)
    gold_entities = extract_entities(gold_row, role="gold", allowed_labels=allowed)
    pred_entities: list[dict] = []

    try:
        payload = parse_prediction_payload(pred_row)
    except (json.JSONDecodeError, TypeError, ValueError):
        payload = {"entities": []}

    for entity in payload.get("entities", []) or []:
        raw_term = str(entity.get("text") or entity.get("vi_term") or "")
        label = str(entity.get("label") or "").strip()
        if allowed is not None and label not in allowed:
            continue

        start, end, cleaned_term = find_real_token_indices(sentence, raw_term)
        if start is None or end is None or cleaned_term is None:
            continue

        pred_entities.append(
            {
                "label": label,
                "vi_term": cleaned_term,
                "start": start,
                "end": end,
            }
        )

    return {
        "vi_sentence": sentence,
        "gold": gold_entities,
        "pred": pred_entities,
    }


def evaluate_cleaned_rows(rows: Iterable[dict], iou_threshold: float = 0.5) -> dict:
    em_tp = em_fp = em_fn = 0
    pm_tp = pm_fp = pm_fn = 0

    for row in rows:
        gold_entities = extract_entities(row, role="gold")
        pred_entities = extract_entities(row, role="pred")

        gold_exact = {
            (entity["vi_term"].lower(), entity["label"], entity["start"], entity["end"])
            for entity in gold_entities
        }
        pred_exact = {
            (entity["vi_term"].lower(), entity["label"], entity["start"], entity["end"])
            for entity in pred_entities
        }

        em_tp += len(gold_exact & pred_exact)
        em_fp += len(pred_exact - gold_exact)
        em_fn += len(gold_exact - pred_exact)

        matched_gold_indices = set()
        matched_pred_indices = set()
        for pred_idx, pred in enumerate(pred_entities):
            for gold_idx, gold in enumerate(gold_entities):
                if pred["label"] != gold["label"]:
                    continue
                if calculate_iou(pred["start"], pred["end"], gold["start"], gold["end"]) >= iou_threshold:
                    matched_pred_indices.add(pred_idx)
                    matched_gold_indices.add(gold_idx)

        pm_tp += len(matched_pred_indices)
        pm_fp += len(pred_entities) - len(matched_pred_indices)
        pm_fn += len(gold_entities) - len(matched_gold_indices)

    return {
        "strict": calculate_metrics(em_tp, em_fp, em_fn),
        "relaxed": calculate_metrics(pm_tp, pm_fp, pm_fn),
        "iou_threshold": iou_threshold,
    }


def clean_and_evaluate(
    dataset_name: str,
    gold_file: Path | str,
    pred_file: Path | str,
    output_file: Path | str,
    iou_threshold: float = 0.5,
) -> dict:
    dataset_name = ensure_dataset_name(dataset_name)
    gold_rows = read_jsonl(gold_file)
    pred_rows = read_jsonl(pred_file)
    limit = min(len(gold_rows), len(pred_rows))
    labels = LABELS[dataset_name]

    cleaned_rows = [
        clean_prediction_row(gold_rows[i], pred_rows[i], allowed_labels=labels)
        for i in range(limit)
    ]
    write_jsonl(cleaned_rows, output_file)

    metrics = evaluate_cleaned_rows(cleaned_rows, iou_threshold=iou_threshold)
    metrics.update(
        {
            "dataset": dataset_name,
            "gold_file": str(gold_file),
            "pred_file": str(pred_file),
            "cleaned_file": str(output_file),
            "sentences_used": limit,
        }
    )
    return metrics


def format_metrics_report(report: dict) -> str:
    strict = report["strict"]
    relaxed = report["relaxed"]
    threshold = report.get("iou_threshold", 0.5)
    return "\n".join(
        [
            f"Dataset: {report.get('dataset', '')}",
            f"Sentences used: {report.get('sentences_used', 0)}",
            "Strict exact match:",
            f"  TP={strict['tp']} FP={strict['fp']} FN={strict['fn']}",
            f"  P={strict['precision']:.4f} R={strict['recall']:.4f} F1={strict['f1']:.4f}",
            f"Relaxed IoU >= {threshold:.2f}:",
            f"  TP={relaxed['tp']} FP={relaxed['fp']} FN={relaxed['fn']}",
            f"  P={relaxed['precision']:.4f} R={relaxed['recall']:.4f} F1={relaxed['f1']:.4f}",
        ]
    )
