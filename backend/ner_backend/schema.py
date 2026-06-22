"""Schema-normalization helpers for gold, prediction, and cleaned JSONL rows."""

from __future__ import annotations

from typing import Iterable


def get_sentence(row: dict) -> str:
    return str(
        row.get("vi_sentence")
        or row.get("vi_sentence_str")
        or row.get("input_text")
        or row.get("sentence")
        or ""
    )


def _first_present(row: dict, keys: Iterable[str]):
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def normalize_entity(entity: dict, lang: str = "vi") -> dict | None:
    label = entity.get("label")
    term = _first_present(entity, (f"{lang}_term", "vi_term", "text", "term"))
    start = _first_present(entity, ("start", f"{lang}_start_token_idx", "start_token_idx"))
    end = _first_present(entity, ("end", f"{lang}_end_token_idx", "end_token_idx"))

    if label is None or term is None or start is None or end is None:
        return None

    try:
        start_i = int(start)
        end_i = int(end)
    except (TypeError, ValueError):
        return None

    term_s = str(term).strip()
    label_s = str(label).strip()
    if not term_s or not label_s or start_i < 0 or end_i < start_i:
        return None

    return {
        "label": label_s,
        "vi_term": term_s,
        "start": start_i,
        "end": end_i,
    }


def extract_entities(row: dict, role: str | None = None, allowed_labels: Iterable[str] | None = None) -> list[dict]:
    if role and isinstance(row.get(role), list):
        raw_entities = row[role]
    elif role == "pred" and isinstance(row.get("pred"), list):
        raw_entities = row["pred"]
    elif role == "gold" and isinstance(row.get("gold"), list):
        raw_entities = row["gold"]
    else:
        raw_entities = row.get("entities", []) or []

    allowed = set(allowed_labels) if allowed_labels is not None else None
    normalized: list[dict] = []
    for entity in raw_entities:
        item = normalize_entity(entity)
        if item is None:
            continue
        if allowed is not None and item["label"] not in allowed:
            continue
        normalized.append(item)

    normalized.sort(key=lambda x: (x["start"], x["end"], x["label"], x["vi_term"]))
    return normalized
