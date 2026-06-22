import json
import re
from typing import Any, Dict


_JSON_RE = re.compile(r"\{.*\}", re.S)


def safe_json_loads(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {"entities": [], "raw": ""}

    try:
        return json.loads(raw)
    except Exception:
        match = _JSON_RE.search(raw)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {"entities": [], "raw": raw}


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def format_entities(entities):
    cleaned = []
    for item in entities or []:
        if isinstance(item, str):
            cleaned.append({"text": item.strip(), "label": "", "start": None, "end": None})
            continue

        if not isinstance(item, dict):
            continue

        text = item.get("text", item.get("term", item.get("entity", "")))
        label = item.get("label", item.get("type", ""))
        start = item.get("start", item.get("start_token_idx", item.get("begin", item.get("start_idx"))))
        end = item.get("end", item.get("end_token_idx", item.get("finish", item.get("end_idx"))))

        if not text and len(item) == 1:
            k, v = next(iter(item.items()))
            if k not in {"text", "term", "entity", "label", "type", "start", "end"}:
                label = k
                text = v

        cleaned.append({
            "text": str(text).strip(),
            "label": str(label).strip(),
            "start": start,
            "end": end,
        })
    return cleaned
