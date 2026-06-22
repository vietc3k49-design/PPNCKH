import re
from typing import List, Dict

from utils import normalize_spaces

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def split_sentences(text: str) -> List[str]:
    text = normalize_spaces(text)
    if not text:
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return parts or [text]


def route_sentence(sentence: str) -> Dict[str, str]:
    s = normalize_spaces(sentence)
    lower = s.lower()
    question_mark = "?" in s or "？" in s
    question_words = ["bao nhiêu", "gì", "sao", "tại sao", "khi nào", "ở đâu", "như thế nào", "liệu", "có nên", "có phải"]
    is_question = question_mark or any(word in lower for word in question_words)
    adapter = "vimq" if is_question else "vimedner"
    return {
        "sentence": s,
        "adapter": adapter,
        "route_label": "Câu hỏi y khoa" if adapter == "vimq" else "Dữ liệu bệnh lý y khoa",
    }


def route_text(text: str) -> List[Dict[str, str]]:
    return [route_sentence(sentence) for sentence in split_sentences(text)]
