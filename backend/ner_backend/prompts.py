"""Prompt and response formatting for SFT training and inference."""

from __future__ import annotations

import json
from typing import Iterable


def build_system_prompt(dataset_name: str, labels: Iterable[str]) -> str:
    label_text = ", ".join(labels)
    return f"""Bạn là hệ thống nhận dạng và trích xuất thực thể y tế (NER).
Dataset/schema: {dataset_name}
Ngôn ngữ đầu vào: tiếng Việt

Chỉ được dùng đúng các nhãn sau:
{label_text}

Quy tắc bắt buộc:
1) Trả về DUY NHẤT 1 JSON object hợp lệ, không dùng markdown code block (không bao bọc bởi ```json), không giải thích gì thêm.
2) JSON phải có đúng key "entities".
3) Mỗi entity gồm đúng 4 key: text, label, start, end.
4) start/end là chỉ số token (tính theo việc tách câu bằng khoảng trắng).
5) end là vị trí kết thúc INCLUSIVE.
6) Nếu không có thực thể, trả về: {{"entities":[]}}"""


def build_messages(dataset_name: str, labels: Iterable[str], text: str) -> list[dict]:
    return [
        {"role": "system", "content": build_system_prompt(dataset_name, labels)},
        {"role": "user", "content": text},
    ]


def format_response(entities: Iterable[dict]) -> str:
    payload = {
        "entities": [
            {
                "text": e["term"],
                "label": e["label"],
                "start": e["start_token_idx"],
                "end": e["end_token_idx"],
            }
            for e in entities
        ]
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_prompt_text(tokenizer, dataset_name: str, labels: Iterable[str], text: str) -> str:
    messages = build_messages(dataset_name, labels, text)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
