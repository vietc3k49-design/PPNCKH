import json

import torch

from config import ADAPTERS, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from model_loader import get_tokenizer, load_adapter
from utils import format_entities, safe_json_loads


def build_system_prompt(adapter_name: str) -> str:
    labels = ADAPTERS[adapter_name]["labels"]
    return f'''Bạn là hệ thống nhận dạng và trích xuất thực thể y tế (NER).
Dataset/schema: {adapter_name}
Ngôn ngữ đầu vào: tiếng Việt

Chỉ được dùng đúng các nhãn sau:
{', '.join(labels)}

Quy tắc bắt buộc:
1) Trả về DUY NHẤT 1 JSON object hợp lệ, không dùng markdown code block, không giải thích gì thêm.
2) JSON phải có đúng key "entities".
3) Mỗi entity gồm đúng 4 key: text, label, start, end.
4) start/end là chỉ số token (tính theo việc tách câu bằng khoảng trắng).
5) end là vị trí kết thúc INCLUSIVE.
6) Nếu không có thực thể, trả về: {{"entities":[]}}'''


def build_messages(adapter_name: str, text: str):
    return [
        {"role": "system", "content": build_system_prompt(adapter_name)},
        {"role": "user", "content": text},
    ]


def generate_for_sentence(sentence: str, adapter_name: str, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    tokenizer = get_tokenizer()
    model = load_adapter(adapter_name)
    messages = build_messages(adapter_name, sentence)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    payload = safe_json_loads(decoded)
    payload["entities"] = format_entities(payload.get("entities", []))
    return payload, decoded
