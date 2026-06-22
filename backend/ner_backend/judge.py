"""Local Qwen LLM-as-a-judge evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import JUDGE_MODEL_NAME
from .io_utils import read_jsonl, write_json
from .postprocess import strip_code_fence
from .schema import extract_entities, get_sentence


JUDGE_KEYS = ("exact_match", "partial_match", "missing", "spurious", "type_error")


def load_judge_model(model_name: str = JUDGE_MODEL_NAME, load_in_4bit: bool = True):
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return model, tokenizer


def _model_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def call_qwen_judge(
    model,
    tokenizer,
    sentence: str,
    gold_entities: list[dict],
    pred_entities: list[dict],
    max_new_tokens: int = 1024,
) -> str:
    messages = [
        {
            "role": "system",
            "content": """Bạn là chuyên gia y tế và ngôn ngữ học làm giám khảo (Evaluator) chấm điểm NER.
Dữ liệu đầu vào bao gồm danh sách thực thể Gold (Chuẩn) và Pred (Dự đoán). Mỗi thực thể có: vi_term, label, start, end.

Hãy so sánh và phân loại vào 5 nhóm sau. (ĐƯA TOÀN BỘ OBJECT CỦA THỰC THỂ VÀO MẢNG):
1. "exact_match": Thực thể Pred khớp hoàn toàn 'vi_term', 'label' và tọa độ (start/end) với Gold.
2. "partial_match": Thực thể Pred khớp 'label' và có giao nhau về tọa độ (start/end) với Gold, nhưng bị thiếu/thừa chữ ở 'vi_term'.
3. "type_error": Thực thể Pred khớp hoặc giao nhau về tọa độ, nhưng bị gán sai 'label'.
4. "spurious": Thực thể do Pred tự nhận diện, hoàn toàn không khớp hoặc giao nhau với bất kỳ thực thể Gold nào.
5. "missing": Thực thể có trong Gold nhưng Pred không tìm thấy (Hãy đưa thực thể của Gold vào nhóm này).

Lưu ý:
- Một câu có thể có nhiều thực thể trùng tên, BẮT BUỘC phải dùng 'start' và 'end' để đối chiếu vị trí.
- CHỈ TRẢ VỀ ĐỊNH DẠNG JSON SAU, KHÔNG GIẢI THÍCH HAY VIẾT THÊM BẤT CỨ CHỮ NÀO KHÁC:
{"exact_match": [], "partial_match": [], "missing": [], "spurious": [], "type_error": []}""",
        },
        {
            "role": "user",
            "content": (
                f'Câu văn: "{sentence}"\n\n'
                f"Thực Thể Chuẩn (Gold):\n{json.dumps(gold_entities, ensure_ascii=False)}\n\n"
                f"Thực Thể Dự Đoán (Pred):\n{json.dumps(pred_entities, ensure_ascii=False)}"
            ),
        },
    ]

    import torch

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt")
    device = _model_input_device(model)
    if device is not None:
        inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)


def compute_judge_metrics(global_stats: dict) -> dict:
    exact = global_stats["exact_match"]
    partial = global_stats["partial_match"]
    missing = global_stats["missing"]
    spurious = global_stats["spurious"]
    type_error = global_stats["type_error"]

    tp_relaxed = exact + partial
    fp_relaxed = spurious + type_error
    fn_relaxed = missing + type_error
    precision_relaxed = tp_relaxed / (tp_relaxed + fp_relaxed) if (tp_relaxed + fp_relaxed) else 0.0
    recall_relaxed = tp_relaxed / (tp_relaxed + fn_relaxed) if (tp_relaxed + fn_relaxed) else 0.0
    f1_relaxed = (
        2 * precision_relaxed * recall_relaxed / (precision_relaxed + recall_relaxed)
        if (precision_relaxed + recall_relaxed)
        else 0.0
    )

    tp_strict = exact
    fp_strict = spurious + type_error + partial
    fn_strict = missing + type_error + partial
    precision_strict = tp_strict / (tp_strict + fp_strict) if (tp_strict + fp_strict) else 0.0
    recall_strict = tp_strict / (tp_strict + fn_strict) if (tp_strict + fn_strict) else 0.0
    f1_strict = (
        2 * precision_strict * recall_strict / (precision_strict + recall_strict)
        if (precision_strict + recall_strict)
        else 0.0
    )

    return {
        "strict": {
            "precision": round(precision_strict, 4),
            "recall": round(recall_strict, 4),
            "f1": round(f1_strict, 4),
        },
        "relaxed": {
            "precision": round(precision_relaxed, 4),
            "recall": round(recall_relaxed, 4),
            "f1": round(f1_relaxed, 4),
        },
    }


def run_pipeline_local_judge(
    gold_path: Path | str,
    pred_path: Path | str,
    output_path: Path | str,
    model_name: str = JUDGE_MODEL_NAME,
    max_samples: int | None = None,
    load_in_4bit: bool = True,
) -> dict:
    gold_data = read_jsonl(gold_path)
    pred_data = read_jsonl(pred_path)
    limit = min(len(gold_data), len(pred_data))
    if max_samples is not None:
        limit = min(limit, max_samples)

    model, tokenizer = load_judge_model(model_name=model_name, load_in_4bit=load_in_4bit)
    start_time = time.time()
    all_results = []
    global_stats = {key: 0 for key in JUDGE_KEYS}

    for i in range(limit):
        gold_row = gold_data[i]
        pred_row = pred_data[i]
        sentence = get_sentence(gold_row) or get_sentence(pred_row)
        gold_entities = extract_entities(gold_row, role="gold")
        pred_entities = extract_entities(pred_row, role="pred")

        try:
            raw_output = call_qwen_judge(model, tokenizer, sentence, gold_entities, pred_entities)
            eval_result = json.loads(strip_code_fence(raw_output))
        except (json.JSONDecodeError, ValueError):
            print(f"Invalid JSON from judge at row {i + 1}; skipped.")
            continue
        except Exception as exc:
            print(f"Judge error at row {i + 1}: {exc}")
            continue

        for key in JUDGE_KEYS:
            global_stats[key] += len(eval_result.get(key, []))

        all_results.append(
            {
                "index": i,
                "sentence": sentence,
                "evaluation": eval_result,
            }
        )
        print(f"Judged {i + 1}/{limit}")

    final_report = {
        "summary_stats": global_stats,
        "metrics": compute_judge_metrics(global_stats),
        "detailed_results": all_results,
        "runtime_seconds": round(time.time() - start_time, 2),
    }
    write_json(final_report, output_path)
    return final_report
