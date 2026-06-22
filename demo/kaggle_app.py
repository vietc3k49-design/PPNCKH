import os
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kaggle_config import ADAPTERS, BASE_MODEL_NAME, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P  # noqa: E402
from text_processing import split_sentences  # noqa: E402
from utils import format_entities, normalize_spaces, safe_json_loads  # noqa: E402

TITLE = "Hệ thống trích xuất thực thể demo"

LABEL_VI = {
    "DISEASE": "bệnh/tình trạng bệnh",
    "SYMPTOM": "triệu chứng",
    "TREATMENT": "điều trị/phương pháp",
    "CAUSE": "nguyên nhân",
    "DIAGNOSTIC": "chẩn đoán/xét nghiệm",
    "SYMPTOM_AND_DISEASE": "triệu chứng/bệnh",
    "MEDICAL_PROCEDURE": "thủ thuật y khoa",
    "DRUG": "thuốc",
}

LABEL_COLORS = {
    "DISEASE": "#dbeafe",
    "SYMPTOM": "#fee2e2",
    "TREATMENT": "#dcfce7",
    "CAUSE": "#fef3c7",
    "DIAGNOSTIC": "#ede9fe",
    "SYMPTOM_AND_DISEASE": "#fce7f3",
    "MEDICAL_PROCEDURE": "#ccfbf1",
    "DRUG": "#e0f2fe",
}

LABEL_TEXT = {
    "DISEASE": "#1e3a8a",
    "SYMPTOM": "#7f1d1d",
    "TREATMENT": "#14532d",
    "CAUSE": "#78350f",
    "DIAGNOSTIC": "#4c1d95",
    "SYMPTOM_AND_DISEASE": "#831843",
    "MEDICAL_PROCEDURE": "#134e4a",
    "DRUG": "#0c4a6e",
}

CARD_BG = {
    "DISEASE": "#f8fbff",
    "SYMPTOM": "#fff7f7",
    "TREATMENT": "#f7fdf8",
    "CAUSE": "#fffaf0",
    "DIAGNOSTIC": "#faf7ff",
    "SYMPTOM_AND_DISEASE": "#fff7fb",
    "MEDICAL_PROCEDURE": "#f6fffd",
    "DRUG": "#f7fcff",
}


def html_escape(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


@lru_cache(maxsize=1)
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


@lru_cache(maxsize=1)
def load_base_model():
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )


@lru_cache(maxsize=2)
def load_adapter(adapter_name: str):
    if adapter_name not in ADAPTERS:
        raise ValueError(f"Unknown adapter: {adapter_name}")
    base = load_base_model()
    path = ADAPTERS[adapter_name]["adapter_path"]
    model = PeftModel.from_pretrained(base, path, is_trainable=False)
    model.eval()
    return model


def build_prompt(adapter_name: str, sentence: str):
    labels = ADAPTERS[adapter_name]["labels"]
    if adapter_name == "vimedner":
        system_prompt = (
            "Bạn là hệ thống nhận dạng và trích xuất thực thể y tế (NER).\n"
            f"Chỉ được dùng đúng các nhãn sau: {', '.join(labels)}\n"
            "Trả về DUY NHẤT JSON hợp lệ với key entities."
        )
        user_prompt = f"Câu:\n{sentence}\n\nTrả về JSON có dạng: {{\"entities\":[]}}"
    else:
        system_prompt = (
            "Bạn là hệ thống phân tích câu hỏi y khoa.\n"
            f"Chỉ được dùng đúng các nhãn sau: {', '.join(labels)}\n"
            "Trả về DUY NHẤT JSON hợp lệ."
        )
        user_prompt = f"Câu:\n{sentence}\n\nTrả về JSON theo mẫu: {{\"entities\":[]}}"
    return system_prompt, user_prompt


def generate_one(sentence: str, adapter_name: str):
    tokenizer = load_tokenizer()
    model = load_adapter(adapter_name)
    system_prompt, user_prompt = build_prompt(adapter_name, sentence)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
        )

    decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    payload = safe_json_loads(decoded)
    entities = format_entities(payload.get("entities", []))
    for entity in entities:
        label = entity.get("label", "")
        entity["label_vi"] = LABEL_VI.get(label, label)
    payload["entities"] = entities
    return payload


def route_sentence(sentence: str):
    return "vimq" if "?" in sentence else "vimedner"


def render_entity_card(entity):
    label = entity.get("label", "")
    label_color = LABEL_COLORS.get(label, "#e5e7eb")
    label_text = LABEL_TEXT.get(label, "#111827")
    bg = CARD_BG.get(label, "#ffffff")
    return f'''
    <div class="entity-card" style="background:{bg}; border-color:{label_color};">
        <div class="entity-title" style="color:{label_text};">{html_escape(entity.get("text", ""))}</div>
        <div class="entity-badges">
            <span class="badge" style="background:{label_color}; color:{label_text};">{html_escape(label)}</span>
            <span class="badge vi">{html_escape(entity.get("label_vi", label))}</span>
        </div>
    </div>
    '''


def render_sentence_block(row):
    entities_html = "".join(render_entity_card(e) for e in row["entities"])
    if not entities_html:
        entities_html = "<div class='empty-entities'>Không có entity nào.</div>"
    return f'''
    <div class="sentence-block">
        <div class="sentence-head">
            <div><b>Câu {row["index"]}</b> — <span class="adapter">{html_escape(row["classification_label"])}</span></div>
            <div class="count">{len(row["entities"])} entity</div>
        </div>
        <div class="sentence-text">{html_escape(row["sentence"])} </div>
        <div class="entities-wrap">{entities_html}</div>
    </div>
    '''


def render_total_table(rows):
    items = []
    idx = 1
    for row in rows:
        for e in row["entities"]:
            label = e.get("label", "")
            color = LABEL_COLORS.get(label, "#e5e7eb")
            items.append(
                f"<tr><td>{idx}</td><td>{html_escape(e.get('text', ''))}</td><td><span class='badge' style='background:{color};'>{html_escape(label)}</span></td><td>{html_escape(e.get('label_vi', label))}</td><td>{html_escape(row['index'])}</td></tr>"
            )
            idx += 1

    if not items:
        return "<div class='empty-entities'>Chưa có entity nào.</div>"

    return f'''
    <div class="table-wrap">
        <table class="total-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Thực thể</th>
                    <th>Nhãn</th>
                    <th>Nhãn tiếng Việt</th>
                    <th>Câu</th>
                </tr>
            </thead>
            <tbody>
                {''.join(items)}
            </tbody>
        </table>
    </div>
    '''


def render_sentence_tabs(rows):
    if not rows:
        return "<div class='empty-entities'>Chưa có câu nào.</div>"

    tabs = []
    panels = []
    radios = []
    for i, row in enumerate(rows, start=1):
        tab_id = f"tab-{i}"
        tabs.append(f"<label for='{tab_id}' class='tab-label'>Câu {row['index']}</label>")
        radios.append(f"<input type='radio' name='sent-tabs' id='{tab_id}' {'checked' if i == 1 else ''}>")
        panels.append(f'''<div class="tab-panel">{render_sentence_block(row)}</div>''')

    return f'''
    <div class="tabs-wrap">
        {''.join(radios)}
        <div class="tab-bar">{''.join(tabs)}</div>
        <div class="tab-panels">{''.join(panels)}</div>
    </div>
    '''


def process_text(text: str):
    text = normalize_spaces(text)
    if not text:
        return "Vui lòng nhập văn bản.", [], "", ""

    sentences = split_sentences(text)
    rows = []
    for i, sent in enumerate(sentences, start=1):
        adapter = route_sentence(sent)
        payload = generate_one(sent, adapter)
        entities = []
        for e in payload.get("entities", []):
            label = str(e.get("label", "")).strip()
            entities.append({
                "text": e.get("text", ""),
                "label": label,
                "label_vi": e.get("label_vi", LABEL_VI.get(label, label)),
            })
        rows.append({
            "index": i,
            "sentence": sent,
            "adapter": adapter,
            "classification_label": ADAPTERS[adapter]["label"],
            "entities": entities,
        })

    summary = (
        f"### Kết quả\n"
        f"- Số câu: **{len(rows)}**\n"
        f"- Dữ liệu bệnh lý y khoa: **{sum(r['adapter'] == 'vimedner' for r in rows)}**\n"
        f"- Dữ liệu câu hỏi y khoa: **{sum(r['adapter'] == 'vimq' for r in rows)}**"
    )
    total_html = render_total_table(rows)
    tabs_html = render_sentence_tabs(rows)
    table_rows = [[r["index"], r["sentence"], r["classification_label"], len(r["entities"])] for r in rows]
    return summary, table_rows, total_html, tabs_html


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")

    text = gr.Textbox(label="Text đầu vào", lines=8)
    run = gr.Button("Chạy")
    summary = gr.Markdown()
    table = gr.Dataframe(headers=["#", "Câu", "Phân loại", "Số entity"], interactive=False)
    total_output = gr.HTML(label="Bảng tổng entity")
    tabs_output = gr.HTML(label="Câu + entity theo tab")

    run.click(process_text, inputs=[text], outputs=[summary, table, total_output, tabs_output])

    gr.HTML(
        """
        <style>
        .sentence-block{border:1px solid #dbe4f0;border-radius:14px;padding:14px 16px;margin:12px 0;background:#fff}
        .sentence-head{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:8px}
        .adapter{background:#eef2ff;color:#1d4ed8;padding:3px 8px;border-radius:999px;font-size:12px}
        .count{color:#475569;font-weight:600}
        .sentence-text{margin:8px 0 12px 0;line-height:1.6}
        .entities-wrap{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
        .entity-card{border:1px solid #dbe4f0;border-radius:12px;padding:12px;background:#fff}
        .entity-title{font-weight:700;margin-bottom:10px;color:#111827}
        .entity-badges{display:flex;gap:8px;flex-wrap:wrap}
        .badge{display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:700;border:1px solid rgba(0,0,0,.08)}
        .badge.vi{background:#e5e7eb;color:#111827}
        .empty-entities{padding:10px 12px;background:#f3f4f6;border-radius:10px;color:#6b7280}
        .table-wrap{overflow:auto;border:1px solid #dbe4f0;border-radius:12px}
        .total-table{width:100%;border-collapse:collapse;background:#fff}
        .total-table th,.total-table td{padding:10px 12px;border-bottom:1px solid #dbe4f0;text-align:left;vertical-align:top}
        .total-table th{background:#f8fbff;position:sticky;top:0;z-index:1}
        .tabs-wrap{margin-top:8px}
        .tabs-wrap input[type='radio']{display:none}
        .tab-bar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
        .tab-label{cursor:pointer;padding:8px 12px;border-radius:999px;background:#eef2ff;color:#1d4ed8;font-weight:700}
        .tab-panel{display:none}
        .tabs-wrap input[type='radio']#tab-1:checked ~ .tab-panels .tab-panel:nth-child(1),
        .tabs-wrap input[type='radio']#tab-2:checked ~ .tab-panels .tab-panel:nth-child(2),
        .tabs-wrap input[type='radio']#tab-3:checked ~ .tab-panels .tab-panel:nth-child(3),
        .tabs-wrap input[type='radio']#tab-4:checked ~ .tab-panels .tab-panel:nth-child(4),
        .tabs-wrap input[type='radio']#tab-5:checked ~ .tab-panels .tab-panel:nth-child(5),
        .tabs-wrap input[type='radio']#tab-6:checked ~ .tab-panels .tab-panel:nth-child(6),
        .tabs-wrap input[type='radio']#tab-7:checked ~ .tab-panels .tab-panel:nth-child(7),
        .tabs-wrap input[type='radio']#tab-8:checked ~ .tab-panels .tab-panel:nth-child(8),
        .tabs-wrap input[type='radio']#tab-9:checked ~ .tab-panels .tab-panel:nth-child(9),
        .tabs-wrap input[type='radio']#tab-10:checked ~ .tab-panels .tab-panel:nth-child(10){display:block}
        </style>
        """
    )

if __name__ == "__main__":
    demo.launch(share=True)
