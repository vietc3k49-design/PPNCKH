import json

import gradio as gr

from src.inference import generate_for_sentence
from src.text_processing import route_text
from src.utils import normalize_spaces


def run_demo(text):
    text = normalize_spaces(text)
    if not text:
        return [], json.dumps({"error": "Vui lòng nhập văn bản y tế."}, ensure_ascii=False, indent=2), []

    routed = route_text(text)
    results = []
    table_rows = []

    for idx, item in enumerate(routed, start=1):
        sentence = item["sentence"]
        adapter = item["adapter"]
        route_label = item["route_label"]
        payload, raw = generate_for_sentence(sentence, adapter)
        result_item = {
            "index": idx,
            "sentence": sentence,
            "route": route_label,
            "adapter": adapter,
            "prediction": payload,
            "raw_output": raw,
        }
        results.append(result_item)
        table_rows.append([idx, route_label, adapter, sentence, len(payload.get("entities", []))])

    pretty_json = json.dumps({"results": results}, ensure_ascii=False, indent=2)
    raw_concat = "\n\n".join([f"[{item['index']}] {item['sentence']}\n{item['raw_output']}" for item in results])
    return table_rows, pretty_json, results, raw_concat


def build_app():
    with gr.Blocks(title="NER Y tế Demo") as demo:
        gr.Markdown("# Demo NER y tế tiếng Việt")
        gr.Markdown("Nhập một đoạn văn bản y tế, hệ thống sẽ tách câu, phân luồng sang vimedner hoặc vimq, rồi trả kết quả NER.")

        with gr.Row():
            input_text = gr.Textbox(label="Văn bản đầu vào", lines=10, placeholder="Nhập đoạn văn bản y tế ở đây...")

        run_button = gr.Button("Chạy demo", variant="primary")

        with gr.Row():
            sentence_table = gr.Dataframe(
                headers=["#", "Luồng", "Adapter", "Câu", "Số entity"],
                label="Kết quả phân luồng",
                interactive=False,
                wrap=True,
                type="array",
            )

        result_json = gr.Code(label="JSON kết quả", language="json")
        raw_details = gr.JSON(label="Chi tiết xử lý")
        raw_text = gr.Textbox(label="Raw output từng câu", lines=10)

        gr.Examples(
            examples=[
                ["Bệnh nhân bị đau đầu và sốt cao trong 3 ngày. Cần uống thuốc gì để giảm triệu chứng?"],
                ["Tôi bị ho kéo dài và đau họng, có cần đi khám không?"],
            ],
            inputs=[input_text],
            label="Ví dụ nhập nhanh",
        )

        run_button.click(
            fn=run_demo,
            inputs=[input_text],
            outputs=[sentence_table, result_json, raw_details, raw_text],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
