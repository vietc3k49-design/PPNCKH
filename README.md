# PPNCKH - Vietnamese Medical NER

Repo này chứa dữ liệu, notebook, backend Python và demo cho bài toán nhận dạng thực thể y tế tiếng Việt. Dự án tập trung vào hai bộ dữ liệu chính: `ViMedNER` và `ViMQ`.

## Cấu trúc thư mục

```text
ppnckh/
  backend/       # Code backend đã tách từ notebook final_code.ipynb
  demo/          # Code demo ứng dụng
  vimedner/      # Dữ liệu ViMedNER dạng JSONL
  vimq/          # Dữ liệu ViMQ dạng JSONL
  output/        # Kết quả chạy thử, báo cáo, file trung gian
```

## Backend

Phần backend nằm trong `backend/ner_backend/` và đã được tách thành các module nhỏ để dễ chạy lại:

- `config.py`: cấu hình model, nhãn và đường dẫn dữ liệu.
- `data.py`: đọc JSONL và tạo dữ liệu SFT.
- `training.py`: huấn luyện LoRA adapter.
- `inference.py`: chạy dự đoán bằng base model + adapter.
- `postprocess.py`: hậu xử lý dự đoán và tính exact/IoU metrics.
- `seqeval_eval.py`: đánh giá BIO bằng `seqeval`.
- `judge.py`: đánh giá bằng Qwen local theo kiểu LLM-as-a-judge.
- `cli.py`: giao diện dòng lệnh.

Xem hướng dẫn chi tiết trong `backend/README.md`.

## Cài đặt

Tạo môi trường Python riêng rồi cài dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Không nên đưa thư mục `.venv/` lên GitHub. File `.gitignore` ở root đã loại thư mục này.

## Lệnh chạy nhanh

Thống kê dữ liệu:

```bash
cd backend
python -m ner_backend.cli describe-data --data-root ..
```

Train adapter:

```bash
python -m ner_backend.cli train \
  --dataset vimq \
  --data-root .. \
  --output-root ../output/ner_adapters
```

Chạy inference:

```bash
python -m ner_backend.cli infer \
  --dataset vimq \
  --data-root .. \
  --adapter-path ../output/ner_adapters/qwen2p5_3b_vimq_ner_lora \
  --output-file ../output/predictions/vimq_test_results_full.jsonl
```

Hậu xử lý và đánh giá:

```bash
python -m ner_backend.cli clean-eval \
  --dataset vimq \
  --data-root .. \
  --pred-file ../output/predictions/vimq_test_results_full.jsonl \
  --output-file ../output/predictions/vimq_cleaned.jsonl
```

## Ghi chú khi đưa lên GitHub

- Dữ liệu trong `vimedner/` và `vimq/` hiện vẫn được giữ để có thể upload cùng repo.
- Nếu sau này có checkpoint/model lớn hơn giới hạn GitHub, nên chuyển sang Git LFS hoặc lưu ở Hugging Face/Kaggle Drive.
