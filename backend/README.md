# Vietnamese Medical NER Backend

Dự án này tách phần code trong notebook `final_code.ipynb` thành các module Python nhỏ hơn để dễ đọc, dễ chạy lại và dễ đưa lên GitHub. Pipeline chính phục vụ bài toán nhận dạng thực thể y tế tiếng Việt cho hai bộ dữ liệu `vimedner` và `vimq`.

## Cấu trúc thư mục

```text
backend/
  final_code.ipynb
  requirements.txt
  ner_backend/
    config.py          # Cấu hình đường dẫn, nhãn, tên model
    data.py            # Đọc dữ liệu JSONL và tạo SFT records
    prompts.py         # Prompt hệ thống và định dạng JSON response
    training.py        # Huấn luyện LoRA adapter
    inference.py       # Nạp base model + LoRA adapter và chạy dự đoán
    postprocess.py     # Hậu xử lý dự đoán, tính exact/IoU metrics
    seqeval_eval.py    # Xuất BIO/CoNLL và đánh giá bằng seqeval
    judge.py           # Đánh giá bằng Qwen local theo kiểu LLM-as-a-judge
    describe_data.py   # Thống kê dữ liệu
    plots.py           # Vẽ biểu đồ từ báo cáo đánh giá
    cli.py             # Giao diện dòng lệnh
```

## Cài đặt

Nên chạy train/inference trên Kaggle, Colab hoặc Linux có GPU vì model Qwen và `bitsandbytes` khá nặng.

```bash
cd backend
pip install -r requirements.txt
```

Trên Windows hoặc máy CPU, bạn vẫn có thể chạy các lệnh nhẹ như thống kê dữ liệu, hậu xử lý, clean prediction và seqeval. Các bước train, inference và LLM-as-a-judge cần GPU để chạy ổn định.

## Dữ liệu

Mặc định code tìm dữ liệu theo cấu trúc thư mục của repo hiện tại:

```text
vimedner/train_fixed_verified.jsonl
vimedner/dev_fixed_verified.jsonl
vimedner/test_fixed_verified.jsonl
vimq/train_fixed_verified.jsonl
vimq/dev_fixed_verified.jsonl
vimq/test_fixed_verified.jsonl
```

Nếu dữ liệu nằm ở nơi khác, truyền thêm `--data-root` hoặc đặt biến môi trường `NER_DATA_ROOT`.

## Luồng chạy chính

Luồng đầy đủ thường là:

```text
describe-data -> train -> infer -> clean-eval -> seqeval hoặc judge
```

Trong đó:

- `describe-data`: kiểm tra và thống kê dữ liệu.
- `train`: huấn luyện LoRA adapter cho từng dataset.
- `infer`: chạy dự đoán trên tập test.
- `clean-eval`: lọc dự đoán lỗi/ảo giác và tính strict/relaxed metrics.
- `seqeval`: chuyển dữ liệu sang BIO/CoNLL và tính điểm bằng seqeval.
- `judge`: dùng Qwen local làm giám khảo để phân loại exact, partial, missing, spurious và type error.

## Các lệnh thường dùng

Thống kê dữ liệu:

```bash
python -m ner_backend.cli describe-data --data-root ..
```

Huấn luyện adapter cho `vimedner`:

```bash
python -m ner_backend.cli train \
  --dataset vimedner \
  --data-root .. \
  --output-root ../output/ner_adapters
```

Huấn luyện adapter cho `vimq`:

```bash
python -m ner_backend.cli train \
  --dataset vimq \
  --data-root .. \
  --output-root ../output/ner_adapters
```

Copy checkpoint đã upload vào đúng thư mục để resume training:

```bash
python -m ner_backend.cli copy-checkpoint \
  --dataset vimq \
  --checkpoint-dir /kaggle/input/.../checkpoint-600 \
  --output-root /kaggle/working/ner_adapters
```

Chạy inference trên tập test:

```bash
python -m ner_backend.cli infer \
  --dataset vimq \
  --data-root .. \
  --adapter-path ../output/ner_adapters/qwen2p5_3b_vimq_ner_lora \
  --output-file ../output/predictions/vimq_test_results_full.jsonl
```

Hậu xử lý prediction và tính exact/IoU metrics:

```bash
python -m ner_backend.cli clean-eval \
  --dataset vimq \
  --data-root .. \
  --pred-file ../output/predictions/vimq_test_results_full.jsonl \
  --output-file ../output/predictions/vimq_cleaned.jsonl
```

Chạy seqeval trên file đã clean:

```bash
python -m ner_backend.cli seqeval \
  --name vimq \
  --gold-file ../output/predictions/vimq_cleaned.jsonl \
  --pred-file ../output/predictions/vimq_cleaned.jsonl \
  --output-dir ../output/seqeval
```

File clean do `clean-eval` tạo ra đã chứa cả `gold` và `pred`, vì vậy lệnh `seqeval` có thể dùng cùng một file cho hai tham số `--gold-file` và `--pred-file`.

Chạy LLM-as-a-judge local:

```bash
python -m ner_backend.cli judge \
  --gold-file ../output/predictions/vimq_cleaned.jsonl \
  --pred-file ../output/predictions/vimq_cleaned.jsonl \
  --output-file ../output/vimq_llm_evaluation_results.json
```

Khi test nhanh phần judge, nên thêm `--max-samples 20` để tránh tốn nhiều thời gian GPU.

## Ghi chú

- Notebook gốc `final_code.ipynb` vẫn được giữ lại để đối chiếu.
- Các lệnh train/inference có thể tải model từ Hugging Face, nên cần mạng và đủ dung lượng bộ nhớ GPU.
- Kết quả trung gian nên lưu trong thư mục `output/` để tránh đưa nhầm model/checkpoint lớn lên GitHub.
