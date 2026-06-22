# NER Y tế Demo

## Chạy local

```bash
pip install -r requirements.txt
python app.py
```

## Ghi chú

- Cần cập nhật `adapter_path` trong `src/config.py` để trỏ tới thư mục adapter thực tế.
- Demo hiện dùng router đơn giản để phân luồng câu sang `vimedner` hoặc `vimq`.
- Nếu chạy trên Colab/GPU, mô hình sẽ ưu tiên 4-bit quantization.
