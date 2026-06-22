from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent

# demo/src/config.py

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Adapter cho tác vụ NER
VIMEDNER_ADAPTER_PATH = r"E:\ppnckh\qwen2p5_3b_vimedner_ner_lora"

# Adapter cho tác vụ NMQ / phân loại câu hỏi
VIMQ_ADAPTER_PATH = r"E:\ppnckh\qwen2p5_3b_vimq_ner_lora"


ADAPTERS = {
    "vimedner": {
        "label": "Dữ liệu bệnh lý y khoa",
        "labels": ("DISEASE", "SYMPTOM", "TREATMENT", "CAUSE", "DIAGNOSTIC"),
        "system_name": "vimedner",
        "adapter_path": VIMEDNER_ADAPTER_PATH,
    },
    "vimq": {
        "label": "Câu hỏi y khoa",
        "labels": ("SYMPTOM_AND_DISEASE", "MEDICAL_PROCEDURE", "DRUG"),
        "system_name": "vimq",
        "adapter_path": VIMQ_ADAPTER_PATH,
    },
}

DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
