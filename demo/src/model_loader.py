from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from config import ADAPTERS, BASE_MODEL_NAME, VIMEDNER_ADAPTER_PATH, VIMQ_ADAPTER_PATH


@lru_cache(maxsize=1)
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


@lru_cache(maxsize=1)
def get_base_model():
    # Trên Windows/local, để tránh lỗi meta/offload của PEFT, load model ở dạng thường.
    # Nếu chạy trên Colab/GPU mạnh hơn, có thể tối ưu lại sau.
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )


def _load_adapter_on_cpu(base_model, adapter_path: str):
    if not adapter_path:
        return base_model
    return PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
    )


@lru_cache(maxsize=2)
def load_adapter(adapter_name: str):
    if adapter_name not in ADAPTERS:
        raise ValueError(f"Unknown adapter: {adapter_name}")

    base_model = get_base_model()
    adapter_path = VIMEDNER_ADAPTER_PATH if adapter_name == "vimedner" else VIMQ_ADAPTER_PATH
    model = _load_adapter_on_cpu(base_model, adapter_path)
    model.eval()
    return model
