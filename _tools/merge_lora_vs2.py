import os
import torch
from llava.model.utils import load_qwen_lora_model

BASE_MODEL = os.environ.get('BASE_MODEL', 'lmms-lab/llava-onevision-qwen2-7b-ov')
LORA_CKPT  = os.environ.get('SFT_CKPT', 'checkpoints/finetuning_test/all_parameters.bin')
OUT_DIR    = os.environ.get('MERGE_OUT_DIR', 'checkpoints/finetuning_test/merged')

model, tokenizer = load_qwen_lora_model(
    model_path=LORA_CKPT,
    model_base=BASE_MODEL,
    lora_enable=True,
    load_full=False,

    # ✅ 학습과 동일하게 맞추기 (중요)
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.05,

    model_max_length=4096,
)

print("[INFO] Merging LoRA...")
model = model.merge_and_unload()

model.to(torch.bfloat16)
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("[DONE] saved:", OUT_DIR)
