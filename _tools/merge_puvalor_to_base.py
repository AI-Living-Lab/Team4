"""
Merge PU-VALOR LoRA (checkpoint-12635) into base model.
Saves a self-contained model that can be loaded as normal base for further LoRA training.
"""
import os, sys
sys.path.insert(0, '/home/aix23102/audiolm/video-SALMONN-2/video_SALMONN2_plus')

import torch
from peft import PeftModel
from safetensors.torch import load_file
from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus

BASE = "/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens"
LORA_CKPT = "/data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_0.3ep_lora_timetoken/checkpoint-12635"
OUT = "/data0/aix23102/checkpoints_open_aligner/puvalor_merged_base"

os.makedirs(OUT, exist_ok=True)

print(f"[1/5] Loading base model from {BASE}")
model = video_SALMONN2_plus.from_pretrained(
    BASE,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

print(f"[2/5] Preparing to load PEFT (detach audio.layers)")
audio_layers = model.audio.layers
del model.audio.layers

print(f"[3/5] Loading PEFT adapter from {LORA_CKPT}")
model = PeftModel.from_pretrained(model, LORA_CKPT)
model.model.audio.layers = audio_layers

print(f"[4/5] Merging LoRA + modules_to_save into base")
model = model.merge_and_unload()

# Sanity check: are any non-LoRA weights left in adapter file that we need to load?
# With modules_to_save=[embed_tokens, lm_head], PeftModel already handled them
adapter_path = os.path.join(LORA_CKPT, "adapter_model.safetensors")
if os.path.exists(adapter_path):
    adapter_weights = load_file(adapter_path)
    non_lora = {k.replace("base_model.model.", "", 1): v
                for k, v in adapter_weights.items()
                if "lora" not in k and "modules_to_save" not in k}
    if non_lora:
        missing, unexpected = model.load_state_dict(non_lora, strict=False)
        print(f"[INFO] Loaded {len(non_lora)} extra non-LoRA weights "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print("[INFO] No extra non-LoRA weights to load")

print(f"[5/5] Saving merged model to {OUT}")
model.save_pretrained(OUT, safe_serialization=True)

# Copy tokenizer files too
import shutil
for f in ["added_tokens.json", "merges.txt", "special_tokens_map.json",
          "tokenizer_config.json", "vocab.json", "preprocessor_config.json"]:
    src = os.path.join(BASE, f)
    if os.path.exists(src):
        shutil.copy(src, OUT)
        print(f"  Copied {f}")

print(f"[DONE] Merged model at {OUT}")
