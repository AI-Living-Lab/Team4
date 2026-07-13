"""
Merge salmonn2plus_v8_chronus / checkpoint-250 LoRA into base model.
merge_v8_unav_to_base.py 의 chronus 변형 — 경로만 랩/ chronus SFT 출력으로 교체.
Produces a self-contained merged base usable as natural RL 의 model_base.

  v8 stores audio.q_tokens in a SEPARATE file (audio_q_tokens.safetensors); load explicitly.
"""
import os, sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'video_SALMONN2_plus'))

import torch
from peft import PeftModel
from safetensors.torch import load_file
from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus

BASE = "/home/team404/workspace/checkpoints/base/video_salmonn2_plus_7B_full"
LORA_CKPT = "/home/team404/workspace/checkpoints/sft/salmonn2plus_v8_chronus/checkpoint-250"
OUT = "/home/team404/workspace/checkpoints/base/salmonn2p_7b_unav_v8_chronus"

assert not os.path.exists(OUT), f"OUT already exists: {OUT}"
os.makedirs(OUT, exist_ok=True)

print(f"[1/6] Loading base model from {BASE}")
model = video_SALMONN2_plus.from_pretrained(
    BASE,
    attn_implementation="sdpa",   # CPU 머지 — forward 없음, flash_attn 회피
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

print(f"[2/6] Detaching audio.layers (frozen encoder, not in adapter)")
audio_layers = model.audio.layers
del model.audio.layers

print(f"[3/6] Loading PEFT adapter from {LORA_CKPT}")
model = PeftModel.from_pretrained(model, LORA_CKPT)
model.model.audio.layers = audio_layers

print(f"[4/6] Merging LoRA + modules_to_save into base")
model = model.merge_and_unload()

# modules_to_save members (visual.merger, audio.qformer, audio.audio_proj,
# model.embed_tokens, lm_head) are stored as PLAIN keys in adapter_model.safetensors
# (no `modules_to_save.default` prefix), so PeftModel.from_pretrained does NOT load
# them into the wrappers. Load them here via strict=False.
adapter_path = os.path.join(LORA_CKPT, "adapter_model.safetensors")
adapter_weights = load_file(adapter_path)
non_lora = {k.replace("base_model.model.", "", 1): v
            for k, v in adapter_weights.items()
            if "lora" not in k and "modules_to_save" not in k}
missing, unexpected = model.load_state_dict(non_lora, strict=False)
print(f"[5/6] Loaded {len(non_lora)} non-LoRA weights "
      f"(state_dict missing={len(missing)}, unexpected={len(unexpected)})")
assert len(unexpected) == 0, f"Unexpected keys when loading non-LoRA: {unexpected[:10]}"

# audio.q_tokens lives in a SEPARATE file in v8 — load it explicitly.
q_path = os.path.join(LORA_CKPT, "audio_q_tokens.safetensors")
assert os.path.exists(q_path), f"missing q_tokens file: {q_path}"
q_state = load_file(q_path)  # {'audio.q_tokens': tensor[1,1,3072]}
qm, qu = model.load_state_dict(q_state, strict=False)
assert len(qu) == 0, f"Unexpected q_tokens keys: {qu}"
assert "audio.q_tokens" not in qm, "audio.q_tokens key did not match model state_dict!"
print(f"  Loaded q_tokens {list(q_state.keys())} shape "
      f"{[tuple(v.shape) for v in q_state.values()]}")

# Sanity: confirm trained q_tokens actually differs from the (uninitialized) base value.
with torch.no_grad():
    print(f"  q_tokens norm after load: {model.audio.q_tokens.norm().item():.4f}")

print(f"[6/6] Saving merged model to {OUT}")
model.save_pretrained(OUT, safe_serialization=True)

import shutil
for f in ["added_tokens.json", "merges.txt", "special_tokens_map.json",
          "tokenizer_config.json", "vocab.json", "preprocessor_config.json",
          "tokenizer.json", "chat_template.json", "generation_config.json"]:
    src = os.path.join(BASE, f)
    if os.path.exists(src):
        shutil.copy(src, OUT)
        print(f"  Copied {f}")

print(f"[DONE] Merged chronus model at {OUT}")
