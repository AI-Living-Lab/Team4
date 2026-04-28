"""Crab+ (Qwen2.5-Omni-7B + I-LoRA) inference runner — cdh 표준 contract 준수.

Usage:
  cd /workspace/jsy/Crab_Plus
  /workspace/miniconda3/envs/crab/bin/python /workspace/jsy/scripts/inference_crab_plus.py \
      --test_json /workspace/jsy/Crab_Plus/data_local/AVUIE_2/unav/test.json \
      --data_root /workspace/jsy/Crab_Plus/data_local \
      --base_model /workspace/models/Qwen2.5-Omni-7B \
      --lora_path /workspace/jsy/Crab_Plus/weight/finetune_weights.bin \
      --output_dir /workspace/jsy/outputs/base/CrabPlus/Unav100QA

(cwd = Crab_Plus/ 여야 함 — models.qwen2_5_omni / peft_hyper 등 상대 import)

출력 (cdh 표준):
  {output_dir}/test_results_rank0.json  (list, test JSON 순서 유지, 각 엔트리 'pred')
  {output_dir}/inference.log (bash 리다이렉트)

필드:
  row = {id, vid, gt_label, gt_segments, original_label, question, pred, error}
"""

import os
import sys
import argparse
import json
import time
import traceback
from pathlib import Path

# Crab+ root 를 sys.path 에 주입 (models / peft_hyper / dataset 등 로컬 import)
CRAB_ROOT = "/workspace/jsy/Crab_Plus"
if CRAB_ROOT not in sys.path:
    sys.path.insert(0, CRAB_ROOT)

import torch
from tqdm import tqdm
from transformers import Qwen2VLImageProcessor, WhisperFeatureExtractor, Qwen2TokenizerFast

from models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from dataset.qwen_omni_utils import process_mm_info


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
LEADING_TEXT = "This is a video and an audio:"


def build_conv(video_path: str, audio_path: str, question: str):
    """Crab+ native message layout 그대로, tail 텍스트만 our hybrid question 으로 교체."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": LEADING_TEXT},
                {
                    "type": "video",
                    "video": video_path,
                    "resized_height": 224,
                    "resized_width": 224,
                    "nframes": 10,
                },
                {"type": "audio", "audio": audio_path, "task": "unav"},
                {"type": "text", "text": question},
            ],
        },
    ]


def load_model(base_model_path: str, lora_path: str, lora_r: int = 128, lora_alpha: int = 256,
               lora_dropout: float = 0.10, lora_nums: int = 3):
    # flash-attn 시도 실패 — Qwen2.5-Omni rotary_emb 가 fp32 저장, flash-attn 2.7.4 kernel 이 cos/sin dtype
    # 엄격 체크 (AssertionError). flash-attn 2.8.3 은 torch 2.6+ wrap_triton 의존. 결국 sdpa 사용.
    print(f"[load] base: {base_model_path}  (sdpa)")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.disable_talker()
    model.config.use_cache = True
    thinker = model.thinker

    # Apply I-LoRA (peft_hyper) — wrap FIRST (creates fp32 LoRA layers), then cast whole model to bf16
    from peft_hyper import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_nums=lora_nums,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    thinker = get_peft_model(thinker, lora_config)

    print(f"[load] LoRA weights: {lora_path}")
    state_dict = torch.load(lora_path, map_location="cpu")
    missing, unexpected = thinker.load_state_dict(state_dict, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}  sd_keys={len(state_dict)}")
    # diagnostic: sample missing keys
    if missing:
        print(f"[load] missing example: {missing[0]}")

    # CRITICAL: cast LoRA-wrapped thinker to bf16 AFTER load
    # (peft_hyper creates fp32 Linear; state_dict is bf16; any keys not in state_dict stay fp32 → matmul error)
    thinker.to(torch.bfloat16)
    thinker.eval().cuda()
    model.thinker = thinker  # rewire
    return model


@torch.inference_mode()
def generate_one(sample, model, mm_processor, data_root: str, max_new_tokens: int = 512):
    video_path = os.path.join(data_root, "AVUIE_2/unav/video", sample["video_path"])
    audio_path = os.path.join(data_root, "AVUIE_2/unav/audio", sample["audio_path"])

    conv = build_conv(video_path, audio_path, sample["question"])
    audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
    text = mm_processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = mm_processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False,
    )
    inputs = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in inputs.items()}
    inputs["use_audio_in_video"] = False

    input_len = inputs["input_ids"].shape[1]
    out = model.thinker.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    gen = out[:, input_len:]
    text_out = mm_processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return text_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "test_results_rank0.json"

    with open(args.test_json) as f:
        samples = json.load(f)

    prev = []
    done_ids = set()
    if args.resume and results_path.exists():
        with open(results_path) as f:
            prev = json.load(f)
        done_ids = {r["id"] for r in prev}
        print(f"[resume] {len(done_ids)} done, skipping")

    model = load_model(args.base_model, args.lora_path)
    mm_processor = Qwen2_5OmniProcessor.from_pretrained(args.base_model)

    print(f"[run] n_samples={len(samples)}  new={len(samples) - len(done_ids)}")
    results = list(prev)
    t0 = time.time()
    for sample in tqdm(samples, desc="Crab+"):
        if sample["id"] in done_ids:
            continue
        err = None
        pred = ""
        try:
            pred = generate_one(sample, model, mm_processor, args.data_root,
                                max_new_tokens=args.max_new_tokens)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        results.append({
            "id": sample["id"],
            "vid": sample["vid"],
            "gt_label": sample.get("gt_label", ""),
            "gt_segments": sample.get("gt_segments", []),
            "original_label": sample.get("original_label", ""),
            "question": sample["question"],
            "pred": pred,
            "error": err,
        })
        if len(results) % 50 == 0:
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    elapsed = time.time() - t0
    n_new = len(samples) - len(done_ids)
    print(f"[done] wrote {results_path}  n={len(results)}  new={n_new}  "
          f"elapsed={elapsed:.1f}s  {elapsed / max(n_new, 1):.2f} s/sample")


if __name__ == "__main__":
    main()
