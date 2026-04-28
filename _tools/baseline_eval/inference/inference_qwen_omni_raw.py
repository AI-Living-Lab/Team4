"""Qwen2.5-Omni-7B raw (no LoRA) inference runner — UnAV-100 baseline.

Crab+ runner 의 I-LoRA 부분을 완전히 제거한 변형.
Prompt 구조는 Crab+ 와 동일 Qwen-Omni native message layout (system + leading text + video + audio + user question).

Usage:
  cd /workspace/jsy/Crab_Plus
  /workspace/miniconda3/envs/crab/bin/python /workspace/jsy/scripts/inference_qwen_omni_raw.py \
      --test_json /workspace/jsy/outputs/base/QwenOmniRaw/test.json \
      --data_root /workspace/jsy/Crab_Plus/data_local \
      --base_model /workspace/models/Qwen2.5-Omni-7B \
      --output_dir /workspace/jsy/outputs/base/QwenOmniRaw/Unav100QA
"""

import os
import sys
import argparse
import json
import time
import traceback
from pathlib import Path

# Crab+ custom Qwen2.5-Omni fork 재사용 (processing + modeling)
CRAB_ROOT = "/workspace/jsy/Crab_Plus"
if CRAB_ROOT not in sys.path:
    sys.path.insert(0, CRAB_ROOT)

import torch
from tqdm import tqdm

from models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from dataset.qwen_omni_utils import process_mm_info


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
LEADING_TEXT = "This is a video and an audio:"


def build_conv(video_path: str, audio_path: str, question: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": LEADING_TEXT},
                {
                    "type": "video", "video": video_path,
                    "resized_height": 224, "resized_width": 224, "nframes": 10,
                },
                {"type": "audio", "audio": audio_path, "task": "unav"},
                {"type": "text", "text": question},
            ],
        },
    ]


def load_model(base_model_path: str, attn_impl: str = "sdpa"):
    """Raw Qwen2.5-Omni 로드 — LoRA 없음. Thinker 만 사용 (Talker 비활성)."""
    print(f"[load] base: {base_model_path}  (attn={attn_impl}, no LoRA)")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
    )
    model.disable_talker()
    model.config.use_cache = True
    model.thinker.to(torch.bfloat16).eval().cuda()
    return model


@torch.inference_mode()
def generate_one(sample, model, mm_processor, data_root: str, max_new_tokens: int = 512):
    video_abs = os.path.join(data_root, "AVUIE_2/unav/video", sample["video_path"])
    audio_abs = os.path.join(data_root, "AVUIE_2/unav/audio", sample["audio_path"])

    conv = build_conv(video_abs, audio_abs, sample["question"])
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
        **inputs, use_cache=True, max_new_tokens=max_new_tokens, do_sample=False,
    )
    gen = out[:, input_len:]
    return mm_processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "test_results_rank0.json"

    with open(args.test_json) as f:
        samples = json.load(f)

    prev, done_ids = [], set()
    if args.resume and results_path.exists():
        with open(results_path) as f:
            prev = json.load(f)
        done_ids = {r["id"] for r in prev}
        print(f"[resume] {len(done_ids)} done, skipping")

    model = load_model(args.base_model, attn_impl=args.attn)
    mm_processor = Qwen2_5OmniProcessor.from_pretrained(args.base_model)
    print(f"[run] n_samples={len(samples)}  new={len(samples) - len(done_ids)}")

    results = list(prev)
    t0 = time.time()
    for sample in tqdm(samples, desc="Qwen-Omni raw"):
        if sample["id"] in done_ids:
            continue
        err, pred = None, ""
        try:
            pred = generate_one(sample, model, mm_processor, args.data_root,
                                max_new_tokens=args.max_new_tokens)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        results.append({
            "id": sample["id"], "vid": sample["vid"],
            "gt_label": sample.get("gt_label", ""),
            "gt_segments": sample.get("gt_segments", []),
            "question": sample["question"],
            "pred": pred, "error": err,
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
