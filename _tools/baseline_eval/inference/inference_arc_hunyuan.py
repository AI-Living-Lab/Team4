"""ARC-Hunyuan-Video-7B UnAV-100 inference runner — cdh 표준 contract 준수.

Usage:
  cd /workspace/jsy/ARC-Hunyuan-Video-7B
  /workspace/miniconda3/envs/archunyuan/bin/python /workspace/jsy/scripts/inference_arc_hunyuan.py \
    --test_json /workspace/jsy/outputs/base/ArcHunyuan/test.json \
    --model_path /workspace/jsy/hf_cache/archunyuan \
    --output_dir /workspace/jsy/outputs/base/ArcHunyuan/Unav100QA

저자 `video_inference.py` 의 build_prompt / prepare_inputs / inference 로직 재사용.
Grounding task wrap 고정 (`<answer>{time range}</answer>`).

출력 (cdh 표준):
  {output_dir}/test_results_rank0.json  list, test JSON 순서, 각 row 에 'pred'
  각 row: {id, vid, gt_label, gt_segments, question, pred, error}
"""

import os
import sys
import argparse
import json
import time
import traceback
from pathlib import Path

ARC_ROOT = "/workspace/jsy/ARC-Hunyuan-Video-7B"
if ARC_ROOT not in sys.path:
    sys.path.insert(0, ARC_ROOT)

import torch
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

# 저자 video_inference.py 의 함수 그대로 import
from video_inference import build_prompt, prepare_inputs


class StopOnSubstring(StoppingCriteria):
    """Stop generation once a specific substring (e.g. '</answer>') appears in decoded output.

    ARC-Hunyuan grounding 은 thinking 마치고 `<answer>...</answer>` 뱉음. 닫힘 만나면 stop.
    """
    def __init__(self, tokenizer, stop_str: str = "</answer>", input_len: int = 0):
        self.tokenizer = tokenizer
        self.stop_str = stop_str
        self.input_len = input_len

    def __call__(self, input_ids, scores, **kwargs):
        # input_ids 가 1D 또는 2D 둘 다 처리
        if input_ids.dim() == 1:
            gen = input_ids[self.input_len:]
        else:
            gen = input_ids[0, self.input_len:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return self.stop_str in text


def load_model(model_path: str, attn_impl: str = "flash_attention_2"):
    # ARC-Hunyuan custom model classes (fallback to transformers if not custom)
    from transformers import AutoProcessor, AutoModelForCausalLM

    # ARC-Hunyuan 은 trust_remote_code 사용 (커스텀 modeling code 포함)
    print(f"[load] base: {model_path}  (attn={attn_impl})")
    try:
        from models.qwen2_5_omni.modeling_qwen2_5_omni import ARCHunyuanVideoForConditionalGeneration  # author path
    except Exception:
        ARCHunyuanVideoForConditionalGeneration = None

    if ARCHunyuanVideoForConditionalGeneration is None:
        # 공식 저자 코드 경로 — ARC-Hunyuan-Video-7B/video_inference.py:36 에서 import 됨
        from video_inference import ARCHunyuanVideoForConditionalGeneration, ARCHunyuanVideoProcessor

    model = ARCHunyuanVideoForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).eval().to("cuda")

    from video_inference import ARCHunyuanVideoProcessor
    processor = ARCHunyuanVideoProcessor.from_pretrained(model_path)
    return model, processor


@torch.inference_mode()
def generate_one(sample, model, processor, max_new_tokens: int = 1024, stop_on_answer: bool = True):
    prompt, video_inputs, audio_inputs = prepare_inputs(
        question=sample["question"],
        video_path=sample["video"],
        audio_path=sample["audio"],
        task=sample.get("task", "Grounding"),
    )
    inputs = processor(
        text=prompt,
        **video_inputs,
        **audio_inputs,
        return_tensors="pt",
    ).to("cuda", dtype=torch.bfloat16)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    if stop_on_answer:
        input_len = inputs["input_ids"].shape[-1]
        stopping = StoppingCriteriaList([
            StopOnSubstring(processor.tokenizer, "</answer>", input_len=input_len),
        ])
        gen_kwargs["stopping_criteria"] = stopping

    out = model.generate(**inputs, **gen_kwargs)
    # out may be 1-D (seq,) or 2-D (1, seq) depending on processor path
    out_tensor = out if out.dim() == 1 else out[0]
    return processor.decode(out_tensor, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--attn", default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--max_new_tokens", type=int, default=1024)
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

    model, processor = load_model(args.model_path, attn_impl=args.attn)
    print(f"[run] n_samples={len(samples)}  new={len(samples) - len(done_ids)}")

    results = list(prev)
    t0 = time.time()
    for sample in tqdm(samples, desc="ARC-Hunyuan"):
        if sample["id"] in done_ids:
            continue
        err, pred = None, ""
        try:
            pred = generate_one(sample, model, processor, max_new_tokens=args.max_new_tokens)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        results.append({
            "id": sample["id"],
            "vid": sample["vid"],
            "gt_label": sample.get("gt_label", ""),
            "gt_segments": sample.get("gt_segments", []),
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
