#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_sample.py
  GDPO ckpt 진단 스크립트.

  학습은 안 돌리고, ckpt 로드 → 데이터셋에서 N개 프롬프트 뽑아서
  여러 temperature로 각각 K개씩 generate → raw/sec/format/iou 비교.

  목적
    - "T=1.5가 학습을 망친 거냐 / policy가 영구 손상된 거냐" 판별
    - greedy(T=0)에서도 format이 깨지면 policy 영구 손상
    - greedy는 정상이고 T=1.5만 깨지면 샘플링 노이즈

Usage
  set -a && source paths.env && set +a
  python _tools/GDPO/debug_sample.py \
    --config _tools/GDPO/config.yaml \
    --model_path ${GDPO_CKPT}/checkpoint-500 \
    --model_base ${BASE_MODEL} \
    --dataset_path data/unav100_train_dense.json \
    --num_prompts 3 \
    --num_gens 8 \
    --temperatures 0.0,1.0,1.5
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "video_SALMONN2_plus"))
sys.path.insert(0, _THIS_DIR)

from gdpo_trainer import load_model_and_tokenizer, load_config
from reward_functions import format_reward, iou_reward, decode_vtg_time

from qwenvl.data.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from transformers import WhisperFeatureExtractor


_SEG_CAPTURE_DEBUG = re.compile(
    r"[Ff]rom\s+((?:<t\d>){1,4}<tdot><t\d>)\s+to\s+((?:<t\d>){1,4}<tdot><t\d>)"
)


def extract_segments(text):
    raw_segs, sec_segs = [], []
    for s_str, e_str in _SEG_CAPTURE_DEBUG.findall(text):
        raw_segs.append(f"from {s_str} to {e_str}")
        s = decode_vtg_time(s_str)
        e = decode_vtg_time(e_str)
        if s is not None and e is not None:
            sec_segs.append(f"({s:.1f}, {e:.1f})")
    raw_str = ", ".join(raw_segs) if raw_segs else "[no valid segment]"
    sec_str = ", ".join(sec_segs) if sec_segs else "[no valid segment]"
    return raw_str, sec_str


def parse_gt_intervals(tokenizer, raw_labels):
    """labels에서 GT 시간 구간 + raw 토큰 형태 추출."""
    gt_intervals = []
    gt_raw = "[none]"
    if raw_labels is None:
        return gt_intervals, gt_raw
    gt_ids = raw_labels[0][raw_labels[0] != -100]
    if len(gt_ids) == 0:
        return gt_intervals, gt_raw
    gt_answer = tokenizer.decode(gt_ids, skip_special_tokens=False)
    raw_segs = []
    for s_str, e_str in _SEG_CAPTURE_DEBUG.findall(gt_answer):
        raw_segs.append(f"from {s_str} to {e_str}")
        s = decode_vtg_time(s_str)
        e = decode_vtg_time(e_str)
        if s is not None and e is not None and e > s:
            gt_intervals.append((s, e))
    if raw_segs:
        gt_raw = ", ".join(raw_segs)
    return gt_intervals, gt_raw


def build_data_args(dataset_path, model_base):
    @dataclass
    class DA:
        dataset_use: str = ""
        model_type: str = "qwen2.5vl"
        video_max_frames: int = 128
        video_min_frames: int = 4
        base_interval: float = 2
        max_pixels: int = 176400
        min_pixels: int = 784
        video_max_frame_pixels: int = 25088
        video_min_frame_pixels: int = 3136
        video_max_total_pixels: int = 1664 * 28 * 28
        video_min_total_pixels: int = 256 * 28 * 28
        run_test: bool = False
        do_sample: bool = False
        num_sample: int = 1
        train_type: str = "sft"
        feature_size: int = 128
        chunk_length: int = 30
        hop_length: int = 160
        sampling_rate: int = 16000
        image_processor: object = None
        audio_processor: object = None

    da = DA()
    da.dataset_use = dataset_path
    da.image_processor = Qwen2VLImageProcessorFast.from_pretrained(model_base)
    da.audio_processor = WhisperFeatureExtractor(
        feature_size=da.feature_size,
        sampling_rate=da.sampling_rate,
        hop_length=da.hop_length,
        chunk_length=da.chunk_length,
    )
    return da


def prepare_inputs(inputs, device):
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if k in ("pixel_values_videos", "audio_feature"):
                out[k] = v.to(device=device, dtype=torch.bfloat16)
            else:
                out[k] = v.to(device)
        else:
            out[k] = v
    return out


def strip_prompt_part(prompt_ids, prompt_mask, labels):
    """labels로 prompt 끝 인덱스 찾기 (GDPOTrainer.compute_loss 와 동일 로직)."""
    if labels is None:
        return prompt_ids, prompt_mask
    answer_start = (labels[0] != -100).nonzero(as_tuple=True)[0]
    if len(answer_start) == 0:
        return prompt_ids, prompt_mask
    end = answer_start[0].item()
    return prompt_ids[:, :end], prompt_mask[:, :end]


def clean_completion(text, special_tokens):
    """gdpo_trainer.py와 동일한 후처리: time token 외 special token 제거."""
    time_tokens = {f"<t{i}>" for i in range(10)} | {"<tdot>"}
    to_remove = set(special_tokens) - time_tokens
    for tok in to_remove:
        text = text.replace(tok, "")
    text = re.sub(r"<\|im_start\|>\s*\w+\s*", "", text).strip()
    return text


@torch.no_grad()
def sample_one(model, tokenizer, prep, temperature, num_gens, max_new_tokens):
    """동일 prompt에서 temperature별로 num_gens개 생성. greedy면 num_gens=1로 강제."""
    is_greedy = temperature <= 0.0
    if is_greedy:
        num_gens = 1
    gen_kwargs = {
        "input_ids": prep["input_ids"],
        "attention_mask": prep["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "do_sample": not is_greedy,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if not is_greedy:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 1.0
    for k in ("pixel_values_videos", "video_grid_thw", "audio_feature", "audio_lengths"):
        if k in prep and prep[k] is not None:
            gen_kwargs[k] = prep[k]

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    prompt_len = prep["input_ids"].size(1)

    outs = []
    for _ in range(num_gens):
        gen_ids = base.generate(**gen_kwargs)
        comp_ids = gen_ids[0, prompt_len:]
        text = tokenizer.decode(comp_ids, skip_special_tokens=False)
        text = clean_completion(text, tokenizer.all_special_tokens)
        outs.append(text)
    return outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model_path", required=True, help="GDPO ckpt (LoRA adapter)")
    parser.add_argument("--model_base", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--num_prompts", type=int, default=3)
    parser.add_argument("--num_gens", type=int, default=8)
    parser.add_argument("--temperatures", default="0.0,1.0,1.5",
                        help="콤마로 구분된 temperature 리스트")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    temperatures = [float(t.strip()) for t in args.temperatures.split(",") if t.strip()]
    torch.manual_seed(args.seed)

    print(f"[DEBUG] temperatures = {temperatures}")
    print(f"[DEBUG] num_prompts={args.num_prompts}, num_gens={args.num_gens}")

    # 모델 로드
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.model_base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # 데이터셋
    data_args = build_data_args(args.dataset_path, args.model_base)
    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    print(f"[DEBUG] dataset size: {len(dataset)}")

    # 통계
    stats = {t: {"format_sum": 0.0, "iou_sum": 0.0, "valid_format": 0, "total": 0}
             for t in temperatures}

    n = min(args.num_prompts, len(dataset))
    for pi in range(n):
        sample = dataset[pi]
        batch = collator([sample])
        batch = prepare_inputs(batch, device)

        labels = batch.get("labels", None)
        prompt_ids, prompt_mask = strip_prompt_part(
            batch["input_ids"], batch["attention_mask"], labels
        )
        prep = {
            "input_ids": prompt_ids,
            "attention_mask": prompt_mask,
            "pixel_values_videos": batch.get("pixel_values_videos", None),
            "video_grid_thw": batch.get("video_grid_thw", None),
            "audio_feature": batch.get("audio_feature", None),
            "audio_lengths": batch.get("audio_lengths", None),
        }

        gt_intervals, gt_raw = parse_gt_intervals(tokenizer, labels)
        gt_sec = (", ".join(f"({s:.1f}, {e:.1f})" for s, e in gt_intervals)
                  if gt_intervals else "[none]")

        print("\n" + "=" * 78)
        print(f"[PROMPT {pi+1}/{n}]")
        print(f"  GT raw: {gt_raw}")
        print(f"  GT sec: {gt_sec}")

        for T in temperatures:
            label = "greedy" if T <= 0.0 else f"T={T}"
            print(f"\n  --- {label} ---")
            outs = sample_one(model, tokenizer, prep, T, args.num_gens, args.max_new_tokens)
            for gi, c in enumerate(outs):
                raw, sec = extract_segments(c)
                fr = format_reward(c)
                ir = iou_reward(c, gt_intervals)
                stats[T]["format_sum"] += fr
                stats[T]["iou_sum"] += ir
                stats[T]["total"] += 1
                if fr > 0:
                    stats[T]["valid_format"] += 1
                print(f"    [{gi}] format={fr:.2f}  iou={ir:.3f}")
                print(f"         raw: {raw}")
                print(f"         sec: {sec}")

    # 요약
    print("\n" + "=" * 78)
    print("[SUMMARY]")
    print(f"{'temperature':<14}{'samples':<10}{'valid_fmt%':<14}{'mean_format':<14}{'mean_iou':<10}")
    for T in temperatures:
        s = stats[T]
        n_tot = max(s["total"], 1)
        print(f"{T:<14}{s['total']:<10}{100*s['valid_format']/n_tot:<14.1f}"
              f"{s['format_sum']/n_tot:<14.3f}{s['iou_sum']/n_tot:<10.3f}")
    print("\n해석:")
    print("  - greedy에서 valid_fmt%가 낮음 → policy가 영구 손상됨 (argmax 이동)")
    print("  - greedy 정상이지만 T=1.5에서 낮음 → 학습 시 샘플링 노이즈 문제")
    print("  - greedy 정상이고 T=1.0도 정상이면 학습 자체 문제 아님")


if __name__ == "__main__":
    main()
