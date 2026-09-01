#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoke_test_fastgen.py — 배치 생성 패치(fastgen) 검증용 스모크 테스트.

검증 항목
  1) [정확성] greedy(do_sample=False) 로 순차 생성 G회 vs 배치 생성 1회 →
     토큰이 동일한가? (M-RoPE / audio_lengths / video_grid_thw tile 이 맞았는지)
  2) [속도]   sampling(do_sample=True) 로 두 경로의 wall-time 비교
  3) [메모리] 각 경로의 peak GPU 메모리

안전장치
  - torch.cuda.set_per_process_memory_fraction() 으로 프로세스 GPU 상한을 강제.
    상한을 넘으면 '이 프로세스가' OOM 으로 죽고, 옆에서 도는 본 학습은 건드리지 않는다.
  - 기본 상한 20GiB. --gpu-budget-gb 로 조정.

사용:
  python gdpo/smoke_test_fastgen.py \
      --model_base /home/team404/workspace/checkpoints/base \
      --dataset_path /home/team404/workspace/data/train/unpucha_v2.json \
      --num_generations 8 --gpu-budget-gb 20
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "video_SALMONN2_plus"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "gdpo"))

from transformers import AutoTokenizer, WhisperFeatureExtractor  # noqa: E402
from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus  # noqa: E402
from qwenvl.data.dataset import (  # noqa: E402
    LazySupervisedDataset,
    DataCollatorForSupervisedDataset,
)
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast  # noqa: E402


# ---------------------------------------------------------------------------
# 패치 대상과 '동일한' tile 헬퍼 (fastgen 패치본에서 그대로 옮김)
# ---------------------------------------------------------------------------
def _tile(x, n):
    """배치 차원이 없는 멀티모달 입력을 샘플 단위로 n배 복제(tile).
    [s0,s1] -> [s0,s1,s0,s1]  (repeat_interleave 의 [s0,s0,s1,s1] 이 아님)"""
    if x is None:
        return None
    if isinstance(x, list):
        return x * n
    reps = [1] * x.dim()
    reps[0] = n
    return x.repeat(*reps)


@dataclass
class GDPODataArgs:
    """트레이너 main() 의 GDPODataArgs 와 동일한 값."""
    dataset_use: str = ""
    model_type: str = "qwen2.5vl"
    video_max_frames: int = 128
    video_min_frames: int = 64
    base_interval: float = 0.2
    max_pixels: int = 176400
    min_pixels: int = 784
    video_max_frame_pixels: int = 28224
    video_min_frame_pixels: int = 784
    video_max_total_pixels: int = 1664 * 28 * 28
    video_min_total_pixels: int = 256 * 28 * 28
    run_test: bool = False
    do_sample: bool = False
    num_sample: int = 1
    train_type: str = "sft"
    tti_time_format: str = "special_token"
    feature_size: int = 128
    chunk_length: int = 30
    hop_length: int = 160
    sampling_rate: int = 16000
    image_processor: object = None
    audio_processor: object = None
    video_cache_size: int = 0


def build_gen_kwargs(b, prompt_ids, prompt_mask, mm, max_new_tokens, do_sample, temperature):
    """트레이너의 _build_gen_kwargs 와 동일한 키 구성.
    second_per_grid_ts 는 원본도 generate 에 넘기지 않으므로 제외."""
    gk = {
        "input_ids": prompt_ids.repeat(b, 1),
        "attention_mask": prompt_mask.repeat(b, 1),
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": 1.0,
    }
    if do_sample:
        gk.update({"temperature": temperature, "top_p": 1.0, "top_k": 50})
    if mm["pixel_values_videos"] is not None:
        gk["pixel_values_videos"] = _tile(mm["pixel_values_videos"], b)
    if mm["video_grid_thw"] is not None:
        gk["video_grid_thw"] = _tile(mm["video_grid_thw"], b)
    if mm["audio_feature"] is not None:
        gk["audio_feature"] = _tile(mm["audio_feature"], b)
    if mm["audio_lengths"] is not None:
        gk["audio_lengths"] = _tile(mm["audio_lengths"], b)
    return gk


def run_sequential(model, G, prompt_ids, prompt_mask, mm, max_new, do_sample, temp):
    """현재(원본) 경로: batch=1 generate 를 G회 순차 호출."""
    outs = []
    plen = prompt_ids.size(1)
    for _ in range(G):
        ids = model.generate(**build_gen_kwargs(1, prompt_ids, prompt_mask, mm, max_new, do_sample, temp))
        outs.append(ids[:, plen:])
    return outs


def run_batched(model, G, prompt_ids, prompt_mask, mm, max_new, do_sample, temp, gen_bs):
    """패치 경로: 입력을 직접 tile 해 배치로 묶어 호출."""
    outs = []
    plen = prompt_ids.size(1)
    remaining = G
    while remaining > 0:
        b = min(gen_bs, remaining)
        ids = model.generate(**build_gen_kwargs(b, prompt_ids, prompt_mask, mm, max_new, do_sample, temp))
        outs.append(ids[:, plen:])
        remaining -= b
    return outs


def flatten_rows(outs, pad_id):
    """[(b, L_i)] -> 행 단위 리스트. EOS 이후는 비교에서 제외하기 위해 그대로 둔다."""
    rows = []
    for t in outs:
        for r in range(t.size(0)):
            rows.append(t[r].tolist())
    return rows


def trim_at_eos(row, eos_id, pad_id):
    out = []
    for tok in row:
        out.append(tok)
        if tok == eos_id:
            break
    while out and out[-1] == pad_id:
        out.pop()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gen_bs", type=int, default=0, help="0 = num_generations (1회 배치 호출)")
    ap.add_argument("--n_samples", type=int, default=2, help="검증에 쓸 데이터 샘플 수")
    ap.add_argument("--gpu-budget-gb", type=float, default=20.0,
                    dest="gpu_budget_gb", help="이 프로세스의 GPU 메모리 상한(GiB)")
    ap.add_argument("--tti_mode", default="on")
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()

    G = args.num_generations
    gen_bs = args.gen_bs or G

    assert torch.cuda.is_available(), "CUDA 필요"
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    frac = min(0.95, args.gpu_budget_gb / total_gb)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print(f"[safety] GPU 총 {total_gb:.1f}GiB / 이 프로세스 상한 {args.gpu_budget_gb:.1f}GiB "
          f"(fraction={frac:.3f}) → 상한 초과 시 본 학습이 아니라 이 프로세스가 죽는다")
    free_b, _ = torch.cuda.mem_get_info(0)
    print(f"[safety] 현재 GPU 여유: {free_b/1024**3:.1f}GiB")

    # ── 모델/토크나이저 ──
    print(f"[load] tokenizer: {args.model_base}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, model_max_length=10000, padding_side="left")
    print(f"[load] model: {args.model_base}")
    model = video_SALMONN2_plus.from_pretrained(
        args.model_base, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
    )
    if args.tti_mode == "off":
        model.config.time_token_id_range = None
        model.config.time_marker_token_len = None
    model = model.cuda().eval()
    print(f"[load] 모델 로드 후 GPU 사용: {torch.cuda.memory_allocated()/1024**3:.1f}GiB")

    # ── 데이터셋 ──
    data_args = GDPODataArgs()
    data_args.dataset_use = args.dataset_path
    data_args.tti_time_format = "special_token" if args.tti_mode == "on" else "off"
    data_args.image_processor = Qwen2VLImageProcessorFast.from_pretrained(args.model_base)
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size, sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length, chunk_length=data_args.chunk_length,
    )
    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    print(f"[load] dataset size: {len(dataset)}")

    pad_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id

    all_ok = True
    for si in range(args.n_samples):
        print(f"\n{'='*70}\n[sample {si}]")
        batch = collator([dataset[si]])
        dev = torch.device("cuda")

        prompt_ids = batch["input_ids"].to(dev)
        prompt_mask = batch["attention_mask"].to(dev)
        labels = batch.get("labels", None)
        if labels is not None:
            labels = labels.to(dev)
            ans = (labels[0] != -100).nonzero(as_tuple=True)[0]
            if len(ans) > 0:
                cut = ans[0].item()
                prompt_ids = prompt_ids[:, :cut]
                prompt_mask = prompt_mask[:, :cut]

        mm = {
            "pixel_values_videos": batch.get("pixel_values_videos", None),
            "video_grid_thw": batch.get("video_grid_thw", None),
            "audio_feature": batch.get("audio_feature", None),
            "audio_lengths": batch.get("audio_lengths", None),
        }
        if mm["pixel_values_videos"] is not None:
            mm["pixel_values_videos"] = mm["pixel_values_videos"].to(dev, dtype=torch.bfloat16)
        if mm["video_grid_thw"] is not None:
            mm["video_grid_thw"] = mm["video_grid_thw"].to(dev)
        if mm["audio_feature"] is not None:
            mm["audio_feature"] = mm["audio_feature"].to(dev, dtype=torch.bfloat16)

        print(f"  prompt_len={prompt_ids.size(1)}")
        for k, v in mm.items():
            if v is None:
                print(f"  {k}: None")
            elif isinstance(v, list):
                print(f"  {k}: list(len={len(v)}) = {v}")
            else:
                print(f"  {k}: shape={tuple(v.shape)}")

        # ── 1) 정확성: greedy 순차 vs greedy 배치 ──
        print("\n  [1] 정확성 검증 (greedy, do_sample=False)")
        with torch.no_grad():
            torch.cuda.reset_peak_memory_stats()
            seq_out = run_sequential(model, G, prompt_ids, prompt_mask, mm,
                                     args.max_new_tokens, False, args.temperature)
            seq_peak = torch.cuda.max_memory_allocated() / 1024**3

            torch.cuda.reset_peak_memory_stats()
            bat_out = run_batched(model, G, prompt_ids, prompt_mask, mm,
                                  args.max_new_tokens, False, args.temperature, gen_bs)
            bat_peak = torch.cuda.max_memory_allocated() / 1024**3

        seq_rows = [trim_at_eos(r, eos_id, pad_id) for r in flatten_rows(seq_out, pad_id)]
        bat_rows = [trim_at_eos(r, eos_id, pad_id) for r in flatten_rows(bat_out, pad_id)]

        # greedy 이므로 G개 행이 모두 같아야 하고, 두 경로도 같아야 한다.
        ref = seq_rows[0]
        n_match = sum(1 for r in bat_rows if r == ref)
        print(f"      순차 greedy 첫 행 == 배치 greedy 행: {n_match}/{G} 일치")
        seq_self = sum(1 for r in seq_rows if r == ref)
        print(f"      (참고) 순차 경로 내부 자기일치: {seq_self}/{G}")
        print(f"      순차 peak={seq_peak:.1f}GiB  배치 peak={bat_peak:.1f}GiB")
        print(f"      decode(seq[0]) = {tokenizer.decode(ref, skip_special_tokens=False)[:200]!r}")
        if n_match != G:
            all_ok = False
            for i, r in enumerate(bat_rows):
                if r != ref:
                    print(f"      ✗ 불일치 행 {i}: {tokenizer.decode(r, skip_special_tokens=False)[:200]!r}")
                    break

        # ── 2) 속도 ──
        if not args.skip_timing:
            print("\n  [2] 속도 비교 (sampling, do_sample=True)")
            with torch.no_grad():
                torch.cuda.synchronize(); t0 = time.time()
                s_out = run_sequential(model, G, prompt_ids, prompt_mask, mm,
                                       args.max_new_tokens, True, args.temperature)
                torch.cuda.synchronize(); t_seq = time.time() - t0

                torch.cuda.synchronize(); t0 = time.time()
                b_out = run_batched(model, G, prompt_ids, prompt_mask, mm,
                                    args.max_new_tokens, True, args.temperature, gen_bs)
                torch.cuda.synchronize(); t_bat = time.time() - t0

            s_len = [len(trim_at_eos(r, eos_id, pad_id)) for r in flatten_rows(s_out, pad_id)]
            b_len = [len(trim_at_eos(r, eos_id, pad_id)) for r in flatten_rows(b_out, pad_id)]
            print(f"      순차: {t_seq:6.2f}s  (완성 길이 {s_len})")
            print(f"      배치: {t_bat:6.2f}s  (완성 길이 {b_len})")
            print(f"      ▶ speedup = {t_seq/max(t_bat,1e-9):.2f}x")

    print(f"\n{'='*70}")
    print("결과:", "✅ 정확성 통과 (배치 == 순차)" if all_ok else "❌ 정확성 실패 — 패치 재검토 필요")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
