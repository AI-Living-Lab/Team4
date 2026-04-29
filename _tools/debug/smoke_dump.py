#!/usr/bin/env python3
"""
TTI debug_interleave smoke test — 모델 로드 없이 tokenizer + image_processor 만으로
LazySupervisedDataset 을 돌려 debug dump 훅이 정상 작동하는지 검증.

사용법:
  python _tools/debug/smoke_dump.py \
    --model_base /workspace/checkpoints/base/video_salmonn2_plus_7B_time_tokens \
    --dataset_use data/debug_interleave_samples.json \
    --out_dir _debug_out/smoke
"""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True,
                    help="Qwen2.5-VL base (image_processor + tokenizer) 가 있는 디렉토리")
    ap.add_argument("--dataset_use", required=True,
                    help="디버그 샘플 JSON 경로 (BASE_DIR 기준 상대 또는 절대)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_interval", type=float, default=0.2)
    ap.add_argument("--video_min_frames", type=int, default=64)
    ap.add_argument("--video_max_frames", type=int, default=128)
    ap.add_argument("--max_pixels", type=int, default=176400)
    ap.add_argument("--min_pixels", type=int, default=784)
    ap.add_argument("--video_max_frame_pixels", type=int, default=28224)
    ap.add_argument("--video_min_frame_pixels", type=int, default=784)
    ap.add_argument("--sample_limit", type=int, default=-1)
    args = ap.parse_args()

    # BASE_CODE 를 sys.path 에 추가
    here = Path(__file__).resolve()
    base_dir = here.parents[2]  # Team4/
    base_code = base_dir / "video_SALMONN2_plus"
    sys.path.insert(0, str(base_code))

    # dataset_use / out_dir 을 절대경로로 정규화 (BASE_DIR 기준)
    def _abs(p: str) -> str:
        p = Path(p)
        return str(p if p.is_absolute() else (base_dir / p).resolve())
    args.dataset_use = _abs(args.dataset_use)
    args.out_dir = _abs(args.out_dir)

    from transformers import AutoTokenizer, WhisperFeatureExtractor
    from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
    from qwenvl.data.dataset import LazySupervisedDataset

    print(f"[smoke] loading tokenizer + image_processor from {args.model_base}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    image_processor = Qwen2VLImageProcessorFast.from_pretrained(args.model_base)
    audio_processor = WhisperFeatureExtractor(
        feature_size=128, sampling_rate=16000,
        hop_length=160, chunk_length=30,
    )

    # data_args — DataArguments 필드를 SimpleNamespace 로 흉내
    data_args = SimpleNamespace(
        dataset_use=args.dataset_use,
        video_max_frames=args.video_max_frames,
        video_min_frames=args.video_min_frames,
        base_interval=args.base_interval,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        video_max_frame_pixels=args.video_max_frame_pixels,
        video_min_frame_pixels=args.video_min_frame_pixels,
        run_test=True,
        do_sample=False,
        num_sample=1,
        train_type="sft",
        feature_size=128, chunk_length=30, hop_length=160, sampling_rate=16000,
        debug_interleave_dir=args.out_dir,
        debug_interleave_generate=False,
        debug_interleave_sample_limit=args.sample_limit,
        image_processor=image_processor,
        audio_processor=audio_processor,
        model_type="qwen2.5vl",
    )

    print(f"[smoke] building dataset from {args.dataset_use}")
    ds = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    print(f"[smoke] dataset size={len(ds)}")

    # 전체 or 앞에서 N개 반복 (sample_limit 로 덤프 제한)
    n_iter = len(ds) if args.sample_limit < 0 else min(len(ds), args.sample_limit)
    for i in range(n_iter):
        try:
            _ = ds[i]
            print(f"  [{i}] dump ok")
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")

    out_p = Path(args.out_dir)
    files = sorted(out_p.glob("*.json")) if out_p.exists() else []
    print(f"\n[smoke] {len(files)} JSON dumps in {out_p}")
    for f in files:
        print(f"  - {f.name}")


if __name__ == "__main__":
    sys.exit(main())
