#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gdpo.py
  GDPO 학습 진입점.
  SFT 체크포인트를 로드하고 trl 기반 GDPO 트레이너로 강화학습 수행.

Usage:
  python _tools/GDPO/train_gdpo.py \
    --model_path /root/checkpoints/checkpoint-26000 \
    --model_base lmms-lab/llava-onevision-qwen2-7b-ov \
    --dataset_path data/unav100_train_gdpo.json \
    --output_dir output/gdpo_run1
"""

import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig
from transformers import AutoTokenizer

# 프로젝트 루트를 PYTHONPATH에 추가
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from _tools.GDPO.gdpo_trainer import GDPOTrainer
from _tools.GDPO.reward_functions import (
    format_reward,
    label_reward,
    iou_reward,
)


def load_gdpo_dataset(dataset_path: str) -> Dataset:
    """GDPO 형식의 JSON 파일을 HuggingFace Dataset으로 변환.

    GDPO JSON 형식:
      [{"video": ..., "audio": ..., "prompt": ..., "events": [...]}, ...]

    GRPOTrainer가 기대하는 형식:
      Dataset with "prompt" column
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # GRPOTrainer는 prompt 컬럼이 필요
    records = []
    for sample in data:
        records.append({
            "prompt": sample["prompt"],
            "video": sample.get("video", ""),
            "audio": sample.get("audio", ""),
            "gt_events": json.dumps(sample.get("events", []), ensure_ascii=False),
        })

    return Dataset.from_list(records)


def make_reward_functions(gt_events_by_prompt: dict):
    """GT 이벤트를 참조하는 리워드 함수들을 생성.

    trl의 GRPOTrainer는 reward_funcs에 callable 리스트를 받습니다.
    각 함수는 (completions, prompts, ...) → list[float] 형태.
    """

    def _format_reward(completions, **kwargs):
        """포맷 리워드: JSON 파싱 가능 여부."""
        return [format_reward(c) for c in completions]

    def _label_reward(completions, prompts=None, **kwargs):
        """라벨 리워드: GT 라벨 매칭 비율."""
        rewards = []
        for i, c in enumerate(completions):
            prompt = prompts[i] if prompts else ""
            gt = gt_events_by_prompt.get(prompt, [])
            rewards.append(label_reward(c, gt))
        return rewards

    def _iou_reward(completions, prompts=None, **kwargs):
        """IoU 리워드: temporal IoU 정확도."""
        rewards = []
        for i, c in enumerate(completions):
            prompt = prompts[i] if prompts else ""
            gt = gt_events_by_prompt.get(prompt, [])
            rewards.append(iou_reward(c, gt))
        return rewards

    return [_format_reward, _label_reward, _iou_reward]


def main():
    parser = argparse.ArgumentParser(description="GDPO Training")
    # 모델 관련
    parser.add_argument("--model_path", required=True,
                        help="SFT 체크포인트 경로 (예: checkpoint-26000)")
    parser.add_argument("--model_base", default="lmms-lab/llava-onevision-qwen2-7b-ov",
                        help="베이스 모델 경로 또는 HF ID")
    # 데이터
    parser.add_argument("--dataset_path", required=True,
                        help="GDPO 형식 데이터 JSON 경로")
    # 학습 설정
    parser.add_argument("--output_dir", default="output/gdpo",
                        help="출력 디렉토리")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="프롬프트당 생성할 응답 수")
    parser.add_argument("--max_completion_length", type=int, default=1024,
                        help="생성 최대 토큰 수")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="학습 에폭 수")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="디바이스당 배치 사이즈")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="그래디언트 누적 스텝")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="학습률")
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL 패널티 계수")
    parser.add_argument("--reward_weights", type=float, nargs=3,
                        default=[0.1, 0.3, 0.6],
                        help="리워드 가중치 [format, label, iou]")
    # LoRA
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # 기타
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()

    # ============================================================
    # 1. 데이터 로드
    # ============================================================
    print(f"[GDPO] Loading dataset from {args.dataset_path}")
    dataset = load_gdpo_dataset(args.dataset_path)
    print(f"[GDPO] Dataset size: {len(dataset)}")

    # GT 이벤트를 prompt로 인덱싱 (리워드 함수에서 참조)
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    gt_events_by_prompt = {}
    for sample in raw_data:
        gt_events_by_prompt[sample["prompt"]] = sample.get("events", [])

    # ============================================================
    # 2. 리워드 함수 설정
    # ============================================================
    reward_funcs = make_reward_functions(gt_events_by_prompt)
    print(f"[GDPO] Reward functions: {len(reward_funcs)} "
          f"(format, label, iou), weights={args.reward_weights}")

    # ============================================================
    # 3. 모델 & 토크나이저 로드
    # ============================================================
    print(f"[GDPO] Loading model from {args.model_path}")
    print(f"[GDPO] Base model: {args.model_base}")

    # LoRA 설정
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ============================================================
    # 4. GDPO 트레이너 설정
    # ============================================================
    training_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        gradient_checkpointing=True,
        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    trainer = GDPOTrainer(
        model=args.model_path,
        reward_funcs=reward_funcs,
        config=training_config,
        train_dataset=dataset,
        peft_config=peft_config,
        reward_weights=args.reward_weights,
        apply_gdpo=True,
    )

    # ============================================================
    # 5. 학습 시작
    # ============================================================
    print("[GDPO] Starting training...")
    trainer.train()

    # 저장
    print(f"[GDPO] Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    print("[GDPO] Training complete!")


if __name__ == "__main__":
    main()
