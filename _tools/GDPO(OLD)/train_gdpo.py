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
from transformers import AutoTokenizer, AutoConfig

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
from llava.model.language_model.video_salmonn_2 import VideoSALMONN2ForCausalLM


def load_model_and_tokenizer(model_path, model_base, lora_r=32, lora_alpha=64, lora_dropout=0.05):
    """VideoSALMONN2 모델과 토크나이저를 로드.

    기존 프로젝트의 load_qwen_lora_model 로직을 간소화.
    SFT LoRA 체크포인트 또는 베이스 모델을 로드합니다.
    """
    # 체크포인트가 로컬 디렉토리인지 확인
    is_local_ckpt = os.path.isdir(model_path)

    if is_local_ckpt:
        # 로컬 체크포인트에서 config 로드
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                ckpt_config = json.load(f)
        else:
            ckpt_config = {}

        # 토크나이저: 체크포인트에 있으면 거기서, 없으면 베이스에서
        tok_path = model_path if os.path.exists(os.path.join(model_path, "tokenizer.json")) else model_base
    else:
        ckpt_config = {}
        tok_path = model_base

    print(f"[GDPO] Loading tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path,
        model_max_length=4096,
        padding_side="right",
    )

    # 베이스 모델 config
    print(f"[GDPO] Loading base config from: {model_base}")
    cfg = AutoConfig.from_pretrained(model_base)

    # config에 model_args가 있으면 반영
    if "model_args" in ckpt_config:
        for k, v in ckpt_config["model_args"].items():
            setattr(cfg, k, v) if hasattr(cfg, k) else None
        cfg.model_args = ckpt_config["model_args"]

    # add_time_token 기본값 설정
    if "add_time_token" in ckpt_config:
        cfg.add_time_token = ckpt_config["add_time_token"]
    elif not hasattr(cfg, "add_time_token"):
        cfg.add_time_token = False

    # 모델 로드
    print(f"[GDPO] Loading base model from: {model_base}")
    model = VideoSALMONN2ForCausalLM.from_pretrained(
        model_base,
        config=cfg,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))

    # LoRA 어댑터 로드 (SFT 체크포인트인 경우)
    adapter_config = os.path.join(model_path, "adapter_config.json") if is_local_ckpt else ""
    if is_local_ckpt and os.path.exists(adapter_config):
        from peft import PeftModel
        print(f"[GDPO] Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
        model = model.to(torch.bfloat16)
    else:
        print("[GDPO] No LoRA adapter found, using base model with new LoRA")

    return model, tokenizer


def load_gdpo_dataset(dataset_path: str) -> Dataset:
    """GDPO 형식의 JSON 파일을 HuggingFace Dataset으로 변환."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

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
    """GT 이벤트를 참조하는 리워드 함수들을 생성."""

    def _format_reward(completions, **kwargs):
        return [format_reward(c) for c in completions]

    def _label_reward(completions, prompts=None, **kwargs):
        rewards = []
        for i, c in enumerate(completions):
            prompt = prompts[i] if prompts else ""
            gt = gt_events_by_prompt.get(prompt, [])
            rewards.append(label_reward(c, gt))
        return rewards

    def _iou_reward(completions, prompts=None, **kwargs):
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
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        model_base=args.model_base,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # LoRA 설정 (어댑터가 없는 경우 새로 적용)
    peft_config = None
    if not hasattr(model, "peft_config"):
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
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_config,
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
