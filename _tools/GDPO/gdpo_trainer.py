#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gdpo_trainer.py 
  transformers.Trainer 상속 + GDPO compute_loss 전체 루프.
  Time-R1(timer1_trainer.py), Omni-R1(grpo_trainer.py) 구조 기반.
  VideoSALMONN2 멀티모달 API에 맞게 구현.

Usage:
  # config.yaml 사용 (권장)
  set -a && source paths.env && set +a
  python _tools/GDPO/gdpo_trainer.py --config _tools/GDPO/config.yaml

  # CLI 직접 지정
  python -m _tools.GDPO_v2_revised.gdpo_trainer \\
    --model_path ${SFT_CKPT} --model_base ${BASE_MODEL} \\
    --dataset_path data/unav100_train_gdpo.json

    set -a && source paths.env && set +a
    python _tools/GDPO/gdpo_trainer.py \
    --config _tools/GDPO/config.yaml \
    --model_path ${SFT_CKPT} \
    --model_base ${BASE_MODEL} \
    --dataset_path data/unav100_train_dense.json \
    --output_dir output/gdpo_test  

참고:
  - Time-R1: https://github.com/xiaomi-research/time-r1
  - Omni-R1: https://github.com/aim-uofa/Omni-R1
  - GDPO:    https://github.com/NVlabs/GDPO
"""


# 임포트
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Callable, List, Dict, Optional, Union

import yaml
import torch
from packaging import version

import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
    PreTrainedModel,
    Trainer,
    # is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model


# if is_wandb_available():
#     import wandb





# 프로젝트 루트
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llava.model.language_model.video_salmonn_2 import VideoSALMONN2ForCausalLM
from llava.dataset.av_dataset import LazyAVSupervisedDataset, DataCollatorForAVSupervisedDataset

# 같은 폴더의 reward_functions
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from reward_functions import format_reward, label_reward, iou_reward

# 리워드 함수 타입 정의
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]






# ============================================================
# 유틸리티
# ============================================================

# 둘 다 로깅용. 일단 둘 다 있어서 가져옴
def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])







# ============================================================
# GDPOTrainer
# ============================================================

class GDPOTrainer(Trainer):
    """GDPO Trainer — transformers.Trainer 상속.

    compute_loss() 안에서 generate → reward → GDPO advantage → loss 전체 수행.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        reward_funcs: list[RewardFunc],
        args: GRPOConfig = None,
        train_dataset=None,
        # eval_dataset = None,
        processing_class=None,
        ref_model: Optional[PreTrainedModel] = None,
        callbacks=None,
        optimizers=(None, None),
        peft_config: Optional["PeftConfig"] = None,
        reward_weights: Optional[list[float]] = None,
    ):
        # ── Args 기본값 ──
        # trl에서 들고온 GRPOConfig 
        if args is None:
            model_name = model.config._name_or_path.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GDPO")


        # ── PEFT ──
        # TODO : llm과 encoder 분리


        if peft_config is not None:
            # 인코더 분리
            speech_encoder = model.speech_encoder
            model.speech_encoder = None
            vision_tower = model.vision_tower if hasattr(model, "vision_tower") else None
            if vision_tower is not None:
                model.vision_tower = None
            
            # 로라 적용
            model = get_peft_model(model, peft_config)
            
            # 인코더 붙이기
            model.model.speech_encoder = speech_encoder
            if vision_tower is not None:
                model.model.model.vision_tower = vision_tower

            # model = get_peft_model(model, peft_config)


        # ── Reference model ──
        self.beta = args.beta
        if self.beta == 0.0:
            self.ref_model = None
        elif ref_model is not None:
            self.ref_model = ref_model
        elif is_deepspeed_zero3_enabled():
            self.ref_model = VideoSALMONN2ForCausalLM.from_pretrained(
                model.config._name_or_path,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
            )
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            # PEFT 사용 시 adapter disable로 ref 역할을 함
            self.ref_model = None


        # ── Processing class ──
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model.config._name_or_path, padding_side="left"
            )
        pad_token_id = processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = processing_class.eos_token_id


        # ── Reward functions ──
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs

        if reward_weights is not None:
            self.reward_weights = torch.tensor(reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)


        # ── Generation config ──
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = getattr(args, "temperature", 1.0)
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.9,
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )


        # ── PPO-clip epsilon ──
        # use_grpo가 False면 대신 ppo-clip을 쓰게 합니다..
        self.epsilon_low = getattr(args, "epsilon", 0.2)
        self.epsilon_high = getattr(args, "epsilon_high", None) or self.epsilon_low
        self.use_grpo = getattr(args, "use_grpo", True)


        # ── Data collator --
        # av_dataset.py의 collator 사용
        data_collator = DataCollatorForAVSupervisedDataset(tokenizer=processing_class)


        # suppress FLOPs warning
        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        # 로그용
        model.warnings_issued["estimate_tokens"] = True


        # ── Metrics ──
        self._metrics = defaultdict(list)

        # ── super().__init__ ──
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        # ── Prepare ref_model for deepspeed ──
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )





    # ============================================================
    # Trainer 오버라이드: 기본 전처리 비활성화
    # ============================================================

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _prepare_inputs(self, inputs):
        return inputs



    # ============================================================
    # Per-token log probabilities (VS2 API)
    # ============================================================

    def _get_per_token_logps(
        self, model, input_ids, attention_mask,
        images=None, spectrogram=None, org_groups=None, real_time=None, modalities=None,
        inputs_embeds=None,
    ):
        """model.forward → logits → per-token log probability.
        inputs_embeds가 주어지면 멀티모달 인코딩을 건너뛰고 직접 사용.
        """
        if inputs_embeds is not None:
            # 캐싱된 임베딩 사용 → 인코딩 스킵
            logits = model(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                dpo_forward=False,
                use_cache=False,
            ).logits
        else:
            logits = model(
                input_ids,
                attention_mask=attention_mask,
                images=images,
                spectrogram=spectrogram,
                org_groups=org_groups,
                real_time=real_time,
                modalities=modalities,
                dpo_forward=False,
                use_cache=False,
            ).logits  # (B, L, V)

        logits = logits[:, :-1, :]   # (B, L_logits-1, V)
        input_ids = input_ids[:, 1:]  # (B, L_input-1)
        # 멀티모달 임베딩 확장으로 logits과 input_ids 길이가 다를 수 있음
        # completion 토큰은 시퀀스 끝에 있으므로 뒤에서부터 정렬
        min_len = min(logits.size(1), input_ids.size(1))
        logits = logits[:, -min_len:, :]
        input_ids = input_ids[:, -min_len:]
        # vocab 범위 밖 토큰 방지 (패딩 등)
        vocab_size = logits.size(-1)
        input_ids = input_ids.clamp(0, vocab_size - 1)

        per_token_logps = []

        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(
                log_probs, dim=1, index=input_ids_row.unsqueeze(1)
            ).squeeze(1)
            per_token_logps.append(token_log_prob)

        return torch.stack(per_token_logps)



    # ============================================================
    # GDPO advantage 계산
    # ============================================================

    def _compute_gdpo_advantages(self, rewards_per_func, rewards):
        """
        리워드 함수가 1개면 기본 GRPO 정규화로 fallback.
        """
        num_funcs = rewards_per_func.shape[1]
        device = rewards_per_func.device

        # 리워드 함수
        if num_funcs <= 1:
            # 기본 GRPO
            mean = rewards.view(-1, self.num_generations).mean(dim=1)
            std = rewards.view(-1, self.num_generations).std(dim=1)
            mean = mean.repeat_interleave(self.num_generations, dim=0)
            std = std.repeat_interleave(self.num_generations, dim=0)
            return (rewards - mean) / (std + 1e-4)

        reward_weights = self.reward_weights.to(device)
        rewards_per_func = torch.nan_to_num(rewards_per_func)

        all_adv = []
        for i in range(num_funcs):
            r_i = rewards_per_func[:, i]
            grouped = r_i.view(-1, self.num_generations)
            mean_g = grouped.mean(dim=1, keepdim=True)
            std_g = grouped.std(dim=1, keepdim=True)
            normed = ((grouped - mean_g) / (std_g + 1e-4)).view(-1)
            all_adv.append(normed)

        stacked = torch.stack(all_adv, dim=1)  # (B*G, num_funcs)
        combined = (stacked * reward_weights.unsqueeze(0)).sum(dim=1)
        advantages = (combined - combined.mean()) / (combined.std() + 1e-4)
        return advantages


    # ============================================================
    # compute_loss — 핵심
    # ============================================================

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GDPOTrainer does not support return_outputs=True")

        device = self.accelerator.device


        # ── 입력 추출 ──
        # av_dataset.py의 DataCollatorForAVSupervisedDataset이 만든 배치 딕셔너리
        prompt_ids = inputs["input_ids"].to(device)
        prompt_mask = inputs["attention_mask"].to(device)

        # 멀티모달 입력 (av_dataset.py가 전처리 완료)
        images = inputs.get("images", None)
        spectrogram = inputs.get("spectrogram", None)
        if spectrogram is not None:
            spectrogram = spectrogram.to(device=device, dtype=torch.bfloat16)
        org_groups = inputs.get("org_groups", None)
        real_time = inputs.get("real_time", None)
        modalities = inputs.get("modalities", ["text"])

        # 리워드 계산용 GT
        gt_events = inputs.get("gt_events", [None])[0] or []
        prompt_text = inputs.get("prompts", [""])[0]


        # ── Generate completions ──
        # num_return_sequences는 inputs_embeds와 호환 안 됨 → G번 루프
        all_completion_ids = []
        prompt_length = prompt_ids.size(1)

        # generate 시 gradient_checkpointing 비활성화 (KV 캐시 충돌 방지)
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

        # 멀티모달 인코딩을 1번만 수행하고 캐싱 (SigLIP + Whisper는 deterministic)
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            base_model = unwrapped_model
            while hasattr(base_model, "model"):
                base_model = base_model.model
            with torch.no_grad():
                (_, cached_position_ids, cached_attention_mask, _, cached_inputs_embeds, _) = (
                    base_model.prepare_inputs_labels_for_multimodal(
                        prompt_ids, None, prompt_mask, None, None,
                        images, modalities,
                        spectrogram=spectrogram, org_groups=org_groups, real_time=real_time,
                    )
                )

            # 캐싱된 inputs_embeds로 G번 generate (인코딩 반복 없음)
            for _ in range(self.num_generations):
                gen_ids = unwrapped_model.generate(
                    inputs_embeds=cached_inputs_embeds,
                    position_ids=cached_position_ids,
                    attention_mask=cached_attention_mask,
                    max_new_tokens=self.max_completion_length,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    min_new_tokens=30,
                )
                all_completion_ids.append(gen_ids)

        # gradient_checkpointing 다시 활성화
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        # 길이 맞춰서 패딩 후 결합
        import torch.nn.functional as F
        max_len = max(c.size(1) for c in all_completion_ids)
        pad_id = self.processing_class.pad_token_id or 0
        padded = [F.pad(c, (0, max_len - c.size(1)), value=pad_id) for c in all_completion_ids]
        completion_ids = torch.cat(padded, dim=0)  # (G, max_len)
        prompt_ids = prompt_ids.repeat(self.num_generations, 1)  # (G, P)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (G, P+C)
        prompt_mask = prompt_mask.repeat(self.num_generations, 1)


        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        if completion_ids.size(1) > 0 and is_eos.any():
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)


        # 샘플 요약 로그 
        # 대신 VS2 코드의 출력을 주석처리함.
        gen_lengths = [c.size(1) for c in all_completion_ids]
        print(f"[GDPO STEP] prompt_len={prompt_length}, gen_lengths={gen_lengths}, comp_len={completion_ids.size(1)}")

        # completion 길이 (뒤에서부터 자르기 위해)
        comp_len = completion_ids.size(1)


        # ── 캐싱된 prompt embeds + completion token embeds 결합 ──
        # completion 토큰을 임베딩으로 변환하여 캐싱된 프롬프트 embeds 뒤에 붙임
        base_model_for_embed = model
        while hasattr(base_model_for_embed, "model"):
            base_model_for_embed = base_model_for_embed.model
        with torch.no_grad():
            completion_embeds = base_model_for_embed.embed_tokens(completion_ids)  # (G, C, H)
        # 각 generation에 대해 prompt_embeds + completion_embeds 결합
        cached_prompt_embeds = cached_inputs_embeds.repeat(self.num_generations, 1, 1)  # (G, P_emb, H)
        prompt_completion_embeds = torch.cat([cached_prompt_embeds, completion_embeds], dim=1)  # (G, P_emb+C, H)

        # attention_mask도 embeds 길이에 맞게 재구성
        cached_prompt_mask = cached_attention_mask.repeat(self.num_generations, 1)  # (G, P_emb)
        embeds_attention_mask = torch.cat([cached_prompt_mask, completion_mask], dim=1)  # (G, P_emb+C)

        # ── Per-token log probs (policy) ──
        all_per_token_logps = []
        for g in range(self.num_generations):
            g_logps = self._get_per_token_logps(
                model,
                prompt_completion_ids[g:g+1],
                embeds_attention_mask[g:g+1],
                inputs_embeds=prompt_completion_embeds[g:g+1],
            )
            all_per_token_logps.append(g_logps)
        per_token_logps = torch.cat(all_per_token_logps, dim=0)
        per_token_logps = per_token_logps[:, -comp_len:] if comp_len > 0 else per_token_logps[:, :0]


        # ── Per-token log probs (reference) ──
        if self.beta != 0.0:
            with torch.inference_mode():
                all_ref_logps = []
                for g in range(self.num_generations):
                    if self.ref_model is not None:
                        g_ref_logps = self._get_per_token_logps(
                            self.ref_model,
                            prompt_completion_ids[g:g+1],
                            embeds_attention_mask[g:g+1],
                            inputs_embeds=prompt_completion_embeds[g:g+1],
                        )
                    else:
                        with self.accelerator.unwrap_model(model).disable_adapter():
                            g_ref_logps = self._get_per_token_logps(
                                model,
                                prompt_completion_ids[g:g+1],
                                embeds_attention_mask[g:g+1],
                                inputs_embeds=prompt_completion_embeds[g:g+1],
                            )
                    all_ref_logps.append(g_ref_logps)
                ref_per_token_logps = torch.cat(all_ref_logps, dim=0)
            ref_per_token_logps = ref_per_token_logps[:, -comp_len:] if comp_len > 0 else ref_per_token_logps[:, :0]

            # KL divergence 계산
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )
            if per_token_kl.numel() > 0:
                print(f"[DEBUG KL] kl: min={per_token_kl.min().item():.4f}, max={per_token_kl.max().item():.4f}, has_nan={per_token_kl.isnan().any().item()}")
            else:
                print(f"[DEBUG KL] WARNING: kl is empty!")



        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        print(f"[GDPO SAMPLE] completion[0][:200]: {completions[0][:200]}")
        # Compute the rewards
        rewards_per_func = torch.zeros(len(completions), len(self.reward_funcs), device=device)

        # 프롬프트 및 gt_events를 G번 복제(batchsize=1 가정)
        prompts_repeated = [prompt_text] * self.num_generations
        gt_events_repeated = [gt_events] * self.num_generations #added

        for i, reward_func in enumerate(self.reward_funcs):
            output = reward_func(
                prompts=prompts_repeated,
                completions=completions,
                gt_events=gt_events_repeated,
            )
            rewards_per_func[:, i] = torch.tensor(output, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)


        # ── Compute GDPO advantage ──
        advantages = self._compute_gdpo_advantages(rewards_per_func, rewards)


        # ── Compute Loss ──
        if self.use_grpo:
            # GRPO-style loss
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
            if self.beta != 0.0:
                per_token_loss = -(per_token_loss - self.beta * per_token_kl)
            else:
                per_token_loss = -per_token_loss
            comp_lengths = completion_mask.sum(dim=1).clamp(min=1)
            loss = ((per_token_loss * completion_mask).sum(dim=1) / comp_lengths).mean()
        else:
            # PPO-clip style loss
            coef_1 = torch.exp(per_token_logps - per_token_logps.detach())
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            if self.beta != 0.0:
                per_token_loss = per_token_loss + self.beta * per_token_kl
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()


        # ── 로깅 ──
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            fname = getattr(reward_func, "__name__", f"func_{i}")
            self._metrics[f"rewards/{fname}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        std_grouped = rewards.view(-1, self.num_generations).std(dim=1)
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped).mean().item())

        if self.beta != 0.0:
            comp_lengths_kl = completion_mask.sum(dim=1).clamp(min=1)
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / comp_lengths_kl).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss


    # ============================================================
    # 로깅
    # ============================================================

    def log(self, logs: dict, start_time=None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()



# ============================================================
# 모델 로딩
# ============================================================

def load_model_and_tokenizer(model_path, model_base):
    """
    베이스 모델을 model_base에서 직접 로드하고,
    인코더 분리 → SFT LoRA 로드 → 인코더 재결합 (train.py와 동일 패턴).
    load_qwen_lora_model을 사용하지 않음 (adapter 자동 로드 방지).
    """
    print(f"[GDPO] Loading model")
    print(f"[GDPO]   model_path (SFT ckpt): {model_path}")
    print(f"[GDPO]   model_base: {model_base}")

    # 체크포인트 config
    ckpt_config = {}
    if os.path.isdir(model_path):
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            ckpt_config = json.load(open(config_file))

    # 토크나이저 로드
    tok_path = model_path if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "tokenizer.json")) else model_base
    print(f"[GDPO] Loading tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, model_max_length=4096, padding_side="left")

    # config 구성
    cfg = AutoConfig.from_pretrained(model_base)
    if "model_args" in ckpt_config:
        cfg.model_args = ckpt_config["model_args"]
        for k, v in ckpt_config["model_args"].items():
            if not hasattr(cfg, k):
                setattr(cfg, k, v)
    if not hasattr(cfg, "add_time_token"):
        cfg.add_time_token = ckpt_config.get("add_time_token", False)
    cfg.mm_spatial_pool_mode = "max"
    cfg.mm_spatial_pool_stride = 4
    cfg.mm_spatial_pool_out_channels = 1152
    cfg.mm_newline_position = "grid"
    cfg.mm_patch_merge_type = "spatial_unpad"
    cfg.image_aspect_ratio = "anyres"
    if not hasattr(cfg, "mm_pooling_position"):
        cfg.mm_pooling_position = "after"

    # audio_config
    model_args = ckpt_config.get("model_args", {})
    audio_config = dict(
        audio_visual=model_args.get("audio_visual", True),
        video_fps=model_args.get("fps", 1),
        whisper_path="/workspace/models/whisper-large-v3",
        num_speech_query_token=model_args.get("num_speech_query_token", 25),
        window_level_Qformer=model_args.get("window_level_Qformer", True),
        second_per_window=model_args.get("second_per_window", 0.5),
        second_stride=model_args.get("second_stride", 0.5),
        use_final_linear=model_args.get("use_final_linear", False),
    )

    # 베이스 모델 로드
    print(f"[GDPO] Loading base model from: {model_base}")
    model = VideoSALMONN2ForCausalLM.from_pretrained(
        model_base, config=cfg, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        **audio_config,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(torch.bfloat16)

    # 인코더 분리
    print("[GDPO] Separating encoders before LoRA loading...")
    speech_encoder = model.speech_encoder
    model.speech_encoder = None
    vision_tower = model.vision_tower if hasattr(model, "vision_tower") else None
    if vision_tower is not None:
        model.vision_tower = None

    # LoRA
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.isdir(model_path) and os.path.exists(adapter_config_path):
        print(f"[GDPO] Loading LoRA adapter (LLM only): {model_path}")
        model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
    else:
        print("[GDPO] No LoRA adapter found, using base model")

    # 인코더 붙이기
    print("[GDPO] Re-attaching encoders...")
    model.model.speech_encoder = speech_encoder
    if vision_tower is not None:
        model.model.model.vision_tower = vision_tower

    model = model.to(torch.bfloat16)

    # tokenizer 연결
    if hasattr(model, "model") and hasattr(model.model, "model"):
        model.model.model.tokenizer = tokenizer
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        model.base_model.model.tokenizer = tokenizer
    if not hasattr(model, "tokenizer"):
        model.tokenizer = tokenizer

    print(f"[GDPO] Model loaded successfully")
    return model, tokenizer



# ============================================================
# Config 로딩
# ============================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()
    def _env_replace(match):
        return os.environ.get(match.group(1), match.group(0))
    raw = re.sub(r"\$\{(\w+)\}", _env_replace, raw)
    return yaml.safe_load(raw)



# ============================================================
# 리워드 함수 생성
# ============================================================

def make_reward_functions():
    def _format_reward(completions, **kwargs):
        return [format_reward(c) for c in completions]

    def _label_reward(completions, gt_events=None, **kwargs):
        gt = gt_events or [[] for _ in completions]
        return [label_reward(c, g) for c, g in zip(completions, gt)]

    def _iou_reward(completions, gt_events=None, **kwargs):
        gt = gt_events or [[] for _ in completions]
        return [iou_reward(c, g) for c, g in zip(completions, gt)]

    _format_reward.__name__ = "format"
    _label_reward.__name__ = "label"
    _iou_reward.__name__ = "iou"
    return [_format_reward, _label_reward, _iou_reward]





# ============================================================
# Main (train_gdpo.py 통합)
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GDPO Training (revised)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_base", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--output_dir", default=None)
    cli = parser.parse_args()

    # ── Config 로딩 ──
    cfg = load_config(cli.config) if cli.config else {}

    def _get(cli_val, *keys, default=None):
        if cli_val is not None:
            return cli_val
        d = cfg
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d if d is not None else default

    model_path = _get(cli.model_path, "model", "model_path")
    model_base = _get(cli.model_base, "model", "model_base", default="lmms-lab/llava-onevision-qwen2-7b-ov")
    dataset_path = _get(cli.dataset_path, "data", "dataset_path")
    output_dir = _get(cli.output_dir, "training", "output_dir", default="output/gdpo_revised")

    if model_path is None or dataset_path is None:
        parser.error("--model_path와 --dataset_path 필수 (config.yaml 또는 CLI)")


    # GDPO 파라미터
    num_generations = _get(None, "gdpo", "num_generations", default=8)
    max_completion_length = _get(None, "gdpo", "max_completion_length", default=1024)
    beta = _get(None, "gdpo", "beta", default=0.04)
    reward_weights = _get(None, "gdpo", "reward_weights", default=[0.1, 0.3, 0.6])


    # training 파라미터
    num_epochs = _get(None, "training", "num_train_epochs", default=1)
    batch_size = _get(None, "training", "per_device_train_batch_size", default=1)
    grad_accum = _get(None, "training", "gradient_accumulation_steps", default=4)
    lr = float(_get(None, "training", "learning_rate", default=5e-6))
    warmup = _get(None, "training", "warmup_ratio", default=0.1)
    scheduler = _get(None, "training", "lr_scheduler_type", default="cosine")
    seed = _get(None, "training", "seed", default=2024)

    logging_steps = _get(None, "logging", "logging_steps", default=1)
    save_steps = _get(None, "logging", "save_steps", default=500)
    save_total_limit = _get(None, "logging", "save_total_limit", default=3)


    # Model 
    model, tokenizer = load_model_and_tokenizer(model_path, model_base)


    # Data
    # av_dataset.py의 전처리를 사용하기 위해 data_args 구성
    from dataclasses import dataclass
    @dataclass
    class GDPODataArgs:
        video_fps: int = 1
        max_time: int = 60
        audio_processor: str = "openai/whisper-large-v3"
        image_processor: object = None
        use_timestamps_crop: bool = False
        is_multimodal: bool = True
        mm_use_im_start_end: bool = False
        image_aspect_ratio: str = "square"
        image_grid_pinpoints: str = None
        image_crop_resolution: int = 224
        image_split_resolution: int = 224

    data_args = GDPODataArgs()

    # SigLIP image processor 로드
    from transformers import AutoImageProcessor
    data_args.image_processor = AutoImageProcessor.from_pretrained(
        "google/siglip-so400m-patch14-384"
    )

    print(f"[GDPO] Loading dataset from {dataset_path}")
    dataset = LazyAVSupervisedDataset(
        data_path=dataset_path,
        tokenizer=tokenizer,
        data_args=data_args,
        is_test=False,
    )
    print(f"[GDPO] Dataset size: {len(dataset)}")


    # Reward Function
    reward_funcs = make_reward_functions()
    print(f"[GDPO] Reward weights: {reward_weights}")


    # GRPOConfig
    grpo_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup,
        lr_scheduler_type=scheduler,
        beta=beta,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        bf16=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="tensorboard",
        logging_dir=os.path.join(output_dir, "logs"),
        remove_unused_columns=False,
    )


    # Trainer
    trainer = GDPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_weights=reward_weights,
    )


    # Train
    print("[GDPO] Starting training...")
    trainer.train()
    print(f"[GDPO] Saving model to {output_dir}")
    trainer.save_model(output_dir)
    print("[GDPO] Training complete!")


if __name__ == "__main__":
    main()
    