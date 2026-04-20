#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gdpo_trainer.py (VS2+ / Qwen2.5-VL 버전)
  transformers.Trainer 상속 + GDPO compute_loss 전체 루프.
  VS2+ (video_SALMONN2_plus) 백본 기반.

Usage:
  set -a && source paths.env && set +a
  python _tools/GDPO_revised/gdpo_trainer.py \
    --config _tools/GDPO_revised/config.yaml \
    --model_path ${SFT_CKPT} \
    --model_base ${BASE_MODEL} \
    --dataset_path data/unav100_train_dense.json \
    --output_dir output/gdpo_vs2plus
"""

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
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model





# 프로젝트 루트
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# VS2+ 모델
sys.path.insert(0, os.path.join(PROJECT_ROOT, "video_SALMONN2_plus"))
from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
from qwenvl.data.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset



# reward 함수
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from reward_functions import format_reward, iou_reward

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]





# ============================================================
# 유틸리티
# ============================================================

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
    """GDPO Trainer for VS2+ (Qwen2.5-VL 백본)."""

    def __init__(
        self,
        model: PreTrainedModel,
        reward_funcs: list[RewardFunc],
        args: GRPOConfig = None,
        train_dataset=None,
        processing_class=None,
        ref_model: Optional[PreTrainedModel] = None,
        callbacks=None,
        optimizers=(None, None),
        peft_config: Optional["PeftConfig"] = None,
        reward_weights: Optional[list[float]] = None,
    ):
        if args is None:
            model_name = model.config._name_or_path.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GDPO")


        # PEFT
        # LoRA adapter가 타겟을 llm으로 이미 지정해뒀기 때문에
        # 이전처럼 인코더 분리를 할 필요가 없음.
        if peft_config is not None:
            model = get_peft_model(model, peft_config)



        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            self.ref_model = None
        elif ref_model is not None:
            self.ref_model = ref_model
        elif is_deepspeed_zero3_enabled():
            self.ref_model = video_SALMONN2_plus.from_pretrained(
                model.config._name_or_path,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
            )
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            # PEFT 사용 시 adapter disable로 ref 역할
            self.ref_model = None



        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model.config._name_or_path, padding_side="left"
            )
        pad_token_id = processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = processing_class.eos_token_id


        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs

        if reward_weights is not None:
            self.reward_weights = torch.tensor(reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)


        # Generation config
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


        # PPO-clip epsilon
        self.epsilon_low = getattr(args, "epsilon", 0.2)
        self.epsilon_high = getattr(args, "epsilon_high", None) or self.epsilon_low
        self.use_grpo = getattr(args, "use_grpo", True)


        # Data collator — VS2+
        # VS2의 av_dataset.py 대체하는 LazySupervisedDataset
        data_collator = DataCollatorForSupervisedDataset(tokenizer=processing_class)


        # 로그 정리 용
        model.warnings_issued["estimate_tokens"] = True

        # 메트릭
        self._metrics = defaultdict(list)



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

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )



    # ============================================================
    # Trainer 오버라이드 : 비활성화
    # ============================================================

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _prepare_inputs(self, inputs):
        return inputs

    # ============================================================
    # Per-token log probabilities (VS2+ API)
    # ============================================================

    def _get_per_token_logps(
        self, model, input_ids, attention_mask,
        pixel_values_videos=None, video_grid_thw=None,
        audio_feature=None, audio_lengths=None,
        position_ids=None, second_per_grid_ts=None,
        inputs_embeds=None,
    ):
        """VS2+의 sft_forward
        → logits
        → per-token log probability.

        inputs_embeds가 주어지면 video/audio encoder 재실행 안 함 (캐싱된 embed 재사용).
        rope_index 계산엔 input_ids/video_grid_thw/audio_lengths 필요하므로 여전히 전달.
        """
        # inputs_embeds가 있으면 pixel_values_videos/audio_feature 전달 안 함 → sft_forward가 재인코딩 스킵
        if inputs_embeds is not None:
            pixel_values_videos = None
            audio_feature = None
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            audio_feature=audio_feature,
            audio_lengths=audio_lengths,
            second_per_grid_ts=second_per_grid_ts,
            inputs_embeds=inputs_embeds,
            train_type="sft",
            use_cache=False,
        ).logits

        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        # 길이 정렬
        min_len = min(logits.size(1), input_ids.size(1))
        logits = logits[:, -min_len:, :]
        input_ids = input_ids[:, -min_len:]
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
    # 기존 코드와 동일
    def _compute_gdpo_advantages(self, rewards_per_func, rewards):
        num_funcs = rewards_per_func.shape[1]
        device = rewards_per_func.device

        if num_funcs <= 1:
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

        stacked = torch.stack(all_adv, dim=1)
        combined = (stacked * reward_weights.unsqueeze(0)).sum(dim=1)
        advantages = (combined - combined.mean()) / (combined.std() + 1e-4)
        return advantages



    # ============================================================
    # compute_loss 
    # ============================================================

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GDPOTrainer does not support return_outputs=True")

        device = self.accelerator.device

        # ── 입력 추출 (VS2+ 데이터셋 필드) ──
        prompt_ids = inputs["input_ids"].to(device)
        prompt_mask = inputs["attention_mask"].to(device)

        # GT 답변 제거 — 프롬프트만 추출
        labels = inputs.get("labels", None)
        if labels is not None:
            labels = labels.to(device)
            answer_start = (labels[0] != -100).nonzero(as_tuple=True)[0]
            if len(answer_start) > 0:
                prompt_end_idx = answer_start[0].item()
                prompt_ids = prompt_ids[:, :prompt_end_idx]
                prompt_mask = prompt_mask[:, :prompt_end_idx]


        # 멀티모달 입력
        pixel_values_videos = inputs.get("pixel_values_videos", None)
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(device=device, dtype=torch.bfloat16)
        video_grid_thw = inputs.get("video_grid_thw", None)
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to(device)

        audio_feature = inputs.get("audio_feature", None)
        if audio_feature is not None:
            audio_feature = audio_feature.to(device=device, dtype=torch.bfloat16)
        audio_lengths = inputs.get("audio_lengths", None)

        second_per_grid_ts = inputs.get("second_per_grid_ts", None)
        if second_per_grid_ts is not None:
            second_per_grid_ts = second_per_grid_ts.to(device)


        # 리워드 계산용 GT — gpt 답변에서 시간 구간 추출
        # multi-segment QA 포맷: "From <t0>...<t3> to <t0>...<t6>. From ..."
        # labels에서 GT 답변 토큰을 디코딩 → 시간 구간 리스트로 파싱
        from reward_functions import decode_vtg_time
        gt_intervals = []
        raw_labels = inputs.get("labels", None)
        if raw_labels is not None:
            raw_labels = raw_labels.to(device)
            gt_token_ids = raw_labels[0][raw_labels[0] != -100]
            if len(gt_token_ids) > 0:
                gt_answer = self.processing_class.decode(gt_token_ids, skip_special_tokens=False)
                # "From <t...> to <t...>" 패턴에서 시간 구간 추출
                segments = re.findall(
                    r"[Ff]rom\s+((?:<t\d>)+<tdot><t\d>)\s+to\s+((?:<t\d>)+<tdot><t\d>)",
                    gt_answer
                )
                for start_str, end_str in segments:
                    s = decode_vtg_time(start_str)
                    e = decode_vtg_time(end_str)
                    if s is not None and e is not None and e > s:
                        gt_intervals.append((s, e))


        # ── Generate completions ──
        all_completion_ids = []
        # TODO : 추후 확인 필요
        # VS2+가 generate 반환하는거에 따라 달라질 수 있음
        prompt_length = prompt_ids.size(1)

        # generate 동안 비활성화
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

        # 일단 캐싱 구현 X
        # TODO : generate 반환에 따라 나중에
        # 캐싱 구현 가능성 있음
        # non-None kwargs만 전달 (HF generate의 model_kwargs validation 통과용)
        gen_kwargs = {
            "input_ids": prompt_ids,
            "attention_mask": prompt_mask,
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": 0.9,
        }
        if pixel_values_videos is not None:
            gen_kwargs["pixel_values_videos"] = pixel_values_videos
        if video_grid_thw is not None:
            gen_kwargs["video_grid_thw"] = video_grid_thw
        if audio_feature is not None:
            gen_kwargs["audio_feature"] = audio_feature
        if audio_lengths is not None:
            gen_kwargs["audio_lengths"] = audio_lengths

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # PEFT wrapper 한 겹 벗겨 실제 video_SALMONN2_plus 인스턴스로 generate.
            # LoRA는 q/k/v Linear에 inline으로 붙어있어 base 직접 호출도 LoRA 적용됨.
            raw_model = (
                unwrapped_model.get_base_model()
                if hasattr(unwrapped_model, "get_base_model")
                else unwrapped_model
            )
            for _ in range(self.num_generations):
                gen_ids = raw_model.generate(**gen_kwargs)
                # HF generate는 [prompt+new_tokens] 전체를 반환 → new_tokens만 분리
                gen_ids = gen_ids[:, prompt_length:]
                all_completion_ids.append(gen_ids)

        # generate 끝나고 활성화
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        # 패딩 후 결합
        import torch.nn.functional as F
        max_len = max(c.size(1) for c in all_completion_ids)
        pad_id = self.processing_class.pad_token_id or 0
        padded = [F.pad(c, (0, max_len - c.size(1)), value=pad_id) for c in all_completion_ids]
        completion_ids = torch.cat(padded, dim=0)
        prompt_ids = prompt_ids.repeat(self.num_generations, 1)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        prompt_mask = prompt_mask.repeat(self.num_generations, 1)

        # EOS 마스크
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        if completion_ids.size(1) > 0 and is_eos.any():
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # 로그
        actual_lengths = completion_mask.sum(dim=1).tolist()
        print(f"[GDPO STEP] prompt_len={prompt_length}, actual_lengths={[int(l) for l in actual_lengths]}, comp_len={completion_ids.size(1)}")

        comp_len = completion_ids.size(1)

        # completion에서 비디오/오디오 placeholder 토큰 제거 (logprob 계산 시 feature 수 불일치 방지)
        _VIDEO_TOKEN_ID = 151656
        _AUDIO_TOKEN_ID = 151657 if hasattr(self, '_audio_token_id') else None
        # VIDEO_TOKEN을 패딩으로 교체
        completion_ids_clean = completion_ids.clone()
        completion_ids_clean[completion_ids_clean == _VIDEO_TOKEN_ID] = pad_id
        # 모델 config에서 audio_token_id 확인
        if hasattr(model.config, 'audio_token_id'):
            completion_ids_clean[completion_ids_clean == model.config.audio_token_id] = pad_id

        # ── Per-token log probs (policy) ──
        # 원본 prompt(반복 전) + 각 completion을 개별 결합 (비디오 토큰 수 일치를 위해)
        prompt_ids_single = prompt_ids[:1]  # 반복 전 원본 (1, P)
        prompt_mask_single = prompt_mask[:1]

        # ── Encoding cache (비디오/오디오 encoder는 frozen이므로 step당 1회만 인코딩) ──
        raw_encode_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        cached_video_embeds = None
        cached_audio_embeds = None
        with torch.no_grad():
            if pixel_values_videos is not None:
                pv = pixel_values_videos.type(raw_encode_model.visual.dtype)
                cached_video_embeds = raw_encode_model.visual(pv, grid_thw=video_grid_thw)
            if audio_feature is not None:
                af = audio_feature.type(raw_encode_model.audio.dtype)
                cached_audio_embeds = raw_encode_model.audio(af).flatten(0, 1)

        _AUDIO_TOKEN_ID_CFG = getattr(model.config, "audio_token_id", None)

        def _build_cached_inputs_embeds(pc_ids):
            """pc_ids에 대한 inputs_embeds 생성.
            text 위치는 trainable embed_tokens (gradient 흐름),
            video/audio 위치는 캐시된 frozen embeds (gradient 멈춤 — OK).
            """
            text_embeds = raw_encode_model.model.embed_tokens(pc_ids)
            if cached_video_embeds is not None:
                mask = (pc_ids == _VIDEO_TOKEN_ID)
                mask_exp = mask.unsqueeze(-1).expand_as(text_embeds)
                v = cached_video_embeds.to(text_embeds.device, text_embeds.dtype)
                text_embeds = text_embeds.masked_scatter(mask_exp, v)
            if cached_audio_embeds is not None and _AUDIO_TOKEN_ID_CFG is not None:
                mask = (pc_ids == _AUDIO_TOKEN_ID_CFG)
                mask_exp = mask.unsqueeze(-1).expand_as(text_embeds)
                a = cached_audio_embeds.to(text_embeds.device, text_embeds.dtype)
                text_embeds = text_embeds.masked_scatter(mask_exp, a)
            return text_embeds

        all_per_token_logps = []
        for g in range(self.num_generations):
            pc_ids = torch.cat([prompt_ids_single, completion_ids_clean[g:g+1]], dim=1)
            pc_mask = torch.cat([prompt_mask_single, completion_mask[g:g+1]], dim=1)
            pc_embeds = _build_cached_inputs_embeds(pc_ids)
            g_logps = self._get_per_token_logps(
                model,
                pc_ids,
                pc_mask,
                video_grid_thw=video_grid_thw,          # rope_index용 (재인코딩 X)
                audio_lengths=audio_lengths,             # rope_index용
                second_per_grid_ts=second_per_grid_ts,
                inputs_embeds=pc_embeds,                 # 캐시된 embed 사용
            )
            g_logps = g_logps[:, -comp_len:] if comp_len > 0 else g_logps[:, :0]
            all_per_token_logps.append(g_logps)
        per_token_logps = torch.cat(all_per_token_logps, dim=0)

        # ── Per-token log probs (reference) ──
        if self.beta != 0.0:
            with torch.inference_mode():
                all_ref_logps = []
                for g in range(self.num_generations):
                    pc_ids = torch.cat([prompt_ids_single, completion_ids_clean[g:g+1]], dim=1)
                    pc_mask = torch.cat([prompt_mask_single, completion_mask[g:g+1]], dim=1)
                    pc_embeds = _build_cached_inputs_embeds(pc_ids)
                    if self.ref_model is not None:
                        g_ref_logps = self._get_per_token_logps(
                            self.ref_model,
                            pc_ids,
                            pc_mask,
                            video_grid_thw=video_grid_thw,
                            audio_lengths=audio_lengths,
                            second_per_grid_ts=second_per_grid_ts,
                            inputs_embeds=pc_embeds,
                        )
                    else:
                        with self.accelerator.unwrap_model(model).disable_adapter():
                            g_ref_logps = self._get_per_token_logps(
                                model,
                                pc_ids,
                                pc_mask,
                                video_grid_thw=video_grid_thw,
                                audio_lengths=audio_lengths,
                                second_per_grid_ts=second_per_grid_ts,
                                inputs_embeds=pc_embeds,
                            )
                    g_ref_logps = g_ref_logps[:, -comp_len:] if comp_len > 0 else g_ref_logps[:, :0]
                    all_ref_logps.append(g_ref_logps)
                ref_per_token_logps = torch.cat(all_ref_logps, dim=0)

            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )


        # Decode completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)
        _TIME_TOKENS = {f"<t{i}>" for i in range(10)} | {"<tdot>"}
        _special_to_remove = set(self.processing_class.all_special_tokens) - _TIME_TOKENS
        for tok in _special_to_remove:
            completions = [c.replace(tok, "") for c in completions]
        completions = [re.sub(r"<\|im_start\|>\s*\w+\s*", "", c).strip() for c in completions]

        print(f"[GDPO SAMPLE] completion[0][:200]: {completions[0][:200]}")

        # Compute rewards
        rewards_per_func = torch.zeros(len(completions), len(self.reward_funcs), device=device)
        gt_intervals_repeated = [gt_intervals] * self.num_generations

        for i, reward_func in enumerate(self.reward_funcs):
            output = reward_func(
                completions=completions,
                gt_intervals=gt_intervals_repeated,
            )
            rewards_per_func[:, i] = torch.tensor(output, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)
        advantages = self._compute_gdpo_advantages(rewards_per_func, rewards)


        # ── Loss ──
        if self.use_grpo:
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
            if self.beta != 0.0:
                per_token_loss = -(per_token_loss - self.beta * per_token_kl)
            else:
                per_token_loss = -per_token_loss
            comp_lengths = completion_mask.sum(dim=1).clamp(min=1)
            loss = ((per_token_loss * completion_mask).sum(dim=1) / comp_lengths).mean()
        else:
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
# 모델 로딩 (VS2+)
# ============================================================

def load_model_and_tokenizer(model_path, model_base):
    """VS2+ (Qwen2.5-VL 기반) 모델 로딩."""
    print(f"[GDPO] Loading VS2+ model")
    print(f"[GDPO]   model_path (SFT ckpt): {model_path}")
    print(f"[GDPO]   model_base: {model_base}")

    # 토크나이저 로드
    tok_path = model_path if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "tokenizer.json")) else model_base
    print(f"[GDPO] Loading tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, model_max_length=6000, padding_side="left")

    # 모델 로드
    print(f"[GDPO] Loading base model from: {model_base}")
    model = video_SALMONN2_plus.from_pretrained(
        model_base,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    # resize_token_embeddings 호출 금지 — base는 vocab_size=152064 (Qwen 패딩 포함)로
    # 유지해야 SFT adapter의 modules_to_save(embed_tokens/lm_head, 152064 rows)와 맞음.
    model = model.to(torch.bfloat16)

    # LoRA 로드 — audio.layers 분리 필요 (SFT train_qwen.py와 동일 패턴)
    # adapter_config의 base_model_name_or_path는 무시됨 (이미 로드된 model 객체를 넘기므로).
    # modules_to_save=[model.embed_tokens, lm_head]는 PeftModel 로딩 시 자동 복원되므로
    # time token 임베딩 수동 복원 불필요.
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.isdir(model_path) and os.path.exists(adapter_config_path):
        print(f"[GDPO] Loading LoRA adapter: {model_path}")
        audio_layers = model.audio.layers
        del model.audio.layers
        model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
        model.model.audio.layers = audio_layers
    else:
        print("[GDPO] No LoRA adapter found, using base model")

    # gradient_checkpointing + 얼린 base 조합에서 grad가 흐르게 하려면
    # 얼린 embedding 출력에 requires_grad=True가 필요.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # HF generate의 _validate_model_kwargs 우회 (monkey-patch).
    # PEFT(LoraModel) wrapping 때문에 HF가 forward 시그니처를 inspect할 때
    # `*args, **kwargs`만 보여서 pixel_values_videos / video_grid_thw /
    # audio_feature / audio_lengths를 "unused"로 false-positive 판정함.
    # 실제로는 base_model.forward / prepare_inputs_for_generation이 정상 처리함.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base._validate_model_kwargs = lambda *a, **kw: None

    # tokenizer 연결
    if not hasattr(model, "tokenizer"):
        model.tokenizer = tokenizer

    print(f"[GDPO] VS2+ model loaded successfully")
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
# 리워드 함수
# ============================================================

def make_reward_functions():
    def _format_reward(completions, **kwargs):
        return [format_reward(c) for c in completions]

    def _iou_reward(completions, gt_intervals=None, **kwargs):
        gt = gt_intervals or [[] for _ in completions]
        return [iou_reward(c, g) for c, g in zip(completions, gt)]

    _format_reward.__name__ = "format"
    _iou_reward.__name__ = "iou"
    return [_format_reward, _iou_reward]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GDPO Training (VS2+)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_base", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--output_dir", default=None)
    cli = parser.parse_args()

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
    model_base = _get(cli.model_base, "model", "model_base")
    dataset_path = _get(cli.dataset_path, "data", "dataset_path")
    output_dir = _get(cli.output_dir, "training", "output_dir", default="output/gdpo_vs2plus")

    if model_path is None or dataset_path is None:
        parser.error("--model_path와 --dataset_path 필수")

    # GDPO 파라미터
    num_generations = _get(None, "gdpo", "num_generations", default=8)
    max_completion_length = _get(None, "gdpo", "max_completion_length", default=512)
    beta = _get(None, "gdpo", "beta", default=0.04)
    reward_weights = _get(None, "gdpo", "reward_weights", default=[1.0, 1.0, 1.0])

    # 학습 파라미터
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

    # Dataset — VS2+ LazySupervisedDataset 사용
    print(f"[GDPO] Loading dataset from {dataset_path}")
    from dataclasses import dataclass
    from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast

    from transformers import WhisperFeatureExtractor

    @dataclass
    class GDPODataArgs:
        dataset_use: str = ""
        model_type: str = "qwen2.5vl"
        video_max_frames: int = 128         # SFT와 동일
        video_min_frames: int = 4
        base_interval: float = 2
        max_pixels: int = 176400            # SFT와 동일 (28*28*225)
        min_pixels: int = 784               # SFT와 동일 (28*28)
        video_max_frame_pixels: int = 25088 # SFT와 동일
        video_min_frame_pixels: int = 3136  # SFT와 동일
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
        audio_processor: object = None      # SFT에서 필요

    data_args = GDPODataArgs()
    data_args.dataset_use = dataset_path
    data_args.image_processor = Qwen2VLImageProcessorFast.from_pretrained(model_base)
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size,
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length,
    )

    dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_args=data_args,
    )
    print(f"[GDPO] Dataset size: {len(dataset)}")

    # Reward
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
