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
  python -m _tools.GDPO_v2_revised.gdpo_trainer --config config.yaml

  # CLI 직접 지정
  python -m _tools.GDPO_v2_revised.gdpo_trainer \\
    --model_path ${SFT_CKPT} --model_base ${BASE_MODEL} \\
    --dataset_path data/unav100_train_gdpo.json

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

'''
if is_wandb_available():
    import wandb
'''




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

    Time-R1/Omni-R1과 동일한 패턴:
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
        if peft_config is not None:
            model = get_peft_model(model, peft_config)


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
        images, spectrogram, org_groups, real_time, modalities,
    ):
        """model.forward → logits → per-token log probability.
        Time-R1/Omni-R1과 동일한 루프 방식.
        images, spectrogram, org_groups, real_time, modalities, <- VS2 대응
        """
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

        logits = logits[:, :-1, :]   # (B, L-1, V)
        input_ids = input_ids[:, 1:]  # (B, L-1)

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
            spectrogram = spectrogram.to(device)
        org_groups = inputs.get("org_groups", None)
        real_time = inputs.get("real_time", None)
        modalities = inputs.get("modalities", ["text"])

        # 리워드 계산용 GT
        gt_events = inputs.get("gt_events", [None])[0] or []
        prompt_text = inputs.get("prompts", [""])[0]


        # ── Generate completions ──
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                images=images,
                spectrogram=spectrogram,
                org_groups=org_groups,
                real_time=real_time,
                modalities=modalities,
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9,
                num_return_sequences=self.num_generations,
            )

        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)


        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)


        # ── 멀티모달 텐서 G배 복제 ──
        # timer1의
        # pixel_values_videos = prompt_inputs["pixel_values_videos"].repeat(
        #     self.num_generations, 1
        # )
        # video_grid_thw = prompt_inputs["video_grid_thw"].repeat_interleave(
        #     self.num_generations, dim=0
        # )
        # 입니다
        images_repeated = images * self.num_generations if images is not None else None
        spec_repeated = spectrogram.repeat(self.num_generations, 1, 1) if spectrogram is not None else None
        org_groups_repeated = org_groups * self.num_generations if org_groups is not None else None
        real_time_repeated = real_time * self.num_generations if isinstance(real_time, list) else None
        modalities_repeated = modalities * self.num_generations




        # ── Per-token log probs (policy) ──
        per_token_logps = self._get_per_token_logps(
            model, prompt_completion_ids, attention_mask,
            images_repeated, spec_repeated,
            org_groups_repeated, real_time_repeated, modalities_repeated,
        )
        per_token_logps = per_token_logps[:, prompt_length - 1:]


        # ── ⑤ Per-token log probs (reference) ──
        if self.beta != 0.0:
            with torch.inference_mode():
                if self.ref_model is not None:
                    # ref 모델 따로 들고왔을 시
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask,
                        images_repeated, spec_repeated,
                        org_groups_repeated, real_time_repeated, modalities_repeated,
                    )
                else:
                    # 따로 들고올 필요 없이 LoRA 스위치 끄고 원본 사용
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            model, prompt_completion_ids, attention_mask,
                            images_repeated, spec_repeated,
                            org_groups_repeated, real_time_repeated, modalities_repeated,
                        )
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
            # KL divergence 계산
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )



        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        prompts_repeated = [prompt_text] * self.num_generations
        # Compute the rewards
        rewards_per_func = torch.zeros(len(completions), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            output = reward_func(
                prompts=prompts_repeated,
                completions=completions,
                gt_events=gt_events,
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
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
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
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
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
    is_local_ckpt = os.path.isdir(model_path)

    if is_local_ckpt:
        config_path = os.path.join(model_path, "config.json")
        ckpt_config = json.load(open(config_path)) if os.path.exists(config_path) else {}
        tok_path = model_path if os.path.exists(os.path.join(model_path, "tokenizer.json")) else model_base
    else:
        ckpt_config = {}
        tok_path = model_base

    # tokenizer
    print(f"[GDPO] Loading tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, model_max_length=4096, padding_side="left")

    # base config
    print(f"[GDPO] Loading base config from: {model_base}")
    cfg = AutoConfig.from_pretrained(model_base)
    if "model_args" in ckpt_config:
        cfg.model_args = ckpt_config["model_args"]
    if "add_time_token" in ckpt_config:
        cfg.add_time_token = ckpt_config["add_time_token"]
    elif not hasattr(cfg, "add_time_token"):
        cfg.add_time_token = False

    # base model
    print(f"[GDPO] Loading base model from: {model_base}")
    model = VideoSALMONN2ForCausalLM.from_pretrained(
        model_base, config=cfg, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))

    # LoRA adapter
    # 없으면 base model
    adapter_config = os.path.join(model_path, "adapter_config.json") if is_local_ckpt else ""
    if is_local_ckpt and os.path.exists(adapter_config):
        print(f"[GDPO] Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
        model = model.to(torch.bfloat16)
    else:
        print("[GDPO] No LoRA adapter found, using base model")

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
        gt = gt_events or []
        return [label_reward(c, gt) for c in completions]

    def _iou_reward(completions, gt_events=None, **kwargs):
        gt = gt_events or []
        return [iou_reward(c, gt) for c in completions]

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
    lr = _get(None, "training", "learning_rate", default=5e-6)
    warmup = _get(None, "training", "warmup_ratio", default=0.1)
    scheduler = _get(None, "training", "lr_scheduler_type", default="cosine")
    seed = _get(None, "training", "seed", default=2024)

    logging_steps = _get(None, "logging", "logging_steps", default=1)
    save_steps = _get(None, "logging", "save_steps", default=500)
    save_total_limit = _get(None, "logging", "save_total_limit", default=3)


    # Model (데이터셋보다 먼저 로드 — tokenizer 필요)
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
