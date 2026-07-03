# 06.27 batch
# 진단가이드 문서 남겨놨습니다.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gdpo_trainer.py (VS2+ / Qwen2.5-VL 버전) — 팀 공용 통합 트레이너

  ★ 통합본 — 그동안 분리돼 있던 변형(clip / sep2 / clip2_sep / CoT)을 단일 트레이너로
    흡수. 모든 기능은 config/CLI 플래그로 켜고 끄며, 기본값(config.yaml)은 기존 GDPO 와
    수치적으로 동일하게 동작한다(아래 "기본값 = 기존 GDPO" 참고).
    레거시 변형 파일(gdpo_trainer_clip.py / _sep2.py / _clip2_sep.py / _f1_cot.py)은
    버전관리용으로 보존하되, 신규 학습은 이 파일을 쓴다.

    통합된 기능:
      - [clip-higher] DAPO clipped surrogate (ε_high>ε_low). config.clip.*.
        clipped surrogate loss = -min( r·A, clip(r, 1-ε_low, 1+ε_high)·A ).
      - [μ rollout 재사용] num_iterations(μ)>1 이면 한 rollout 을 μ 번 정책 업데이트에
        재사용 → r≠1 이라야 clip 이 binding. μ=1 이면 기존 GDPO 와 수치 동일.
      - [grad_accum 호환] block-repeat sampler([p0..p_{N-1}]×μ)로 HF 가 N microbatch
        마다 optimizer.step() → **μ iteration 사이에 policy 가 실제로 업데이트**되어
        grad_accum>1 이어도 clip 의 r≠1 이 유지된다. effective batch = grad_accum(N) × num_gpu.
        (단순 HF grad_accum 은 μ 재사용이 한 step 에 묶여 r≈1 → clip 무력. 그래서 순서를
         block-repeat 로 바꾸는 게 핵심.)
      - [rollout 캐시] prompt-시그니처 키 캐시(슬롯/카운터 X). microbatch 순서가
        DDP/블록반복으로 흩어져도 잘못된 재사용/크래시 없음(불일치=miss→재생성).
        μ 도달 엔트리는 즉시 evict → 캐시 크기 ~grad_accum. old_logps 는 fresh 시점 1회
        스냅샷, reuse 는 그 R 그대로.
      - [sep2] gdpo.multi_seg_weight(기본 1.0=off): GT>=2 prompt advantage ×w (single
        prior 상쇄). clip-higher on/off 모두 호환(advantage 공통 경로).
        _compute_gdpo_advantages 의 prompt별 num_generations 그룹 정규화는 grad_accum 무관.
      - [동적 reward 채널] --reward_module 로 채널 수/이름 가변(REWARD_CHANNELS 선언 또는
        함수명 규약). CoT(<think>/<answer>) reward 모듈도 동일 경로로 지원 — 트레이너는
        raw completion 만 넘기고 채널 파싱은 reward 모듈이 담당. 별도 CoT 트레이너 불필요.
      - [modules_to_save 분리] lora.train_embeddings / lora.train_lm_head 개별 토글.
      - [진단 로깅] ratio / clip_frac / gen_entropy / advantage 통계(wandb).
      - [loss 이중스케일 없음] HF(training_step)가 loss/=grad_accum 1회만 수행, 본 코드는
        나누지 않음(per-prompt 평균 반환).

    기본값 = 기존 GDPO: config.yaml(num_iterations=1, multi_seg_weight=1.0,
      train_lm_head=false)로 돌리면 통합 전 동작과 수치 동일. clip-higher / sep2 등은
      config_clip*.yaml / config_*sep*.yaml / CLI override 로 켠다.

────────────────────────────────────────────────────────
  [cdh/gdpo_trainer.py에 hyj/gdpo_trainer_rM_fp.py 의 최신 로직을 융합한버전]
  cdh 베이스에서 유지:
    - tti_mode (off/on): base config 의 time_token_id_range 활성/비활성 처리
    - --reward_module 동적 선택 (reward_functions | reward_functions_rM_fp)
    - SFT adapter 이어학습 시 RL trainable 제한 (q/k/v LoRA + embed/lm_head)
    - generate 중 GC 강제 ON/OFF (모듈 순회, use_reentrant=False)
    - 상위호환 ref 로직 (_is_peft_model 분기)
  hyj 에서 이식:
    - 머지 모델 위 fresh RL LoRA 학습 (config.lora.enabled → peft_config)
    - generation: top_k=50 / repetition_penalty=1.0 (다양성 + time-token 반복 허용)
    - resume_from_checkpoint 지원
    - rank-0 한정 로깅 + raw/sec segment 디버그 출력
    - get_peft_model 후 enable_input_require_grads 재적용
  clip 버전(gdpo_trainer_clip.py)에서 통합:
    - DAPO clip-higher (ε_high>ε_low) clipped surrogate loss (config.clip.*)
    - μ(num_iterations)>1 rollout 재사용 — _RepeatSampler + rollout 캐시
      (_generate_rollout/_completion_logps 분리). μ=1 이면 기존 GDPO 와 수치 동일.
    - ratio / clip_frac 진단 메트릭 로깅
    - reward 채널은 가변(CoT 포함)으로 유지 — clip 경로와 무관하게 동작
    ⚠️ clip-higher 는 공용 기본값에서 OFF (config.yaml: clip.num_iterations=1 →
       μ=1 이라 clip 미작동 = 기존 GDPO 와 수치 동일). 켜려면 clip.num_iterations>1.
       [통합] grad_accum>1 과 병행 가능(block-repeat sampler) — 더는 grad_accum=1 강제 아님.
       (config_clip.yaml / config_*sep*.yaml / CLI override.)

두 가지 학습 경로 (model_path 가 무엇이냐로 자동 분기):
  (A) adapter 이어학습 : model_path=SFT LoRA adapter, model_base=time-token base
                          → adapter 로드 + RL trainable 제한. ref = pre-SFT base.
  (B) fresh LoRA       : model_path=model_base=SFT-머지 모델, config.lora.enabled=true
                          → 머지 weight 위 새 LoRA. ref = SFT 정책(LoRA disable).

Usage:
  # 0) 환경 준비 — CKPT_DIR / TRAIN_DIR / WANDB_* 는 paths.env 로 이미 환경에 export 돼 있음
  #    (모델 경로는 아래처럼 CLI 인자로 직접 전달. SFT_CKPT/BASE_MODEL export 불필요.)
  conda activate salmonn2p

  # 1) 기본 (모델/데이터 경로를 CLI 로 직접 지정)
  python _tools/GDPO/gdpo_trainer.py \
      --config       _tools/GDPO/config.yaml \
      --model_path   ${CKPT_DIR}/sft/salmonn2p_7b_unav_fps5_off \
      --model_base   ${CKPT_DIR}/video_salmonn2_plus_7B_time_tokens \
      --dataset_path ${TRAIN_DIR}/unav100_v2.json

  # 2) 멀티 GPU
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  torchrun --standalone --nproc_per_node=4 \
      _tools/GDPO/gdpo_trainer.py \
      --config       _tools/GDPO/config.yaml \
      --model_path   ${CKPT_DIR}/sft/salmonn2p_7b_unav_fps5_off \
      --model_base   ${CKPT_DIR}/video_salmonn2_plus_7B_time_tokens \
      --dataset_path ${TRAIN_DIR}/unav100_v2.json

  # 3) 모든 CLI 인자를 명시 (CLI > config.yaml 우선)
  python _tools/GDPO/gdpo_trainer.py \
      --config       _tools/GDPO/config.yaml \
      --model_path   ${CKPT_DIR}/sft/salmonn2p_7b_unav_fps5_off \  # 경로A=SFT LoRA adapter / 경로B=SFT-머지 모델 (필수)
      --model_base   ${CKPT_DIR}/video_salmonn2_plus_7B_time_tokens \  # time-token VS2+ base
      --dataset_path ${TRAIN_DIR}/unav100_v2.json \  # 학습 JSON (필수)
      --output_dir   ${CKPT_DIR}/gdpo/salmonn2p_7b_unav_rM \  # 체크포인트/로그 저장 (미지정→output/gdpo_vs2plus)
      --max_steps    1000 \                  # >0 이면 num_train_epochs 무시 (smoke-test/길이제한)
      --run_name     gdpo_rM_run1 \          # wandb/tracker run 이름 (미지정→output_dir basename)
      --tti_mode     off \                   # off | on  (모델 time_token_id_range ↔ 데이터 마커 짝)
      --reward_module reward_functions \     # reward 구현 교체 (아래 표)
      --resume_from_checkpoint True          # True=output_dir 최신 자동재개 / 경로=특정 / 미지정=처음부터

  # 4) 빠른 동작 확인 (3 step)
  python _tools/GDPO/gdpo_trainer.py \
      --config       _tools/GDPO/config.yaml \
      --model_path   ${CKPT_DIR}/sft/salmonn2p_7b_unav_fps5_off \
      --model_base   ${CKPT_DIR}/video_salmonn2_plus_7B_time_tokens \
      --dataset_path ${TRAIN_DIR}/unav100_v2.json \
      --max_steps 3

  # 5) 체크포인트에서 재개 (모델/데이터 경로는 동일하게 전달)
  python _tools/GDPO/gdpo_trainer.py \
      --config       _tools/GDPO/config.yaml \
      --model_path   ${CKPT_DIR}/sft/salmonn2p_7b_unav_fps5_off \
      --model_base   ${CKPT_DIR}/video_salmonn2_plus_7B_time_tokens \
      --dataset_path ${TRAIN_DIR}/unav100_v2.json \
      --output_dir   ${CKPT_DIR}/gdpo/salmonn2p_7b_unav_rM \
      --resume_from_checkpoint ${CKPT_DIR}/gdpo/salmonn2p_7b_unav_rM/checkpoint-500

  # 6) 백그라운드 실행 (SSH 끊겨도 유지) — setsid+nohup 으로 세션 분리 + HUP 무시.
  #    학습 로그는 output_dir/train.log 로 남으므로 콘솔 출력은 /dev/null 로 버림.
  #    (관리: tail -f $OUT/train.log / ps -ef|grep gdpo_trainer / pkill -f gdpo_trainer.py)
  OUT=${CKPT_DIR}/gdpo/salmonn2p_7b_unav_rM
  setsid nohup python _tools/GDPO/gdpo_trainer.py \
      --config       _tools/GDPO/config.yaml \
      --model_path   ${CKPT_DIR}/sft/salmonn2p_7b_unav_fps5_off \
      --model_base   ${CKPT_DIR}/video_salmonn2_plus_7B_time_tokens \
      --dataset_path ${TRAIN_DIR}/unav100_v2.json \
      --output_dir   "$OUT" \
      < /dev/null > /dev/null 2>&1 &

CLI 인자 (모두 선택; 미지정 시 config.yaml → 내부 기본값 순으로 fallback):
  --config                  config.yaml 경로. 나머지 인자/하이퍼파라미터의 기본 소스.
  --model_path              [필수] SFT adapter(경로A, adapter_config.json 존재) 또는
                            SFT-머지 모델(경로B). 어느 쪽이냐로 학습 경로 자동 분기.
                            (config: model.model_path)
  --model_base              VS2+ time-token base 모델. (config: model.model_base)
  --dataset_path            [필수] UnAV-100 multi-segment QA JSON. (config: data.dataset_path)
  --output_dir              체크포인트/로그 저장 경로. (config: training.output_dir / 기본 output/gdpo_vs2plus)
  --max_steps   int         step 수 제한. >0 이면 num_train_epochs 무시. (config: training.max_steps)
  --run_name    str         wandb/tracker run 이름. (config: logging.run_name / 기본 output_dir basename)
  --tti_mode    off|on      TTI 모드. off=time_token_id_range 무시(rope OFF 분기) /
                            on=base config 유지(rope ON 분기, 데이터에 마커 필요).
                            (config: model.tti_mode / 기본 off)
  --reward_module str       reward 함수 모듈명. (config: reward.module / 기본 reward_functions)
                              reward_functions       → 채널 [format, iou(=MUSEG r_M)]
                              reward_functions_rM_fp  → iou = r_M - K*n_unmatched_pred (FP penalty)
                              reward_functions_prec   → iou = r_M - λ*outside_ratio (precision penalty)
                              reward_functions_rM_cov → iou = r_M - γ*coverage_excess (길이편향 억제)
                              reward_functions_f1     → iou = F1(best-match) 베이스라인
                              reward_functions_rM_sep → 채널 [format, len, global(F1), local]
                                                        (학습신호 세분화 — multi-seg 붕괴 대응)
                            ── 채널 호환 (load_reward_module) ──
                            모듈이 REWARD_CHANNELS=[(name,fn,needs_gt),...] 를 선언하면 그대로
                            사용(임의 채널 수/이름). 없으면 format_reward/iou_reward(+옵션
                            timestamp/modality) convention 으로 조립. → 새 채널 reward 를
                            추가해도 trainer 무수정. reward_weights 길이는 채널 수에 자동정렬,
                            wandb 는 rewards/<채널명> 으로 자동 로깅.
  --resume_from_checkpoint  'True'=output_dir 내 최신 체크포인트 자동 재개 /
                            <경로>=특정 체크포인트 / 미지정=처음부터.

학습 경로별 주의:
  - 경로A(adapter 이어학습): config 의 lora.enabled=false 로 둘 것 (PeftModel 이중 적용 방지).
  - 경로B(fresh LoRA)     : config 의 lora.enabled=true.

config 파일(모두 이 통합 트레이너로 실행):
  - config.yaml            : 팀 공용 기본 (num_iterations=1, multi_seg_weight=1.0 → 기존 GDPO 동일)
  - config_clip.yaml       : clip-higher 실험 (μ=2)
  - config_clip_sep.yaml   : 세분화 reward(reward_functions_rM_sep, 4채널) + clip-higher
  - config_clip2_sep2.yaml : 세분화 reward(sep2) + clip-higher + grad_accum>1(block-repeat)
  - config_clip2_sep2_mu1.yaml : sep2 + grad_accum>1, clip OFF(μ=1, 순수 GDPO + 큰 배치)
  - config_v2_rM_fp.yaml   : reward_functions_rM_fp (r_M + FP 페널티)
  - config_cot*.yaml       : CoT(<think>/<answer>) reward(reward_functions_*cot*) + CoT 데이터셋

clip-higher (config.clip.*):
  clipped surrogate loss = -min( r·A, clip(r, 1-ε_low, 1+ε_high)·A ).  μ(num_iterations)>1 이라야
  r=exp(new-old)≠1 → clip 이 실제 binding.
  [통합] grad_accum>1 도 지원 — block-repeat sampler 가 μ iteration 사이 optimizer.step() 을
  보장하므로 grad_accum=1 강제 아님(effective batch = grad_accum × num_gpu).

디버그 env:
  TTI_DEBUG=1 [TTI_DEBUG_STEPS=3]   rank-0 한정, 첫 N step 동안 TTI 정합성 계측 출력.
  LEN_REWARD_MODE=binary|graded     reward_functions_rM_sep 의 len 채널 모드 (기본 binary).
                                    graded=1-|n_pred-n_gt|/max — near-collapse 에서 gradient 생존.
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
    from peft import PeftConfig, PeftModel, get_peft_model, LoraConfig





# 프로젝트 루트
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# VS2+ 모델
sys.path.insert(0, os.path.join(PROJECT_ROOT, "video_SALMONN2_plus"))
from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
from qwenvl.data.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset



# reward 함수 — 어떤 모듈을 쓸지는 런타임에 --reward_module 로 선택(load_reward_module).
# trainer 를 reward 마다 복제하지 않고 import 대상만 바꾸기 위함.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)


def load_reward_module(module_name: str):
    """reward 모듈을 동적 로드해 채널 spec 리스트 [(name, fn, needs_gt), ...] 반환.

    채널 호환 규약 (두 가지 — 모듈이 어느 쪽이든 자유롭게 선택):
      (1) REWARD_CHANNELS 선언 모듈 (권장, 임의 채널 수/이름)
            모듈이 REWARD_CHANNELS = [(name:str, fn:callable, needs_gt:bool), ...] 를
            정의하면 그대로 사용한다. 채널 개수·이름·순서는 모듈이 전적으로 결정.
            예) reward_functions_rM_sep → [format, len, global, local]
            (needs_gt=True 면 fn(completion, gt) 로, False 면 fn(completion) 로 호출됨.)
      (2) convention 모듈 (레거시, 하위호환)
            REWARD_CHANNELS 가 없으면 함수명 규약으로 조립:
              필수: format_reward, iou_reward
              옵션: timestamp_reward, modality_reward (CoT 모듈에만; 없으면 무시)
            예) reward_functions / reward_functions_rM_fp(_cot)

    trainer 본체는 반환된 spec(이름/개수)에 무관하게 동작한다:
      - reward_weights 길이는 채널 수에 맞춰 main()에서 자동조정
      - wandb 메트릭은 채널 __name__ 으로 자동 로깅(rewards/<name>)
    → 새 채널을 가진 reward 모듈을 추가해도 trainer 수정 불필요.
    """
    import importlib
    mod = importlib.import_module(module_name)

    # (1) 모듈이 채널 스펙을 직접 선언 → 그대로 사용 (임의 채널 호환)
    raw = getattr(mod, "REWARD_CHANNELS", None)
    if raw is not None:
        specs = []
        for ch in raw:
            if not (isinstance(ch, (tuple, list)) and len(ch) == 3):
                raise ValueError(
                    f"[GDPO] {module_name}.REWARD_CHANNELS 항목은 (name, fn, needs_gt) "
                    f"3-tuple 이어야 함: {ch!r}")
            name, fn, needs_gt = ch
            if not callable(fn):
                raise ValueError(f"[GDPO] reward 채널 '{name}' 의 fn 이 callable 이 아님: {fn!r}")
            specs.append((str(name), fn, bool(needs_gt)))
        if not specs:
            raise ValueError(f"[GDPO] {module_name}.REWARD_CHANNELS 가 비어 있음")
        return specs

    # (2) 레거시 convention fallback (기존 모듈 무영향)
    specs = [("format", mod.format_reward, False),   # (name, fn, needs_gt)
             ("iou", mod.iou_reward, True)]
    for name in ("timestamp", "modality"):           # 옵션 (CoT)
        fn = getattr(mod, f"{name}_reward", None)
        if fn is not None:
            specs.append((name, fn, True))
    return specs

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


# ============================================================
# TTI 디버그 계측 (env TTI_DEBUG=1 로 활성; rank-0 한정, 첫 N step만)
#   FP_PENALTY_K 와 동일한 env 패턴. 평상시(off) 런엔 영향 0.
# ============================================================
TTI_DEBUG = os.environ.get("TTI_DEBUG", "0").lower() in ("1", "true", "yes", "on")
TTI_DEBUG_STEPS = int(os.environ.get("TTI_DEBUG_STEPS", "3"))   # compute_loss 첫 N step만


def _is_rank0() -> bool:
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0"


def _tti_dbg(msg: str):
    """TTI_DEBUG on + rank-0 일 때만 출력."""
    if TTI_DEBUG and _is_rank0():
        print(f"[TTI-DBG] {msg}", flush=True)


def count_time_markers(ids, id_range) -> int:
    """ids(1D tensor 또는 리스트) 중 time-token id_range [lo,hi] 안에 드는 토큰 수."""
    if id_range is None:
        return 0
    lo, hi = id_range
    if isinstance(ids, torch.Tensor):
        return int(((ids >= lo) & (ids <= hi)).sum().item())
    return sum(1 for x in ids if lo <= int(x) <= hi)


# ── resume RNG 로딩 우회 (torch>=2.4 weights_only=True 기본값 대응) ──
# checkpoint 의 rng_state_*.pth 는 np.random.get_state()(numpy 객체)를 담는데,
# torch.load(weights_only=True) 가 numpy global 언피클을 거부 →
# transformers._load_rng_state 가 resume 시 UnpicklingError 로 크래시.
# 신뢰된 자체 체크포인트이므로 numpy 관련 global 을 allowlist 해 정상 로드.
try:
    import numpy as _np
    import torch.serialization as _ts
    import _codecs as _cod
    # _codecs.encode: numpy random state 직렬화에 포함됨 → rng_state_*.pth 로드 시 필요
    _safe = [_np.ndarray, _np.dtype, _cod.encode]
    for _m in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            _mod = __import__(_m, fromlist=["_reconstruct", "scalar"])
            _safe += [getattr(_mod, "_reconstruct", None), getattr(_mod, "scalar", None)]
        except Exception:
            pass
    try:
        import numpy.dtypes as _ndt
        _safe += [getattr(_ndt, _n) for _n in dir(_ndt) if _n.endswith("DType")]
    except Exception:
        pass
    _ts.add_safe_globals([x for x in _safe if x is not None])
except Exception as _e:
    print(f"[GDPO] add_safe_globals(numpy) 실패(무시): {_e}")





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
# RepeatSampler (μ>1 rollout 재사용용)
# ============================================================

class _RepeatSampler:
    """base sampler 의 각 인덱스를 num_repeat 번 연속 반복해 내보내는 래퍼.

    μ>1 일 때 dataloader 가 (i,i,j,j,...) 처럼 같은 샘플을 μ 번 연속 주도록 해서,
    compute_loss 가 첫 호출에만 rollout 을 생성하고 이후 μ-1 번은 캐시를 재사용해도
    distinct 샘플을 스킵/낭비하지 않게 한다. (TRL GRPOTrainer 의 RepeatSampler 와 동일 취지)
    """
    def __init__(self, base_sampler, num_repeat: int):
        self.base_sampler = base_sampler
        self.num_repeat = max(1, int(num_repeat))

    def __len__(self):
        return len(self.base_sampler) * self.num_repeat

    def __iter__(self):
        for idx in self.base_sampler:
            for _ in range(self.num_repeat):
                yield idx


class _BlockRepeatSampler:
    """base sampler 를 block_size(N)개씩 묶어, 각 블록을 num_repeat(μ)번 연속 반복.
       → [p0..p_{N-1}] ×μ,  [pN..p_{2N-1}] ×μ, ...  (마지막 부분 블록 <N 은 drop)

    grad_accum=N 과 함께 쓰면 microbatch 순서가 'i,j,i,j'(블록반복) 가 되어, HF 가
    N microbatch 마다 optimizer.step() 할 때 **μ iteration 사이에 policy 가 실제로
    업데이트**된다(= PPO/GRPO clip 의 r≠1 보장). N=1 → _RepeatSampler 와 동일(각 인덱스
    μ번). μ=1 → base 를 N청크로 1회 = 표준 grad_accum(블록 누적)."""
    def __init__(self, base_sampler, block_size: int, num_repeat: int):
        self.base_sampler = base_sampler
        self.N = max(1, int(block_size))
        self.mu = max(1, int(num_repeat))

    def __len__(self):
        n_full = len(self.base_sampler) // self.N      # 부분 블록 drop
        return n_full * self.N * self.mu

    def __iter__(self):
        block = []
        for idx in self.base_sampler:
            block.append(idx)
            if len(block) == self.N:
                for _ in range(self.mu):
                    for j in block:
                        yield j
                block = []
        # 마지막 부분 블록(<N)은 버린다 (캐시 정렬 단순화)


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
            # [hyj] gradient_checkpointing + 얼린 base 조합에서 fresh LoRA 로 grad 가
            # 흐르도록 임베딩 출력에 requires_grad 재적용 (fresh LoRA 경로 필수).
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()



        # Reference model
        self.beta = args.beta
        # PEFT 모델이면 disable_adapter()로 ref 역할 → ref_model 메모리 절약(~14GB).
        # compute_loss에 이미 disable_adapter 분기가 구현돼 있음.
        _is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        if self.beta == 0.0:
            self.ref_model = None
        elif ref_model is not None:
            self.ref_model = ref_model
        elif _is_peft_model:
            # PEFT 사용 시 adapter disable로 ref 역할 (메모리 절약)
            print("[GDPO] PEFT model detected → ref_model=None (use disable_adapter path)")
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = video_SALMONN2_plus.from_pretrained(
                model.config._name_or_path,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
            )
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
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
            top_p=1.0,
            top_k=50,                 # [hyj] base config top_k=1 override (다양성 확보)
            repetition_penalty=1.0,   # [hyj] time token 반복 페널티 제거
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )


        # ── DAPO clip-higher (ε_high > ε_low) ──────────────────────────────
        # clipped surrogate:  loss = -min( r·A,  clip(r, 1-ε_low, 1+ε_high)·A )
        #   r = π_θ(o)/π_old(o) = exp(logp - old_logp).  ε_high 를 키워(upper clip 완화)
        #   저확률 토큰의 확률 상승 여지를 남겨 탐색을 유지 = clip-higher.
        # DAPO 권장값 ε_low=0.2 / ε_high=0.28. config(clip.*)에서 args 로 주입됨.
        self.epsilon_low = getattr(args, "epsilon", 0.2)
        self.epsilon_high = getattr(args, "epsilon_high", None) or 0.28
        # clip-higher 경로 사용 여부 (이 trainer 의 기본 동작).
        self.clip_higher = getattr(args, "clip_higher", True)
        self.use_grpo = getattr(args, "use_grpo", True)

        # ── μ (num_iterations): 한 rollout 을 μ 번 정책 업데이트에 재사용 ──────────
        # μ>1 이라야 old(생성시점 snapshot) 와 현재 정책이 갈라져 r=exp(new-old)≠1 →
        # clip-higher 가 실제 binding 됨. μ=1 이면 매 step 재생성(=순수 GDPO 동작).
        self.num_iterations = max(1, int(getattr(args, "num_iterations", 1)))

        # ── [clip2] grad_accum 지원 (effective batch ≥ N×num_gpu) ────────────
        # block-repeat sampler 로 microbatch 순서를 'i,j,i,j' 로 만들고, HF 가 N(grad_accum)
        # microbatch 마다 optimizer.step() → μ iteration 사이에 policy 가 업데이트되어 clip
        # 의 r≠1 이 유지된다. 캐시는 **prompt 시그니처 키**로 관리(슬롯/카운터 X) → DDP 가
        # microbatch 순서를 흩어도 잘못된 재사용/크래시 없음(시그니처 불일치=cache miss→재생성).
        #   _rollout_cache: sig -> {"R": rollout, "uses": int}. μ 도달 엔트리는 즉시 evict(embed 해제).
        self.grad_accum = max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
        self._rollout_cache = {}

        # ── [sep2] multi-segment(GT>=2) prompt 의 advantage 가중 ──────────────
        # GDPO 는 prompt 그룹 내부에서 advantage 를 정규화하므로(한 microbatch=한 prompt) reward
        # 절대값으로는 "multi 가 single 보다 중요"가 전달되지 않는다(정규화로 제거). 대신
        # multi-GT step 의 (정규화된) advantage 를 w 배 해서 gradient mass 를 키워, 데이터
        # prior(single 편향, 4:6)를 상쇄한다. w=1.0 이면 sep1 과 동일 동작.
        # ⚠️ clip-higher on/off 무관: advantage 는 _generate_rollout 에서 산출되어 두 loss
        #    경로(clip_higher / 순수 GDPO)가 공통으로 R["advantages"] 를 쓰므로 자동 호환.
        self.multi_seg_weight = float(getattr(args, "multi_seg_weight", 1.0))  # 1.0=off(통합 기본)


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

    def _get_train_sampler(self, *args, **kwargs):
        # [clip2] block-repeat 순서로 공급.
        #   ⚠️ accelerate 가 배치를 rank 들에 **round-robin** 으로 분배한다(BatchSamplerShard,
        #      split_batches=False). 따라서 글로벌 block_size 를 grad_accum 으로만 잡으면
        #      분배 후 각 rank 가 [a,a](인접반복)를 받아 → grad_accum 이 a_fresh·a_cached 를
        #      같은 optimizer step 에 묶음 → policy 업데이트 없이 r≈1 → clip 무력 + effective
        #      batch 도 grad_accum 만큼 안 늘어남(같은 prompt 중복).
        #   ✅ block_size = grad_accum × num_processes 로 잡으면, round-robin 분배 후 각 rank 가
        #      정확히 [p0..p_{N-1}]×μ (N=grad_accum, **서로 다른** prompt) 를 받는다:
        #        global [a,b,c,d]×μ  --round-robin(G=2)-->  rank0=[a,c]×μ, rank1=[b,d]×μ
        #      → μ 사이 policy 업데이트(clip r≠1) + effective batch = grad_accum × num_processes.
        #      (1 GPU 면 ×1 → block_size=grad_accum, 그대로 정상.)
        base = super()._get_train_sampler(*args, **kwargs)
        if base is not None and (self.num_iterations > 1 or self.grad_accum > 1):
            world = max(1, int(getattr(self.accelerator, "num_processes", 1)))
            return _BlockRepeatSampler(base, block_size=self.grad_accum * world,
                                       num_repeat=self.num_iterations)
        return base

    def _prompt_sig(self, inputs):
        """prompt 식별용 경량 시그니처(충돌 사실상 0). 캐시 키로 사용 → 순서 무관 정확."""
        t = inputs["input_ids"][0]
        return (int(t.shape[0]), int(t.sum().item()),
                int(t[::7].sum().item()), int(t.float().square().sum().item()))

    # ============================================================
    # Per-token log probabilities (VS2+ API)
    # ============================================================

    def _get_per_token_logps(
        self, model, input_ids, attention_mask,
        pixel_values_videos=None, video_grid_thw=None,
        audio_feature=None, audio_lengths=None,
        position_ids=None, second_per_grid_ts=None,
        inputs_embeds=None, return_entropy=False,
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
        entropies = [] if return_entropy else None
        for logits_row, input_ids_row in zip(logits, input_ids):
            # 메모리 절약: full log_softmax([seq,vocab]) 상주 복사 없이 logp = logit − logsumexp
            token_logit = torch.gather(
                logits_row, dim=1, index=input_ids_row.unsqueeze(1)
            ).squeeze(1)
            per_token_logps.append(token_logit - torch.logsumexp(logits_row, dim=-1))
            if return_entropy:
                # 정책 분포 엔트로피 H=-Σ p·logp (position별, vocab 합). 로깅용이라 detach.
                log_probs = logits_row.log_softmax(dim=-1)
                entropies.append((-(log_probs.exp() * log_probs).sum(dim=-1)).detach())
                del log_probs

        logps = torch.stack(per_token_logps)
        if return_entropy:
            return logps, torch.stack(entropies)
        return logps




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

    def _completion_logps(self, model, R, raw_encode_model, return_entropy=False):
        """캐시된 completion 에 대해 per-token logps 계산.
        grad 흐름 여부(no_grad/grad)는 호출하는 쪽 컨텍스트가 결정한다.
        return_entropy=True 면 (logps, entropy[num_gen, comp_len]) 반환 (로깅용, detach)."""
        _VIDEO_TOKEN_ID = 151656
        comp_len = R["comp_len"]
        all_logps = []
        all_ent = [] if return_entropy else None
        for g in range(self.num_generations):
            pc_ids = torch.cat([R["prompt_ids_single"], R["completion_ids_clean"][g:g + 1]], dim=1)
            pc_mask = torch.cat([R["prompt_mask_single"], R["completion_mask"][g:g + 1]], dim=1)
            # text 위치는 trainable embed_tokens(grad), video/audio 위치는 캐시된 frozen embed.
            text_embeds = raw_encode_model.model.embed_tokens(pc_ids)
            if R["cached_video_embeds"] is not None:
                mask = (pc_ids == _VIDEO_TOKEN_ID)
                mask_exp = mask.unsqueeze(-1).expand_as(text_embeds)
                v = R["cached_video_embeds"].to(text_embeds.device, text_embeds.dtype)
                text_embeds = text_embeds.masked_scatter(mask_exp, v)
            if R["cached_audio_embeds"] is not None and R["audio_token_id_cfg"] is not None:
                mask = (pc_ids == R["audio_token_id_cfg"])
                mask_exp = mask.unsqueeze(-1).expand_as(text_embeds)
                a = R["cached_audio_embeds"].to(text_embeds.device, text_embeds.dtype)
                text_embeds = text_embeds.masked_scatter(mask_exp, a)
            _out = self._get_per_token_logps(
                model, pc_ids, pc_mask,
                video_grid_thw=R["video_grid_thw"],
                audio_lengths=R["audio_lengths"],
                second_per_grid_ts=R["second_per_grid_ts"],
                inputs_embeds=text_embeds,
                return_entropy=return_entropy,
            )
            if return_entropy:
                g_logps, g_ent = _out
                g_ent = g_ent[:, -comp_len:] if comp_len > 0 else g_ent[:, :0]
                all_ent.append(g_ent)
            else:
                g_logps = _out
            g_logps = g_logps[:, -comp_len:] if comp_len > 0 else g_logps[:, :0]
            all_logps.append(g_logps)
        if return_entropy:
            return torch.cat(all_logps, dim=0), torch.cat(all_ent, dim=0)
        return torch.cat(all_logps, dim=0)

    def _generate_rollout(self, model, inputs):
        """한 rollout 생성 + 캐시. μ>1 이면 이 결과를 μ 번의 정책 업데이트에 재사용한다.
        포함: completion 샘플링, reward/advantage, old(behavior) logps 스냅샷, ref logps(KL용),
              frozen video/audio embed 캐시. (생성·로깅은 fresh rollout 때 1회만 수행)"""
        device = self.accelerator.device
        _unwrapped = self.accelerator.unwrap_model(model)

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

        # [TTI-DBG] ⑤ prompt 에 time-marker 가 보존됐는지 검증 (rank0, 첫 N step)
        if TTI_DEBUG and self.accelerator.is_main_process and getattr(self, "_tti_dbg_step0", None) is None:
            self._tti_dbg_step0 = self.state.global_step
        if (TTI_DEBUG and self.accelerator.is_main_process
                and self.state.global_step < self._tti_dbg_step0 + TTI_DEBUG_STEPS):
            _id_range = getattr(_unwrapped.config, "time_token_id_range", None)
            _n_mark = count_time_markers(prompt_ids[0], _id_range)
            _tti_dbg(f"⑤ step {self.state.global_step}: prompt_len={prompt_ids.size(1)}, "
                     f"model.config.time_token_id_range={_id_range}, "
                     f"#time_markers_in_prompt={_n_mark}")
            if _id_range is not None and _n_mark == 0:
                _tti_dbg("⚠️ WARNING: tti ON(range set) 인데 prompt 에 time-marker 0개 "
                         "(데이터 마커 미삽입 의심 — rope ON 분기와 불일치)")
            try:
                _vs_id = self.processing_class.convert_tokens_to_ids("<|vision_start|>")
                _row = prompt_ids[0].tolist()
                if _vs_id in _row:
                    _p = _row.index(_vs_id)
                    _dec = self.processing_class.decode(_row[_p:_p + 40], skip_special_tokens=False)
                    _tti_dbg(f"   prompt[vision_start:+40]= {_dec!r}")
            except Exception as _e:
                _tti_dbg(f"   vision_start 슬라이스 디코드 실패: {_e}")

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
        from reward_functions import decode_vtg_time
        gt_intervals = []
        raw_labels = inputs.get("labels", None)
        if raw_labels is not None:
            raw_labels = raw_labels.to(device)
            gt_token_ids = raw_labels[0][raw_labels[0] != -100]
            if len(gt_token_ids) > 0:
                gt_answer = self.processing_class.decode(gt_token_ids, skip_special_tokens=False)
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
        prompt_length = prompt_ids.size(1)
        # generate 동안 gradient checkpointing 비활성화 (모듈 직접 순회 — 래퍼 우회).
        for _m in model.modules():
            if hasattr(_m, "gradient_checkpointing"):
                _m.gradient_checkpointing = False

        gen_kwargs = {
            "input_ids": prompt_ids,
            "attention_mask": prompt_mask,
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": 1.0,
            "top_k": 50,
            "repetition_penalty": 1.0,
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
            raw_model = (
                unwrapped_model.get_base_model()
                if hasattr(unwrapped_model, "get_base_model")
                else unwrapped_model
            )
            if self.accelerator.is_main_process and self.state.global_step == 0:
                gc = raw_model.generation_config
                print(f"[GDPO DEBUG] gen_kwargs (sampling): do_sample={gen_kwargs.get('do_sample')}, "
                      f"temperature={gen_kwargs.get('temperature')}, top_p={gen_kwargs.get('top_p')}, "
                      f"top_k={gen_kwargs.get('top_k', '<not set>')}, "
                      f"repetition_penalty={gen_kwargs.get('repetition_penalty', '<not set>')}")
                print(f"[GDPO DEBUG] raw_model.generation_config: "
                      f"do_sample={getattr(gc, 'do_sample', None)}, "
                      f"temperature={getattr(gc, 'temperature', None)}, "
                      f"top_p={getattr(gc, 'top_p', None)}, top_k={getattr(gc, 'top_k', None)}, "
                      f"repetition_penalty={getattr(gc, 'repetition_penalty', None)}")
            for _ in range(self.num_generations):
                gen_ids = raw_model.generate(**gen_kwargs)
                gen_ids = gen_ids[:, prompt_length:]
                all_completion_ids.append(gen_ids)

        # generate 끝 → gradient checkpointing 재활성화 (use_reentrant=False)
        import functools as _functools
        from torch.utils.checkpoint import checkpoint as _torch_checkpoint
        _gc_func = _functools.partial(_torch_checkpoint, use_reentrant=False)
        for _m in model.modules():
            if hasattr(_m, "gradient_checkpointing"):
                _m.gradient_checkpointing = True
                _m._gradient_checkpointing_func = _gc_func

        # 패딩 후 결합
        import torch.nn.functional as F
        max_len = max(c.size(1) for c in all_completion_ids)
        pad_id = self.processing_class.pad_token_id or 0
        padded = [F.pad(c, (0, max_len - c.size(1)), value=pad_id) for c in all_completion_ids]
        completion_ids = torch.cat(padded, dim=0)
        prompt_ids = prompt_ids.repeat(self.num_generations, 1)
        prompt_mask = prompt_mask.repeat(self.num_generations, 1)

        # EOS 마스크
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        if completion_ids.size(1) > 0 and is_eos.any():
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        actual_lengths = completion_mask.sum(dim=1).tolist()
        if self.accelerator.is_main_process:
            print(f"[GDPO STEP] prompt_len={prompt_length}, actual_lengths={[int(l) for l in actual_lengths]}, comp_len={completion_ids.size(1)}")
        comp_len = completion_ids.size(1)

        # completion 의 비디오/오디오 placeholder 토큰 제거 (logprob 계산 시 feature 수 불일치 방지)
        _VIDEO_TOKEN_ID = 151656
        completion_ids_clean = completion_ids.clone()
        completion_ids_clean[completion_ids_clean == _VIDEO_TOKEN_ID] = pad_id
        if hasattr(_unwrapped.config, 'audio_token_id'):
            completion_ids_clean[completion_ids_clean == _unwrapped.config.audio_token_id] = pad_id

        prompt_ids_single = prompt_ids[:1]
        prompt_mask_single = prompt_mask[:1]

        # ── Encoding cache (frozen video/audio encoder → rollout 당 1회) ──
        raw_encode_model = _unwrapped.get_base_model() if hasattr(_unwrapped, "get_base_model") else _unwrapped
        cached_video_embeds = None
        cached_audio_embeds = None
        with torch.no_grad():
            if pixel_values_videos is not None:
                pv = pixel_values_videos.type(raw_encode_model.visual.dtype)
                cached_video_embeds = raw_encode_model.visual(pv, grid_thw=video_grid_thw)
            if audio_feature is not None:
                af = audio_feature.type(raw_encode_model.audio.dtype)
                cached_audio_embeds = raw_encode_model.audio(af).flatten(0, 1)
        audio_token_id_cfg = getattr(_unwrapped.config, "audio_token_id", None)

        # rollout 상태 캐시 (다음 μ-1 번 compute_loss 호출이 그대로 재사용)
        R = {
            "prompt_ids_single": prompt_ids_single,
            "prompt_mask_single": prompt_mask_single,
            "completion_ids_clean": completion_ids_clean,
            "completion_mask": completion_mask,
            "comp_len": comp_len,
            "prompt_length": prompt_length,
            "video_grid_thw": video_grid_thw,
            "audio_lengths": audio_lengths,
            "second_per_grid_ts": second_per_grid_ts,
            "cached_video_embeds": cached_video_embeds,
            "cached_audio_embeds": cached_audio_embeds,
            "audio_token_id_cfg": audio_token_id_cfg,
        }

        # ── old(behavior) policy logps 스냅샷 ──
        # μ=1 이면 매 step 재생성 → old=현재 정책 detach 로 충분(추가 forward 생략, 원래 비용 유지).
        # μ>1 이면 별도 no_grad 스냅샷을 떠서 이후 재사용 step 들의 ratio 기준(과거 정책)으로 고정.
        if self.num_iterations > 1:
            with torch.no_grad():
                R["old_per_token_logps"] = self._completion_logps(model, R, raw_encode_model)
        else:
            R["old_per_token_logps"] = None   # compute_loss 에서 per_token_logps.detach() 사용

        # ── reference logps (KL용, 정책과 무관하게 고정 → 캐시) ──
        R["ref_per_token_logps"] = None
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    R["ref_per_token_logps"] = self._completion_logps(self.ref_model, R, raw_encode_model)
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        R["ref_per_token_logps"] = self._completion_logps(model, R, raw_encode_model)

        # ── Decode completions ──
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)
        _TIME_TOKENS = {f"<t{i}>" for i in range(10)} | {"<tdot>"}
        _special_to_remove = set(self.processing_class.all_special_tokens) - _TIME_TOKENS
        for tok in _special_to_remove:
            completions = [c.replace(tok, "") for c in completions]
        completions = [re.sub(r"<\|im_start\|>\s*\w+\s*", "", c).strip() for c in completions]
        if self.accelerator.is_main_process:
            print(f"[GDPO SAMPLE] completion[0]: {completions[0]}")

        # ── Rewards + advantages ──
        rewards_per_func = torch.zeros(len(completions), len(self.reward_funcs), device=device)
        gt_intervals_repeated = [gt_intervals] * self.num_generations
        for i, reward_func in enumerate(self.reward_funcs):
            output = reward_func(completions=completions, gt_intervals=gt_intervals_repeated)
            rewards_per_func[:, i] = torch.tensor(output, dtype=torch.float32, device=device)
        rewards = rewards_per_func.sum(dim=1)
        advantages = self._compute_gdpo_advantages(rewards_per_func, rewards)
        # [sep2] multi-segment(GT>=2) prompt 면 advantage 를 w 배 → single prior 상쇄.
        #   한 step 의 모든 rollout 은 같은 GT(=같은 multi/single) 라 그룹 전체에 동일 스케일.
        #   (정규화 이후 곱이므로 within-group 상대 신호는 유지, gradient 크기만 w 배)
        _n_gt = len(gt_intervals)
        _is_multi = _n_gt >= 2
        if _is_multi and self.multi_seg_weight != 1.0:
            advantages = advantages * self.multi_seg_weight
        if self.accelerator.is_main_process:
            print(f"[GDPO sep2] n_gt={_n_gt} multi={_is_multi} "
                  f"adv_weight={self.multi_seg_weight if _is_multi else 1.0}")
        R["rewards_per_func"] = rewards_per_func
        R["rewards"] = rewards
        R["advantages"] = advantages

        # ── [hyj] 디버그: GT/예측 세그먼트 raw + 초 단위 출력 (rank 0) ──
        if self.accelerator.is_main_process:
            _SEG_CAPTURE_DEBUG = re.compile(
                r"[Ff]rom\s+((?:<t\d>){1,4}<tdot><t\d>)\s+to\s+((?:<t\d>){1,4}<tdot><t\d>)"
            )

            def _answer_tuples(text):
                m = re.search(r"<answer>(.*?)</answer>", text, re.S)
                ans = m.group(1) if m else text
                segs = []
                for s_str, e_str in _SEG_CAPTURE_DEBUG.findall(ans):
                    s = decode_vtg_time(s_str)
                    e = decode_vtg_time(e_str)
                    if s is not None and e is not None:
                        segs.append(f"({s:.1f}, {e:.1f})")
                return ", ".join(segs) if segs else None   # None = answer 파싱 실패

            gt_raw_str = "[none]"
            if raw_labels is not None:
                _gt_ids = raw_labels[0][raw_labels[0] != -100]
                if len(_gt_ids) > 0:
                    gt_answer_raw = self.processing_class.decode(_gt_ids, skip_special_tokens=False)
                    gt_raw_segs = [f"from {s} to {e}" for s, e in _SEG_CAPTURE_DEBUG.findall(gt_answer_raw)]
                    if gt_raw_segs:
                        gt_raw_str = ", ".join(gt_raw_segs)
            gt_sec_str = ", ".join(f"({s:.1f}, {e:.1f})" for s, e in gt_intervals) if gt_intervals else "[none]"
            print(f"[GDPO SAMPLES] GT raw: {gt_raw_str}")
            print(f"[GDPO SAMPLES] GT sec: {gt_sec_str}")
            # 채널 이름은 reward_funcs 의 __name__ (load_reward_module 가 부여) → 하드코딩 인덱스 X.
            # 모듈마다 채널 구성이 달라도(format/iou | format/len/global/local | +CoT) 그대로 출력.
            _chan_names = [getattr(f, "__name__", f"ch{j}") for j, f in enumerate(self.reward_funcs)]
            for gi, c in enumerate(completions):
                _vals = ", ".join(f"{nm}={rewards_per_func[gi, j].item():.3f}"
                                  for j, nm in enumerate(_chan_names))
                _ans = _answer_tuples(c)
                print(f"  [{gi}] {_vals}")
                print(f"       raw:    {c}")                      # 원본 예측 그대로
                if _ans is not None:
                    print(f"       answer: {_ans}")               # (s, e) 튜플
                else:
                    print(f"       answer: [parse fail]")          # 파싱 실패(원본은 위 raw 참고)

        return R

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GDPOTrainer does not support return_outputs=True")

        _unwrapped = self.accelerator.unwrap_model(model)

        # ── [clip2] prompt-시그니처 키 rollout 캐시 (순서 무관·DDP 안전) ──────────
        #   같은 prompt 가 μ 번 들어오면 첫 호출에만 생성→캐시, 이후 μ-1 번 재사용.
        #   캐시 키 = prompt 시그니처 → microbatch 순서가 흩어져도(블록반복/DDP) 잘못된
        #   재사용/크래시 없음(불일치=miss→재생성). μ 도달 엔트리는 즉시 evict(embed 해제).
        sig = self._prompt_sig(inputs)
        entry = self._rollout_cache.get(sig)
        if entry is None or entry["uses"] >= self.num_iterations:
            # fresh: 새 rollout 생성(이때 old_logps 스냅샷 고정). cache miss(μ_iter≥1) 도 여기서 self-heal.
            R = self._generate_rollout(model, inputs)
            entry = {"R": R, "uses": 1}
            self._rollout_cache[sig] = entry
            fresh = True
        else:
            entry["uses"] += 1          # reuse: old_logps 는 고정(R 그대로), policy 만 갱신됨 → r≠1
            R = entry["R"]
            fresh = False
        # μ 도달 → evict(embed 캐시 해제). 캐시 크기 ~grad_accum 로 bound.
        if entry["uses"] >= self.num_iterations:
            self._rollout_cache.pop(sig, None)
        # 비정상적으로 캐시가 커지면(순서 교란) 오래된 것부터 정리(메모리 가드)
        while len(self._rollout_cache) > self.grad_accum + 1:
            self._rollout_cache.pop(next(iter(self._rollout_cache)))
        if self.accelerator.is_main_process and self.num_iterations > 1:
            print(f"[GDPO ROLLOUT] step={self.state.global_step} "
                  f"uses={entry['uses']}/{self.num_iterations} ({'fresh' if fresh else 'cached'}) "
                  f"cache={len(self._rollout_cache)}")

        comp_len = R["comp_len"]
        completion_mask = R["completion_mask"]
        advantages = R["advantages"]

        # ── 새 정책 per-token logps (grad O) — 캐시된 completion 에 대해 재계산 ──
        #    return_entropy=True → 정책 분포 엔트로피도 함께(로깅용, detach).
        raw_encode_model = _unwrapped.get_base_model() if hasattr(_unwrapped, "get_base_model") else _unwrapped
        per_token_logps = self._completion_logps(
            model, R, raw_encode_model, return_entropy=False)  # entropy off (메모리 절약; μ=1 진단 불필요)

        # old(behavior) logps: μ>1 이면 생성시점 스냅샷, μ=1 이면 현재 정책 detach(=ratio 1).
        old_per_token_logps = R["old_per_token_logps"]
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()

        # ── KL (캐시된 ref vs 새 정책) ──
        per_token_kl = None
        if self.beta != 0.0 and R["ref_per_token_logps"] is not None:
            ref_per_token_logps = R["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )

        # ── Loss ──
        if self.clip_higher:
            # DAPO clipped surrogate: -min( r·A, clip(r, 1-ε_low, 1+ε_high)·A )
            coef_1 = torch.exp(per_token_logps - old_per_token_logps)
            coef_2 = torch.clamp(coef_1, 1.0 - self.epsilon_low, 1.0 + self.epsilon_high)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            if per_token_kl is not None:
                per_token_loss = per_token_loss + self.beta * per_token_kl
            comp_lengths = completion_mask.sum(dim=1).clamp(min=1)
            loss = ((per_token_loss * completion_mask).sum(dim=1) / comp_lengths).mean()
        elif self.use_grpo:
            per_token_loss = torch.exp(per_token_logps - old_per_token_logps) * advantages.unsqueeze(1)
            if per_token_kl is not None:
                per_token_loss = -(per_token_loss - self.beta * per_token_kl)
            else:
                per_token_loss = -per_token_loss
            comp_lengths = completion_mask.sum(dim=1).clamp(min=1)
            loss = ((per_token_loss * completion_mask).sum(dim=1) / comp_lengths).mean()
        else:
            coef_1 = torch.exp(per_token_logps - old_per_token_logps)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            if per_token_kl is not None:
                per_token_loss = per_token_loss + self.beta * per_token_kl
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # ── 로깅 ──
        rewards_per_func = R["rewards_per_func"]
        rewards = R["rewards"]
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        # [wandb] Advantage 통계 (group-norm + multi_seg_weight 반영 후 실제 사용값)
        with torch.no_grad():
            self._metrics["advantage_mean"].append(self.accelerator.gather_for_metrics(advantages).mean().item())
            self._metrics["advantage_std"].append(self.accelerator.gather_for_metrics(advantages).std().item())
            self._metrics["advantage_abs_mean"].append(self.accelerator.gather_for_metrics(advantages.abs()).mean().item())

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            fname = getattr(reward_func, "__name__", f"func_{i}")
            self._metrics[f"rewards/{fname}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        std_grouped = rewards.view(-1, self.num_generations).std(dim=1)
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped).mean().item())

        # clip-higher 진단: ratio 평균 + clip 비율 (μ>1 에서 실제 binding 되는지 확인용)
        with torch.no_grad():
            ratio = torch.exp(per_token_logps - old_per_token_logps)
            mask_f = completion_mask.float()
            denom = mask_f.sum().clamp(min=1)
            mean_ratio = (ratio * mask_f).sum() / denom
            clipped = ((ratio > 1.0 + self.epsilon_high) | (ratio < 1.0 - self.epsilon_low)).float()
            clip_frac = (clipped * mask_f).sum() / denom
            self._metrics["ratio"].append(self.accelerator.gather_for_metrics(mean_ratio).mean().item())
            self._metrics["clip_frac"].append(self.accelerator.gather_for_metrics(clip_frac).mean().item())

        if per_token_kl is not None:
            comp_lengths_kl = completion_mask.sum(dim=1).clamp(min=1)
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / comp_lengths_kl).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # [clip2] 여기서 반환하는 loss 는 'prompt 1개'에 대한 평균(토큰/그룹 평균)이며,
        # grad_accum 으로 나누지 않는다. HF Trainer.training_step 이
        # (model_accepts_loss_kwargs=False 이므로) loss/=gradient_accumulation_steps 로 한 번만
        # 스케일 → N microbatch 누적 시 N-prompt 평균이 됨. 이중 스케일 없음.
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

def load_model_and_tokenizer(model_path, model_base, tti_mode="off"):
    """VS2+ (Qwen2.5-VL 기반) 모델 로딩.

    tti_mode:
      - "off": base config의 time_token_id_range를 무시(None으로 덮어씀).
               rope_index가 OFF 분기를 타도록 강제. 데이터에 time marker가
               섞여있지 않을 때 사용.
      - "on" : base config 그대로 사용. 데이터에 time marker가 포함된 경우.
    """
    print(f"[GDPO] Loading VS2+ model (tti_mode={tti_mode})")
    print(f"[GDPO]   model_path (SFT ckpt): {model_path}")
    print(f"[GDPO]   model_base: {model_base}")

    # 토크나이저 로드
    tok_path = model_path if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "tokenizer.json")) else model_base
    print(f"[GDPO] Loading tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, model_max_length=10000, padding_side="left")

    # 모델 로드
    print(f"[GDPO] Loading base model from: {model_base}")
    model = video_SALMONN2_plus.from_pretrained(
        model_base,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )

    # TTI mode 적용: OFF 모드일 때 base config의 time_token_id_range를 비활성화.
    # base "video_salmonn2_plus_7B_time_tokens"는 time_token_id_range가 박혀있어
    # 이대로 두면 rope_index가 항상 TTI-ON 분기를 타서, 데이터에 time marker가
    # 없는 경우 IndexError가 발생함.
    if tti_mode == "off":
        if getattr(model.config, "time_token_id_range", None) is not None:
            print(f"[GDPO] tti_mode=off → clearing model.config.time_token_id_range "
                  f"(was {model.config.time_token_id_range})")
            model.config.time_token_id_range = None
        if getattr(model.config, "time_marker_token_len", None):
            model.config.time_marker_token_len = None
    else:
        # tti_mode == "on": 경로 B(머지 모델)는 config 에 time_token_id_range 가 없을 수 있음
        # (SFT 머지 과정에서 유실). 데이터(special_token 마커)와 일치시키려면 토크나이저에서
        # 복원해 모델 config 에 세팅 → generation 시 모델 내부 rope 도 ON(fixed-position) 분기를
        # 타게 함. (경로 A 의 time-token base 는 이미 들어있어 이 분기 무영향.)
        if getattr(model.config, "time_token_id_range", None) is None:
            t0 = tokenizer.convert_tokens_to_ids("<t0>")
            tdot = tokenizer.convert_tokens_to_ids("<tdot>")
            if isinstance(t0, int) and isinstance(tdot, int) and t0 >= 0 and tdot >= 0:
                model.config.time_token_id_range = (min(t0, tdot), max(t0, tdot))
                model.config.time_marker_token_len = 5
                print(f"[GDPO] tti_mode=on + config 에 time_token_id_range 없음(경로B 머지모델) "
                      f"→ 토크나이저에서 복원: {model.config.time_token_id_range}, marker_len=5")
            else:
                print("[GDPO] ⚠️ tti_mode=on 인데 <t0>/<tdot> 토큰 ID 확인 실패 "
                      "→ time_token_id_range 미설정 (desync 유지)")
    # [TTI-DBG] ① 최종 model.config 상태 확인 (ON 모드는 기존에 미출력이었음)
    _tti_dbg(f"① load_model: tti_mode={tti_mode} | "
             f"model.config.time_token_id_range={getattr(model.config, 'time_token_id_range', None)} | "
             f"time_marker_token_len={getattr(model.config, 'time_marker_token_len', None)}")
    # resize_token_embeddings 호출 금지 — base는 vocab_size=152064 (Qwen 패딩 포함)로
    # 유지해야 SFT adapter의 modules_to_save(embed_tokens/lm_head, 152064 rows)와 맞음.
    model = model.to(torch.bfloat16)

    # LoRA 로드 — audio.layers 분리 필요 (SFT train_qwen.py와 동일 패턴)
    # adapter_config의 base_model_name_or_path는 무시됨 (이미 로드된 model 객체를 넘기므로).
    # modules_to_save=[model.embed_tokens, lm_head]는 PeftModel 로딩 시 자동 복원되므로
    # time token 임베딩 수동 복원 불필요.
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    adapter_loaded = os.path.isdir(model_path) and os.path.exists(adapter_config_path)
    if adapter_loaded:
        # ── (경로 A) adapter 이어학습 : SFT LoRA adapter 를 base 위에 로드 ──
        print(f"[GDPO] Loading LoRA adapter: {model_path}")
        audio_layers = model.audio.layers
        del model.audio.layers
        model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
        model.model.audio.layers = audio_layers

        # ── RL 학습대상 제한 (adapter 이어학습 경로 전용) ──────────────────
        # puvalor 계열 어댑터는 LoRA(q,k,v,o) + visual.merger + audio.qformer/q_tokens/
        # audio_proj + embed_tokens + lm_head 까지 전부 trainable 로 로드된다. 하지만
        # RL(GRPO)은 단일 LR(5e-6) 이라 SFT 의 per-module LR(merger 1e-5, audio 2e-5 등)을
        # 재현할 수 없고, noisy reward gradient 로 aligner 까지 흔들면 SFT 가 만든 멀티모달
        # grounding 이 손상될 위험이 크다. → policy 중심으로 학습대상을 제한:
        #   학습: LoRA(q,k,v) + embed_tokens + lm_head
        #   freeze: o_proj LoRA, visual.merger, audio.qformer/q_tokens/audio_proj
        _KEEP_LORA = ("q_proj", "k_proj", "v_proj")   # o_proj 제외
        _KEEP_SAVE = ("embed_tokens", "lm_head")      # merger / audio.* 제외
        _kept_names, _frozen_names = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            keep = any(m in n for m in _KEEP_LORA) if "lora_" in n \
                else any(m in n for m in _KEEP_SAVE)
            if keep:
                _kept_names.append(n)
            else:
                p.requires_grad_(False)
                _frozen_names.append(n)

        def _mod_summary(names):
            tags = ("q_proj", "k_proj", "v_proj", "o_proj", "visual.merger",
                    "audio.qformer", "audio.q_tokens", "audio.audio_proj",
                    "embed_tokens", "lm_head")
            return sorted({m for n in names for m in tags if m in n})

        _n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[GDPO] RL trainable 제한 → 학습 모듈: {_mod_summary(_kept_names)}")
        print(f"[GDPO]                      freeze 모듈: {_mod_summary(_frozen_names)}")
        print(f"[GDPO] trainable params: {_n_train/1e6:.1f}M")
        # ─────────────────────────────────────────────────────────────────
    else:
        # ── (경로 B) fresh LoRA : 머지 모델을 그대로 두고 main() 의 config.lora 로
        #    새 LoRA 를 씌운다. get_peft_model 이 비-LoRA 파라미터를 자동 freeze 하므로
        #    여기서 별도 trainable 제한 불필요(embed/lm_head 도 자동 freeze). ──
        print("[GDPO] No LoRA adapter found → fresh RL LoRA 경로 "
              "(main 의 config.lora 로 새 LoRA 적용 예정)")

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

def make_reward_functions(specs):
    """load_reward_module 의 spec 리스트 [(name, fn, needs_gt), ...] → trainer 용 콜러블 리스트.
    needs_gt=False(format) 는 completion 만, True(iou/timestamp/modality) 는 (completion, gt) 로 호출.
    채널 수 가변 — 비-CoT(2개)·CoT(4개) 동일 경로."""
    out = []
    for name, fn, needs_gt in specs:
        if needs_gt:
            def _r(completions, gt_intervals=None, _fn=fn, **kwargs):
                gt = gt_intervals or [[] for _ in completions]
                return [_fn(c, g) for c, g in zip(completions, gt)]
        else:
            def _r(completions, _fn=fn, **kwargs):
                return [_fn(c) for c in completions]
        _r.__name__ = name
        out.append(_r)
    return out


# ============================================================
# [VAL] Validation-set 평가 (in-trainer, reload 없음) — toggle 가능
#
# checkpoint selection 용. val.enabled(또는 --val_path) 일 때만 동작; 없으면 기존 동작 동일.
# save_steps 마다(on_save) 현재 메모리 정책으로 val 프롬프트를 greedy 생성 →
#   segment mIoU(pairwise) + pairwise F1_avg(θ=0.1/0.3/0.5/0.7/0.9) → 둘 평균(combined).
# 지표 정의는 eval/eval_miou.py 의 pairwise 블록과 1:1 동일하게 복제.
# val_metrics.jsonl 에 step별 append + wandb 로깅. 종료 후 select_best_ckpt.py 로 best 선택.
# ============================================================
VAL_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
# 시간토큰 "From <t..> to <t..>" 파싱 (eval_miou.py 와 동일 정규식/디코드)
_VAL_TOKTIME = r"(?:<t\d>)+(?:<tdot>(?:<t\d>)+)?"
_VAL_TOK_SEG = re.compile(rf"({_VAL_TOKTIME})\s*(?:to|-|–|—|~)\s*({_VAL_TOKTIME})", re.IGNORECASE)


def _val_fix(s, e, max_time):
    if e <= s:
        e = min(s + 0.1, max_time)
    return [min(s, max_time), min(e, max_time)]


def _val_decode_tok(token_str, max_time):
    if "<tdot>" in token_str:
        a, _, b = token_str.partition("<tdot>")
        ip = re.findall(r"<t(\d)>", a)
        dp = re.findall(r"<t(\d)>", b)
    else:
        ip = re.findall(r"<t(\d)>", token_str)
        dp = []
    if not ip:
        return None
    return min(int("".join(ip)) + (int(dp[0]) / 10.0 if dp else 0.0), max_time)


def _val_parse_tokens(text, max_time):
    out = []
    for sa, sb in _VAL_TOK_SEG.findall(text or ""):
        s, e = _val_decode_tok(sa, max_time), _val_decode_tok(sb, max_time)
        if s is None or e is None:
            continue
        out.append(_val_fix(s, e, max_time))
    return out


def _val_extract_answer_scope(text):
    """CoT: <answer>...</answer> 안만 (없으면 전체)."""
    spans = re.findall(r"<answer>(.*?)</answer>", text or "", re.DOTALL | re.IGNORECASE)
    return " ".join(spans) if spans else (text or "")


def _val_tiou(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _val_best_iou(seg, others):
    return max((_val_tiou(seg, o) for o in others), default=0.0)


def _val_merge(segs):
    """겹치는 구간들을 disjoint union 으로 병합."""
    if not segs:
        return []
    s = sorted([[float(a), float(b)] for a, b in segs], key=lambda x: x[0])
    out = [s[0][:]]
    for a, b in s[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _val_total(segs):
    return sum(e - s for s, e in segs)


def _val_sample_iou(gt, pred):
    """샘플당 1개 IoU (R-AVST all_iou): 전체 GT merge vs 전체 pred merge inter/union."""
    g, p = _val_merge(gt), _val_merge(pred)
    if not g and not p:
        return 0.0
    inter = 0.0
    for gs, ge in g:
        for ps, pe in p:
            inter += max(0.0, min(ge, pe) - max(gs, ps))
    union = _val_total(g) + _val_total(p) - inter
    return inter / union if union > 0 else 0.0


def _val_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _val_frac_ge(arr, th):
    return _val_mean([1.0 if x >= th else 0.0 for x in arr]) if arr else 0.0


def compute_val_metrics(records):
    """records: [{"gt": [[s,e]..], "pred": [[s,e]..]}]  → eval_miou.py 정의와 동일.

    test 와 지표를 맞추기 위해 IoU 는 sample 법, F1 은 pairwise 법을 쓴다.
      sample_miou  = 100 * mean_over_samples( all_iou(merge(gt), merge(pred)) )   ← eval_miou sample 블록
      pairwise_miou= 100 * mean_over_all_GT_segs( best_iou(g, pred) )             ← (참고용)
      f1_avg       = mean_over_θ( 2PR/(P+R) ),  R/P = pairwise best_iou≥θ frac    ← eval_miou pairwise 블록
      combined     = (sample_miou + f1_avg) / 2          ← checkpoint selection 기준

    호환: select_best_ckpt.py 가 읽는 키(seg_miou/f1_avg/combined)는 유지하되,
          seg_miou 는 이제 **sample IoU** 값을 담는다(sample_miou 와 동일). pairwise 값은 pairwise_miou 로 별도 보존.
    """
    gt_iou_all, pred_iou_all, sample_ious = [], [], []
    for r in records:
        gt, pred = r["gt"], r["pred"]
        if not gt:
            continue
        gt_iou_all.extend(_val_best_iou(g, pred) for g in gt)
        pred_iou_all.extend(_val_best_iou(p, gt) for p in pred)
        sample_ious.append(_val_sample_iou(gt, pred))
    sample_miou = 100.0 * _val_mean(sample_ious)
    pairwise_miou = 100.0 * _val_mean(gt_iou_all)
    f1, rec, prc = {}, {}, {}
    for t in VAL_THRESHOLDS:
        R = _val_frac_ge(gt_iou_all, t)
        P = _val_frac_ge(pred_iou_all, t)
        rec[str(t)] = round(100.0 * R, 4)
        prc[str(t)] = round(100.0 * P, 4)
        f1[str(t)] = round(100.0 * (2 * P * R / (P + R) if (P + R) > 0 else 0.0), 4)
    f1_avg = _val_mean(list(f1.values()))
    n_parse_ok = sum(1 for r in records if r["pred"])
    return {
        "seg_miou": round(sample_miou, 4),      # ← sample IoU (select_best 호환 키)
        "sample_miou": round(sample_miou, 4),
        "pairwise_miou": round(pairwise_miou, 4),
        "f1_avg": round(f1_avg, 4),
        "combined": round((sample_miou + f1_avg) / 2.0, 4),
        "R": rec, "P": prc, "F1": f1,
        "n_samples": len(records),
        "n_gt_segs": len(gt_iou_all),
        "n_pred_segs": len(pred_iou_all),
        "n_parse_ok": n_parse_ok,
    }


@torch.no_grad()
def _val_greedy_generate(trainer, inputs, max_new_tokens):
    """train 과 동일 멀티모달 입력으로 1샘플 greedy 생성 → completion 텍스트(특수토큰 유지)."""
    model = trainer.model
    acc = trainer.accelerator
    device = acc.device
    tok = trainer.processing_class

    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    labels = inputs.get("labels", None)
    if labels is not None:
        labels = labels.to(device)
        answer_start = (labels[0] != -100).nonzero(as_tuple=True)[0]
        if len(answer_start) > 0:
            pe = answer_start[0].item()
            prompt_ids = prompt_ids[:, :pe]
            prompt_mask = prompt_mask[:, :pe]

    pvv = inputs.get("pixel_values_videos", None)
    if pvv is not None:
        pvv = pvv.to(device=device, dtype=torch.bfloat16)
    vgt = inputs.get("video_grid_thw", None)
    if vgt is not None:
        vgt = vgt.to(device)
    af = inputs.get("audio_feature", None)
    if af is not None:
        af = af.to(device=device, dtype=torch.bfloat16)
    al = inputs.get("audio_lengths", None)
    # [VAL-FIX] 비디오 시간축 rope 스케일. 누락 시 time-token grounding 이 어긋나
    #   eval 추론경로(train_qwen.py: **inputs 로 전달) 대비 mIoU 가 ~절반으로 깎인다.
    spg = inputs.get("second_per_grid_ts", None)
    if spg is not None and hasattr(spg, "to"):
        spg = spg.to(device)

    prompt_length = prompt_ids.size(1)
    # generate 동안 gradient checkpointing 비활성화 (rollout 과 동일 패턴)
    for _m in model.modules():
        if hasattr(_m, "gradient_checkpointing"):
            _m.gradient_checkpointing = False

    gen_kwargs = {
        "input_ids": prompt_ids,
        "attention_mask": prompt_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,        # greedy (deterministic) — selection 재현성
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }
    if pvv is not None:
        gen_kwargs["pixel_values_videos"] = pvv
    if vgt is not None:
        gen_kwargs["video_grid_thw"] = vgt
    if af is not None:
        gen_kwargs["audio_feature"] = af
    if al is not None:
        gen_kwargs["audio_lengths"] = al
    if spg is not None:
        gen_kwargs["second_per_grid_ts"] = spg

    import warnings as _warnings
    with unwrap_model_for_generation(model, acc) as unwrapped_model:
        raw_model = (
            unwrapped_model.get_base_model()
            if hasattr(unwrapped_model, "get_base_model")
            else unwrapped_model
        )
        # greedy(do_sample=False)인데 model.generation_config 에 temperature/top_p/top_k 가
        # 남아 매 샘플 UserWarning 이 뜬다(무해하지만 로그 스팸). 그 경고만 억제.
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", message=r"`do_sample` is set to `False`")
            gen_ids = raw_model.generate(**gen_kwargs)
    gen_ids = gen_ids[:, prompt_length:]

    # gradient checkpointing 복구 (use_reentrant=False)
    import functools as _ft
    from torch.utils.checkpoint import checkpoint as _ckpt
    _gcf = _ft.partial(_ckpt, use_reentrant=False)
    for _m in model.modules():
        if hasattr(_m, "gradient_checkpointing"):
            _m.gradient_checkpointing = True
            _m._gradient_checkpointing_func = _gcf

    return tok.batch_decode(gen_ids, skip_special_tokens=False)[0]


class ValEvalCallback(transformers.TrainerCallback):
    """save_steps 마다 여러 val 셋으로 greedy 평가 → val_metrics.jsonl + wandb.

    각 세트를 개별로 찍고(val_<name>/*), 세트별 지표의 '평균'을 top-level
    combined/seg_miou/f1_avg 로 저장한다 → select_best_ckpt.py 가 평균 기준으로
    best step 을 고른다(세트 1개면 기존과 동일 동작). 학습은 종료시키지 않음.
    """

    def __init__(self, trainer, val_sets, out_dir,
                 max_new_tokens=256, max_time=999.9, natural=False):
        # val_sets: list of (name, LazySupervisedDataset)
        self.trainer = trainer
        self.out_dir = out_dir
        self.max_new_tokens = int(max_new_tokens)
        self.max_time = float(max_time)
        self.natural = bool(natural)
        self.jsonl_path = os.path.join(out_dir, "val_metrics.jsonl")
        # 각 세트의 GT — LazySupervisedDataset 가 list_data_dict 를 셔플하므로(dataset.py),
        # 셔플된 순서 그대로 gt_segments 를 읽어 예측↔GT 정렬을 보장.
        self.val_sets = []
        for name, ds in val_sets:
            gt_list = []
            for x in ds.list_data_dict:
                segs = x.get("gt_segments") or []
                gt_list.append([[float(s), float(e)] for s, e in segs])
            assert len(gt_list) == len(ds), (
                f"val[{name}] GT({len(gt_list)}) != dataset({len(ds)}) — 순서/내용 불일치")
            self.val_sets.append({"name": name, "ds": ds, "gt": gt_list})

    def on_save(self, args, state, control, **kwargs):
        self._run(state)

    def _eval_one(self, ds, gt_list, acc, world, rank):
        """한 val 세트를 샤딩 greedy 평가 → (main process) metric dict / (그 외) None."""
        from accelerate.utils import gather_object
        trainer = self.trainer
        local = []
        for i in range(len(ds)):
            if i % world != rank:
                continue
            inst = ds[i]
            batch = trainer.data_collator([inst])
            text = _val_greedy_generate(trainer, batch, self.max_new_tokens)
            scope = _val_extract_answer_scope(text) if self.natural else text
            pred = _val_parse_tokens(scope, self.max_time)
            local.append({"idx": i, "gt": gt_list[i], "pred": pred})
        gathered = gather_object(local)   # collective — 모든 rank 가 호출해야 함
        if not acc.is_main_process:
            return None
        seen, recs = set(), []
        for r in gathered:
            if r["idx"] in seen:
                continue
            seen.add(r["idx"])
            recs.append(r)
        return compute_val_metrics(recs)

    def _run(self, state):
        trainer = self.trainer
        acc = trainer.accelerator
        world = max(1, int(getattr(acc, "num_processes", 1)))
        rank = int(getattr(acc, "process_index", 0))
        step = int(state.global_step)

        model = trainer.model
        was_training = model.training
        model.eval()
        per_set = {}   # name -> metric dict (main process only)
        try:
            for vs in self.val_sets:
                m = self._eval_one(vs["ds"], vs["gt"], acc, world, rank)
                if m is not None:
                    per_set[vs["name"]] = m
        finally:
            if was_training:
                model.train()

        if not acc.is_main_process:
            return

        names = [vs["name"] for vs in self.val_sets]

        def _avg(key):
            return sum(per_set[n][key] for n in names) / len(names)

        # top-level = 세트 평균 (select_best_ckpt.py 호환 키)
        rec = {
            "step": step,
            "combined": round(_avg("combined"), 4),
            "seg_miou": round(_avg("seg_miou"), 4),
            "f1_avg": round(_avg("f1_avg"), 4),
            "n_parse_ok": sum(per_set[n]["n_parse_ok"] for n in names),
            "n_samples": sum(per_set[n]["n_samples"] for n in names),
            "sets": per_set,
        }
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        parts = "  ||  ".join(
            f"{n}: comb={per_set[n]['combined']:.2f}"
            f"(mIoU={per_set[n]['sample_miou']:.2f},F1={per_set[n]['f1_avg']:.2f},"
            f"parse_ok={per_set[n]['n_parse_ok']})"
            for n in names)
        print(f"[VAL] step {step}: AVG combined={rec['combined']:.2f} "
              f"(seg_miou={rec['seg_miou']:.2f}, f1_avg={rec['f1_avg']:.2f})  ||  {parts}")

        log = {
            "val/combined": rec["combined"],
            "val/seg_miou": rec["seg_miou"],
            "val/f1_avg": rec["f1_avg"],
        }
        for n in names:
            log[f"val_{n}/combined"] = per_set[n]["combined"]
            log[f"val_{n}/seg_miou"] = per_set[n]["seg_miou"]
            log[f"val_{n}/f1_avg"] = per_set[n]["f1_avg"]
            log[f"val_{n}/n_pred_segs"] = per_set[n]["n_pred_segs"]
        try:
            trainer.log(log)
        except Exception as _e:
            print(f"[VAL] wandb log 스킵: {_e}")


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
    parser.add_argument("--max_steps", type=int, default=None,
                        help="smoke-test 등에서 step 수 제한. >0이면 num_train_epochs 무시.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="wandb/tracker run name 재정의. 없으면 config.logging.run_name 사용.")
    parser.add_argument("--tti_mode", type=str, default=None, choices=[None, "off", "on"],
                        help="TTI 모드. off=time_token_id_range 무시 / on=base config 그대로. "
                             "기본값은 config.model.tti_mode 또는 'off'.")
    parser.add_argument("--reward_module", type=str, default=None,
                        help="reward 함수 모듈명. reward_functions(기본, iou=MUSEG r_M) | "
                             "reward_functions_rM_fp(iou=r_M-0.2*FP). "
                             "없으면 config.reward.module 또는 reward_functions.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="'True'=output_dir 내 최신 체크포인트 자동 재개, 또는 체크포인트 경로. "
                             "미지정=처음부터.")
    parser.add_argument("--val_path", type=str, default=None,
                        help="[VAL] validation JSON. 지정 시 save_steps 마다 greedy 평가→val_metrics.jsonl. "
                             "없으면 config.val.dataset_path/val.enabled 따름(기본 OFF).")
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

    # ── CLI 인자 + 실행 환경 기록 (train.log 처럼 저장경로에 별도 보존) ──
    # rank-0 에서만 기록. output_dir 가 아직 없을 수 있으니 생성.
    # 무엇으로 돌린 런인지(주입한 인자 + 실제 커맨드라인) 사후 재현용.
    if _is_rank0():
        os.makedirs(output_dir, exist_ok=True)
        _args_record = {
            "cli_args": vars(cli),                       # 파싱된 argparse 인자
            "argv": sys.argv,                            # 실제 커맨드라인 (python ... --config ...)
            "resolved": {                                # config 와 병합된 최종값(일부)
                "model_path": model_path,
                "model_base": model_base,
                "dataset_path": dataset_path,
                "output_dir": output_dir,
            },
            "cwd": os.getcwd(),
        }
        # 기존 파일이 있으면(=resume 등 재실행) 덮어쓰지 않고 번호를 붙여 이력 보존.
        #   cli_args.json → cli_args.1.json → cli_args.2.json ...
        _args_path = os.path.join(output_dir, "cli_args.json")
        if os.path.exists(_args_path):
            _n = 1
            while os.path.exists(os.path.join(output_dir, f"cli_args.{_n}.json")):
                _n += 1
            _args_path = os.path.join(output_dir, f"cli_args.{_n}.json")
        with open(_args_path, "w", encoding="utf-8") as _f:
            json.dump(_args_record, _f, ensure_ascii=False, indent=2)
        print(f"[GDPO] CLI 인자 기록: {_args_path}")

    # GDPO 파라미터
    num_generations = _get(None, "gdpo", "num_generations", default=8)
    max_completion_length = _get(None, "gdpo", "max_completion_length", default=512)
    beta = _get(None, "gdpo", "beta", default=0.04)
    reward_weights = _get(None, "gdpo", "reward_weights", default=[1.0, 1.0, 1.0])
    # [hyj] temperature 를 config 에서 읽어 GRPOConfig 로 전달 (기존 cdh 는 누락 → 항상 기본값).
    temperature = float(_get(None, "gdpo", "temperature", default=1.0))

    # ── DAPO clip-higher (config: clip.*) ──────────────────────────────────
    #   enabled : clipped surrogate 경로 사용 (이 trainer 의 기본 동작).
    #   epsilon / epsilon_high : DAPO 권장값 0.2 / 0.28 (ε_high>ε_low = clip-higher).
    #   num_iterations(μ) : 한 rollout 을 μ 번 정책 업데이트에 재사용. μ>1 이라야 r≠1 →
    #       clip 이 실제 binding. μ=1 이면 r≈1 = 순수 GDPO 와 수치 동일.
    #       ⚠️ μ>1 은 grad_accum=1 필수(아래 경고 참고).
    clip_enabled = bool(_get(None, "clip", "enabled", default=True))
    epsilon_low = float(_get(None, "clip", "epsilon", default=0.2))
    epsilon_high = float(_get(None, "clip", "epsilon_high", default=0.28))
    num_iterations = int(_get(None, "clip", "num_iterations", default=1))

    # [sep2] multi-segment(GT>=2) prompt advantage 가중치 (config: gdpo.multi_seg_weight)
    #   통합 기본 1.0(=off, 기존 GDPO 동일). >1 이면 GT>=2 prompt advantage ×w → single prior(4:6) 상쇄.
    multi_seg_weight = float(_get(None, "gdpo", "multi_seg_weight", default=1.0))

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
    report_to = _get(None, "logging", "report_to", default="tensorboard")
    run_name = _get(cli.run_name, "logging", "run_name", default=os.path.basename(output_dir))

    # TTI 모드 결정 (cli > config.model.tti_mode > "off")
    tti_mode = _get(cli.tti_mode, "model", "tti_mode", default="off")
    # tti_mode → 데이터셋 입력 마커 포맷 매핑.
    #   on  : special_token 마커를 video/audio 청크 사이에 인터리빙 (rope ON 분기와 짝)
    #   off : 마커 미삽입 (rope OFF 분기와 짝)
    # 모델 config(time_token_id_range) 와 데이터셋(tti_time_format) 양쪽이 일치해야 함.
    tti_time_format = "special_token" if tti_mode == "on" else "off"

    # Model
    # 모델 로드/LoRA fresh init 보다 먼저 시딩 (Trainer 내부 set_seed 는 get_peft_model 이후라 늦음).
    transformers.set_seed(seed)
    model, tokenizer = load_model_and_tokenizer(model_path, model_base, tti_mode=tti_mode)

    # ── [hyj] PEFT — RL용 fresh LoRA (경로 B) ──────────────────────────────
    # model_path 가 SFT-머지 모델(adapter_config.json 없음)일 때, SFT 어댑터를 다시
    # 얹지 않고 머지 weight 위에 새 LoRA 를 만들어 RL 학습한다. ref 모델은 GDPOTrainer 가
    # LoRA disable 로 처리(=SFT 정책).  config.lora.enabled=true 일 때만 활성.
    #   ⚠️ adapter 이어학습(경로 A)을 쓸 거면 config 의 lora.enabled 를 false 로 둘 것
    #      (이미 PeftModel 인 model 에 또 LoRA 를 씌우면 이중 적용됨).
    peft_config = None
    if _get(None, "lora", "enabled", default=False):
        if isinstance(model, PeftModel):
            print("[GDPO] ⚠️ 이미 LoRA adapter 가 로드된 모델인데 config.lora.enabled=true "
                  "→ fresh LoRA 적용을 건너뜀 (이중 적용 방지). adapter 이어학습으로 진행.")
        else:
            lora_r = int(_get(None, "lora", "r", default=32))
            lora_alpha = int(_get(None, "lora", "lora_alpha", default=64))
            lora_dropout = float(_get(None, "lora", "lora_dropout", default=0.05))
            # LLM 블록 타깃 suffix — config(lora.target_modules)로 오버라이드 가능.
            #   기본: attention(q/k/v/o_proj)만 (audio/visual 제외, SFT와 동일 범위).
            #   예) MLP까지 열려면 config 에 target_modules:[q_proj,k_proj,v_proj,o_proj,
            #       gate_proj,up_proj,down_proj]. model.layers.* 한정은 그대로 유지.
            _tm_suffixes = tuple(_get(None, "lora", "target_modules",
                                      default=["q_proj", "k_proj", "v_proj", "o_proj"]))
            target_modules = [
                n for n, _ in model.named_modules()
                if n.startswith("model.layers.")
                and n.split(".")[-1] in _tm_suffixes
            ]
            # [sep2] modules_to_save 를 embed_tokens / lm_head 개별 토글로 분리.
            #   embed_tokens / lm_head 는 modules_to_save → full trainable copy.
            #   ref(=disable_adapter)는 original(=SFT 머지본) 을 쓰므로 KL 기준은 SFT 정책 유지.
            #   통합 기본: 둘 다 false → q/k/v/o LoRA 만. 예) lm_head 만 학습하려면
            #   train_embeddings:false + train_lm_head:true (config_*sep*.yaml 참고).
            train_emb = bool(_get(None, "lora", "train_embeddings", default=False))
            train_lm_head = bool(_get(None, "lora", "train_lm_head", default=False))
            modules_to_save = [m for m, on in (("embed_tokens", train_emb),
                                               ("lm_head", train_lm_head)) if on] or None
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                modules_to_save=modules_to_save,
                bias="none",
                task_type="CAUSAL_LM",
            )
            print(f"[GDPO] Fresh RL LoRA: r={lora_r}, alpha={lora_alpha}, "
                  f"#target_modules={len(target_modules)} (suffixes={list(_tm_suffixes)}), "
                  f"modules_to_save={modules_to_save}")

    # Dataset — VS2+ LazySupervisedDataset 사용
    print(f"[GDPO] Loading dataset from {dataset_path}")
    from dataclasses import dataclass
    from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast

    from transformers import WhisperFeatureExtractor

    @dataclass
    class GDPODataArgs:
        dataset_use: str = ""
        model_type: str = "qwen2.5vl"
        video_max_frames: int = 128         #  SFT 이상값 
        video_min_frames: int = 64          # SFT 이상값
        base_interval: float = 0.2          # SFT 이상값 (fps5). 이전 2(fps0.5)에서 변경
        max_pixels: int = 176400            # SFT와 동일 (28*28*225)
        min_pixels: int = 784               # SFT와 동일 (28*28)
        video_max_frame_pixels: int = 28224 # SFT 이상값 (이전 25088)
        video_min_frame_pixels: int = 784   # SFT 이상값 (이전 3136)
        video_max_total_pixels: int = 1664 * 28 * 28
        video_min_total_pixels: int = 256 * 28 * 28
        run_test: bool = False
        do_sample: bool = False
        num_sample: int = 1
        train_type: str = "sft"
        # 입력 측 TTI 마커 포맷. tti_mode 에서 파생 (on→special_token, off→off).
        # LazySupervisedDataset 가 이 값으로 video/audio 청크 사이에 마커를 인터리빙.
        # 이 값이 없으면 dataset 은 기본 "off" → special 모드인데도 마커 미삽입 → rope ON 분기와 불일치.
        tti_time_format: str = "special_token"
        feature_size: int = 128
        chunk_length: int = 30
        hop_length: int = 160
        sampling_rate: int = 16000
        image_processor: object = None
        audio_processor: object = None      # SFT에서 필요
        # [GDPO Tier1] 비디오 디코드 메모이즈 캐시 크기(0=OFF). μ>1 일 때만 켠다(아래 주입).
        video_cache_size: int = 0

    data_args = GDPODataArgs()
    data_args.dataset_use = dataset_path
    data_args.tti_time_format = tti_time_format
    # [GDPO Tier1] μ>1(clip-higher rollout 재사용)에서 block-repeat 가 같은 비디오를 μ 번
    #   요청하지만 cached 패스는 디코드 결과를 버린다(헛디코딩 → 호스트 RAM 압박). 최근
    #   grad_accum+1 개 비디오를 메모이즈해 재디코드를 생략 → batch(grad_accum) 키울 여유 확보.
    #   μ=1 이면 재사용이 없어 0(OFF)=기존 동작. (캐시 호스트 RAM 비용 ~수십~백수십MB, 바운드)
    data_args.video_cache_size = (int(grad_accum) + 1) if int(num_iterations) > 1 else 0
    print(f"[GDPO] tti_mode={tti_mode} → data_args.tti_time_format={tti_time_format}")
    print(f"[GDPO Tier1] video_cache_size={data_args.video_cache_size} "
          f"(μ={num_iterations}, grad_accum={grad_accum} → μ>1 일 때 헛디코딩 제거)")
    # [TTI-DBG] ② config(model.time_token_id_range) ↔ data(tti_time_format) 정합성.
    #   special_token ↔ id_range 있음, off ↔ id_range None 이어야 짝이 맞음.
    if TTI_DEBUG and _is_rank0():
        _mc_range = getattr(model.config, "time_token_id_range", None)
        _has_range = (_mc_range is not None)
        _want_markers = (tti_time_format == "special_token")
        _tti_dbg(f"② main: tti_mode={tti_mode}, tti_time_format={tti_time_format}, "
                 f"model.config.time_token_id_range={_mc_range}")
        if _has_range != _want_markers:
            _tti_dbg(f"⚠️ WARNING desync: tti_time_format={tti_time_format} 인데 "
                     f"model.config.time_token_id_range={_mc_range} "
                     f"(ON↔range, OFF↔None 이어야 함) — rope 분기/마커 불일치 가능")
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

    # ── [VAL] validation 데이터셋 (train 과 동일 data_args, 경로만 교체) ──
    #   토글: --val_path 주면 강제 ON / 없으면 config.val.enabled / 둘 다 없으면 OFF.
    #   (val 블록 없는 이전 config·이전 호출과 100% 호환 — 기본 OFF = 기존 동작 동일)
    val_path = _get(cli.val_path, "val", "dataset_path", default=None)
    val_enabled = (cli.val_path is not None) or bool(_get(None, "val", "enabled", default=False))
    val_sets = []   # list of (name, LazySupervisedDataset)

    def _make_val_ds(path):
        vda = GDPODataArgs()
        vda.dataset_use = path
        vda.tti_time_format = tti_time_format
        vda.image_processor = data_args.image_processor
        vda.audio_processor = data_args.audio_processor
        return LazySupervisedDataset(tokenizer=tokenizer, data_args=vda)

    if val_enabled and val_path:
        primary_name = str(_get(None, "val", "name", default="unav"))
        val_sets.append((primary_name, _make_val_ds(val_path)))
        print(f"[GDPO][VAL] val dataset[{primary_name}]: {val_path} (size={len(val_sets[-1][1])})")
        # 추가 val 세트: config.val.extra = [{name, dataset_path}, ...] (없으면 단일 세트)
        for ex in (_get(None, "val", "extra", default=None) or []):
            ex_name = str(ex.get("name", f"val{len(val_sets)}"))
            ex_path = ex.get("dataset_path")
            if not ex_path:
                print(f"[GDPO][VAL] ⚠️ extra val '{ex_name}' dataset_path 없음 → 스킵")
                continue
            val_sets.append((ex_name, _make_val_ds(ex_path)))
            print(f"[GDPO][VAL] val dataset[{ex_name}]: {ex_path} (size={len(val_sets[-1][1])})")
    elif val_enabled and not val_path:
        print("[GDPO][VAL] ⚠️ val.enabled 인데 val_path 없음 → val 평가 비활성")

    # Reward — 모듈 선택 (단일 trainer, import 대상만 교체)
    reward_module = _get(cli.reward_module, "reward", "module", default="reward_functions")
    _specs = load_reward_module(reward_module)
    reward_funcs = make_reward_functions(_specs)
    _chan_names = [n for n, _, _ in _specs]
    print(f"[GDPO] reward_module={reward_module}  채널={_chan_names} "
          f"(iou 구현체={getattr(_specs[1][1], '__name__', '?')})")
    # reward_weights 길이를 채널 수에 맞춤 (비-CoT 2채널은 그대로, CoT 4채널은 패딩/절단)
    if len(reward_weights) != len(reward_funcs):
        if len(reward_weights) > len(reward_funcs):
            reward_weights = list(reward_weights)[:len(reward_funcs)]
        else:
            reward_weights = list(reward_weights) + [1.0] * (len(reward_funcs) - len(reward_weights))
        print(f"[GDPO] ⚠️ reward_weights 길이 자동조정 → {reward_weights} (채널 {len(reward_funcs)}개). "
              f"CoT 4채널은 명시 권장: baseline [1,1,0,0] / guard [1,1,0.5,0.5]")
    print(f"[GDPO] Reward weights: {reward_weights}")

    # GRPOConfig
    grpo_kwargs = dict(
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
        temperature=temperature,
        bf16=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=report_to,
        run_name=run_name,
        logging_dir=os.path.join(output_dir, "logs"),
        remove_unused_columns=False,
    )
    # max_steps: cli > config.training.max_steps (>0 일 때만 적용; epoch 무시)
    _max_steps = _get(cli.max_steps, "training", "max_steps", default=None)
    if _max_steps is not None and int(_max_steps) > 0:
        grpo_kwargs["max_steps"] = int(_max_steps)
    grpo_args = GRPOConfig(**grpo_kwargs)
    # clip-higher 파라미터를 args 에 부착 (GDPOTrainer.__init__ 가 getattr 로 읽음).
    #   GRPOConfig 생성자에 없는 필드일 수 있어 생성 후 setattr 로 안전하게 주입.
    grpo_args.epsilon = epsilon_low
    grpo_args.epsilon_high = epsilon_high
    grpo_args.clip_higher = clip_enabled
    grpo_args.num_iterations = max(1, num_iterations)
    grpo_args.multi_seg_weight = multi_seg_weight   # [sep2] GDPOTrainer.__init__ 가 getattr 로 읽음
    _clip_state = (f"μ={grpo_args.num_iterations} → clip 작동"
                   if grpo_args.num_iterations > 1 else "μ=1 → clip 미작동(=GDPO 동일)")
    print(f"[GDPO] clip-higher: enabled={clip_enabled}, ε_low={epsilon_low}, ε_high={epsilon_high}, {_clip_state}")
    print(f"[GDPO sep2] multi_seg_weight(w)={multi_seg_weight} "
          f"({'multi-GT advantage ×w' if multi_seg_weight != 1.0 else 'w=1 → sep1 동일'})")
    # [clip2] grad_accum 지원 — block-repeat sampler 로 μ iteration 사이에 step 이 일어나므로
    #   grad_accum>1 이어도 clip 이 정상 binding (기존 'grad_accum=1 필수' 경고는 더 이상 해당 없음).
    _N = int(grad_accum)
    _world = int(os.environ.get("WORLD_SIZE", "1"))
    _eff_batch = _N * _world
    print(f"[GDPO clip2] grad_accum(N)={_N}, world_size={_world} → effective batch(distinct prompt/step)={_eff_batch}")
    print(f"[GDPO clip2] block-repeat sampler: 한 블록 N={_N} prompt 를 μ={grpo_args.num_iterations}번 반복 "
          f"→ N microbatch 마다 step (μ 사이 policy 업데이트). "
          f"고유 prompt 총량 = max_steps × N × world / μ")
    if _eff_batch < 4:
        print(f"[GDPO clip2] ⚠️ effective batch={_eff_batch} (<4). reward 추정/gradient noisy 할 수 있음 "
              f"→ grad_accum 또는 GPU 수 ↑ 권장.")

    # Trainer  ([hyj] peft_config 전달 → 경로 B 일 때 fresh LoRA 적용)
    trainer = GDPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_weights=reward_weights,
        peft_config=peft_config,
    )

    # ── [VAL] save_steps 마다 greedy 평가 콜백 부착 (checkpoint selection 용; val OFF면 스킵) ──
    if val_sets:
        val_max_new = int(_get(None, "val", "max_new_tokens", default=max_completion_length))
        val_max_time = float(_get(None, "val", "max_time", default=999.9))
        val_natural = bool(_get(None, "val", "natural", default=False))
        trainer.add_callback(ValEvalCallback(
            trainer=trainer,
            val_sets=val_sets,
            out_dir=output_dir,
            max_new_tokens=val_max_new,
            max_time=val_max_time,
            natural=val_natural,
        ))
        _names = ", ".join(n for n, _ in val_sets)
        print(f"[GDPO][VAL] ValEvalCallback 부착 — save_steps({save_steps})마다 평가 "
              f"[{_names}] (greedy, max_new={val_max_new}, max_time={val_max_time}) → "
              f"{output_dir}/val_metrics.jsonl  (top-level combined = 세트 평균)")

    # Train ([hyj] resume 지원: 'True'=최신 체크포인트 자동 / 경로=특정 체크포인트 / None=처음부터)
    _rc = cli.resume_from_checkpoint
    if _rc in ("", "None"):
        _rc = None
    elif _rc == "True":
        _rc = True
    print(f"[GDPO] Starting training... (resume_from_checkpoint={_rc})")
    trainer.train(resume_from_checkpoint=_rc)
    print(f"[GDPO] Saving model to {output_dir}")
    trainer.save_model(output_dir)
    print("[GDPO] Training complete!")


if __name__ == "__main__":
    main()
