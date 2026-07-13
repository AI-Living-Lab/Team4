# Copyright (2025) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adopted from https://github.com/QwenLM/Qwen2.5-VL. The original license is located at 'third-party-license/qwenvl.txt'.

import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    model_base: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_mm_audio: bool = field(default=False)
    tune_mm_qformer: bool = field(default=False)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_ckpt: str = field(default="No")
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj",
        metadata={"help": "Comma-separated linear layer names for LoRA target_modules."},
    )
    invert_time_row_mask: bool = field(
        default=False,
        metadata={"help": "embed_tokens/lm_head gradient row-mask 반전. "
                          "False(기본)=time-token row만 학습(special_token SFT). "
                          "True=time-token row만 freeze하고 나머지 전부 학습(chronus SFT — 평문토큰 학습)."},
    )
    tune_embed_lm_head: bool = field(
        default=False,
        metadata={"help": "embed_tokens/lm_head 전체 vocab row 학습(row-mask 없음). "
                          "타임토큰 없는 순수 백본(_full)에서 chronus SFT 할 때 사용. "
                          "time_token_ids 유무와 무관하게 embed/lm_head 를 modules_to_save 에 추가."},
    )

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    eval_dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    base_interval: float = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    run_test: bool = field(default=False)
    do_sample: bool = field(default=False)
    num_sample: int = field(default=1)
    train_type: str = field(default="sft")
    feature_size: int = field(default=128)
    chunk_length: int = field(default=30)
    hop_length: int = field(default=160)
    sampling_rate: int = field(default=16000)
    # Ordinal loss 관련 (time-token ordinal supervision)
    ordinal_enabled: bool = field(default=False)
    time_ndig_int: int = field(default=4)   # 정수부 자릿수 (e.g. 4 for 4+1, 3 for 3+2)
    time_ndig_dec: int = field(default=1)   # 소수부 자릿수 (e.g. 1 for 4+1, 2 for 3+2)
    # TTI (Time-Token Interleaving) input-side marker mode
    #   "off"           : 마커 미삽입 (Qwen2.5-VL 베이스라인) — 기본값
    #   "special_token" : <t0><t0><t1><tdot><t5> (5 special tokens / chunk, XXX.Y)
    #   "natural_text"  : "second{XXXX.Y}" (9 text tokens / chunk)
    #   "from_to"       : "From <t*>×5 to <t*>×5" (14 tokens; 출력 GT 와 동일 포맷)
    # 출력(GT/labels) 형식은 모드와 무관 (항상 special_token). 이 옵션은 video/audio
    # 청크 사이에 끼우는 입력 측 마커 표현만 바꾼다.
    tti_time_format: str = field(default="off")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    early_stopping_patience: int = field(default=0)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    pred_rank: int = field(default=0)
    no_audio: bool = field(default=False)
    # Per-module LR / weight_decay overrides (None = fall back to learning_rate / weight_decay)
    lora_lr: Optional[float] = None
    lora_wd: Optional[float] = None
    embed_lr: Optional[float] = None         # for model.embed_tokens (new token rows only)
    embed_wd: Optional[float] = None
    lm_head_lr: Optional[float] = None       # for lm_head (new token rows only)
    lm_head_wd: Optional[float] = None
    visual_merger_lr: Optional[float] = None
    visual_merger_wd: Optional[float] = None
    audio_qformer_lr: Optional[float] = None
    audio_qformer_wd: Optional[float] = None
    audio_proj_lr: Optional[float] = None
    audio_proj_wd: Optional[float] = None
    audio_q_tokens_lr: Optional[float] = None
    audio_q_tokens_wd: Optional[float] = None
    # Ordinal loss 관련
    lambda_ord: float = field(default=0.0)            # 0 이면 ordinal loss 사용 안 함 (CE only)
    ord_head_lr: Optional[float] = None
    ord_head_wd: Optional[float] = None
    ordinal_unav_weight: float = field(default=1.0)   # UnAV sample 의 ord_loss 가중치 (PU-VALOR=1.0 기준)
