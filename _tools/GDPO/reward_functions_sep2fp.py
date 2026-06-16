#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_sep2fp.py
  reward_functions_rM_sep2 (non-CoT 4채널) + fp 채널 = 5채널 non-CoT 버전.
  reward_functions_sep2_cotp5 의 *비-CoT* 대응물 (answer 블록 추출 없이 completion 전체 파싱).

  채널 5개 (전부 [0,1], completion 전체 기준):
    format — 'From <t\\d>{3}<tdot><t\\d> to ... .' strict 나열 (개수 중립)  [rM_sep2]
    len    — graded + multi-pred1 floor                                  [rM_sep2]
    global — MUSEG F1 (precision 내장)                                    [rM_sep2]
    local  — NGIoU best-match / max(n)                                    [rM_sep2]
    fp     — precision = 1 - n_unmatched_pred/n_pred  (rM_fp 로직 [0,1]판)

  데이터 = unav100_v2.json (non-CoT, "From X to Y." 직접 답), 프롬프트도 non-CoT.
  원본 reward_functions* 무수정, import 만.
"""
import os
from typing import List, Tuple

# rM_sep2 의 4채널 그대로 재사용 (전부 completion 전체를 _parse → non-CoT 그대로 맞음)
from reward_functions_rM_sep2 import (
    format_reward,
    len_matching_reward,
    global_reward,
    local_reward,
)
from reward_functions_rM_sep import _parse          # completion 전체 세그먼트 파싱
from reward_functions_rM_fp import n_unmatched_pred  # FP 개수 (completion 전체)


def fp_reward(completion: str, gt_intervals, **kwargs) -> float:
    """precision = 1 - n_unmatched_pred/n_pred ∈ [0,1] (FP 적을수록 1)."""
    if not gt_intervals:
        return 0.0
    preds = _parse(completion)
    if not preds:
        return 0.0
    return 1.0 - n_unmatched_pred(completion, gt_intervals) / len(preds)


iou_reward = global_reward   # 레거시 'iou' 슬롯
REWARD_CHANNELS = [
    ("format", format_reward,       False),
    ("len",    len_matching_reward, True),
    ("global", global_reward,       True),
    ("local",  local_reward,        True),
    ("fp",     fp_reward,           True),
]

__all__ = [
    "format_reward", "len_matching_reward", "global_reward", "local_reward",
    "fp_reward", "iou_reward", "REWARD_CHANNELS",
]
