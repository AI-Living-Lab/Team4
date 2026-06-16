#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_cot_final.py — 최종 "다 때려박기" CoT reward.

진단(sft_7b_unav_v8_rl_cot_p2_*): 모델이 "긴 1개 blanket" 으로 붕괴 →
  멀티세그(F1_multi 22)·짧은 GT(<5s mIoU 5~10) 천장. P2/clip-higher/FP↓ 다 ~28% 막힘.
  담요는 (a) 1개라 FP페널티 회피, (b) r_G coverage 챙김, (c) 과잉길이는 무벌점이라 도망.

처방 (세 실패 모드 동시 차단, 서로 안 싸우게):
  iou = r_M_bestmatch  −  λ·outside_ratio  −  κ·FN_ratio
    · r_M_bestmatch : 순차쌍(brittle) 대신 best-match recall/precision → "3개 중 2개"도 보상
                       (멀티가 살짝 어긋나도 부당벌점 안 받게 → 멀티 생존)
    · λ·outside     : 예측시간 중 GT 밖 비율 → 담요의 과잉길이 직격 (FP가 못 잡던 것)
    · κ·FN          : 매칭 안 된 GT 비율 → 미탐(이벤트 놓침) 벌 → 다 덮게 (under-pred 차단)
  + CoT: format / timestamp 채널은 reward_functions_cot_p2 그대로 재사용.

env knob:
  FP_PENALTY_K (λ, outside 가중)   미설정 0.4
  FN_PENALTY_K (κ, FN 가중)        미설정 0.1
  IOU_MATCH_THR (매칭 IoU 임계)    미설정 0.3
  RM_RG_WEIGHT (r_M 의 r_G 가중)   reward_functions 가 읽음(미설정 0.5)

3채널: [format, iou, timestamp]  (메인 trainer 가변로딩)
"""
import os
import re
from typing import List, Tuple

from reward_functions import (
    ngiou_pair, ngiou_segments, decode_vtg_time, _SEG_CAPTURE_RE,
    _merge_intervals, _intersection_length, RM_RG_WEIGHT,
)
from reward_functions_cot_p2 import format_reward, timestamp_reward  # CoT 채널 재사용

LAMBDA_OUT = float(os.environ.get("FP_PENALTY_K", "0.4"))   # λ : outside 페널티
KAPPA_FN   = float(os.environ.get("FN_PENALTY_K", "0.1"))   # κ : FN 페널티
IOU_THR    = float(os.environ.get("IOU_MATCH_THR", "0.3"))  # 매칭 임계


def _parse_answer_segments(completion: str) -> List[Tuple[float, float]]:
    """answer 블록(없으면 전체)에서 'From <t..> to <t..>' → [(s,e)] 초 (e>s)."""
    if not isinstance(completion, str):
        return []
    text = completion.replace("<|im_end|>", "").strip()
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    block = m.group(1) if m else text
    out = []
    for s_str, e_str in _SEG_CAPTURE_RE.findall(block):
        s, e = decode_vtg_time(s_str), decode_vtg_time(e_str)
        if s is not None and e is not None and e > s:
            out.append((s, e))
    return out


def _r_M_bestmatch(preds, gt) -> float:
    """best-match r_M = w·r_G + (1-w)·r_L_f1.
      r_G  : union(GT) vs union(pred) NGIoU (기존)
      r_L  : recall(각 GT의 best pred ngiou) 와 precision(각 pred의 best GT ngiou) 의 F1.
    """
    if not preds or not gt:
        return 0.0
    r_G = ngiou_segments(preds, list(gt))
    recall = sum(max(ngiou_pair(g, p) for p in preds) for g in gt) / len(gt)
    precision = sum(max(ngiou_pair(g, p) for g in gt) for p in preds) / len(preds)
    r_L = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    w = RM_RG_WEIGHT
    return w * r_G + (1.0 - w) * r_L


def _outside_ratio(preds, gt) -> float:
    """예측 총길이 중 GT 밖 비율 = (|P| − |P∩G|)/|P| ∈ [0,1]. 담요일수록 ↑."""
    if not preds:
        return 0.0
    pu = _merge_intervals(preds)
    pred_len = sum(e - s for s, e in pu)
    if pred_len <= 0:
        return 0.0
    inter = _intersection_length(pu, _merge_intervals(list(gt))) if gt else 0.0
    return max(0.0, pred_len - inter) / pred_len


def _fn_ratio(preds, gt) -> float:
    """매칭 안 된 GT 비율 = (best-match IoU<thr 인 GT 수)/|GT| ∈ [0,1]. 미탐일수록 ↑."""
    if not gt:
        return 0.0
    if not preds:
        return 1.0
    unmatched = 0
    for g in gt:
        best = max(ngiou_pair(g, p) for p in preds)   # ngiou∈[0,1], IoU 대용
        if best < IOU_THR:
            unmatched += 1
    return unmatched / len(gt)


def iou_reward(completion: str, gt_intervals: List[Tuple[float, float]], **kwargs) -> float:
    """r_M_bestmatch − λ·outside − κ·FN. answer 블록만 사용."""
    if not gt_intervals or not isinstance(completion, str):
        return 0.0
    preds = _parse_answer_segments(completion)
    if not preds:
        return 0.0
    base = _r_M_bestmatch(preds, gt_intervals)
    pen = LAMBDA_OUT * _outside_ratio(preds, gt_intervals) + KAPPA_FN * _fn_ratio(preds, gt_intervals)
    return base - pen


__all__ = ["format_reward", "iou_reward", "timestamp_reward",
           "_r_M_bestmatch", "_outside_ratio", "_fn_ratio", "_parse_answer_segments"]
