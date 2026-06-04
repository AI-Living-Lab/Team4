#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_prec.py
  MUSEG r_M + 시간단위 precision 페널티 (설계 A).

      iou_reward = r_M(pred, gt) - λ * outside_ratio
      outside_ratio = (|P| - |P ∩ G|) / |P|     ∈ [0, 1]

  목적: 모델이 "큰 blanket 구간 하나"로 GT union 을 대충 덮는 mode collapse
        (멀티세그먼트 출력 능력 상실)를 깨고 분절 예측을 복원.

  기존 reward_functions_rM_fp.py 의 n_unmatched_pred(=FP penalty)는
  *세그먼트 개수* 단위(best-match IoU<0.3 인 pred 세그먼트 수)라 거칠다.
  blanket 한 개는 페널티가 최대 1개분 → overshoot 양에 비례하지 않아 collapse 를 못 막음.
  여기서는 "GT 에 없는데 예측한 *시간량*" 을 예측 총길이 대비 비율로 직접 감점한다.
    - blanket 이 GT-틈을 많이 덮을수록 outside_ratio↑ → 감점↑ → 분절 유도.
    - r_M 의 recall(모든 GT 덮기)은 그대로 유지.

  λ 는 env FP_PENALTY_K 로 오버라이드(sweep용). 미설정 시 0.5.
    주의: 비율 페널티(∈[0,1])라 0.5~1.0 권장. (rM_fp 의 0.2 는 개수 페널티용 스케일이라 다름.)

  주의: 원본 reward_functions.py 는 수정하지 않고 import 만 한다.
"""
import os
from typing import List, Tuple

# 원본에서 그대로 재사용 (원본 무수정).
from reward_functions import (
    r_M,                  # MUSEG Segment Matching reward (recall 측 유지)
    decode_vtg_time,      # time token → sec (max_time=999.9)
    _SEG_CAPTURE_RE,      # "from X to Y" 캡쳐
    format_reward,        # trainer 가 함께 import → 변경 없이 re-export
    _merge_intervals,     # union 병합
    _intersection_length, # 두 union 간 교집합 길이
)

# ---- 하이퍼파라미터 ----
LAMBDA_PREC = float(os.environ.get("FP_PENALTY_K", "0.5"))   # precision 페널티 가중 λ


def _parse_pred_segments(completion: str) -> List[Tuple[float, float]]:
    """r_M 과 동일 방식으로 pred segment 파싱 (start<end 인 것만)."""
    if not isinstance(completion, str):
        return []
    text = completion.replace("<|im_end|>", "").strip()
    preds: List[Tuple[float, float]] = []
    for s_str, e_str in _SEG_CAPTURE_RE.findall(text):
        s = decode_vtg_time(s_str)
        e = decode_vtg_time(e_str)
        if s is not None and e is not None and e > s:
            preds.append((s, e))
    return preds


def outside_ratio(completion: str,
                  gt_intervals: List[Tuple[float, float]]) -> float:
    """예측 시간 중 GT 밖(헛예측) 비율 = (|P| - |P∩G|) / |P|  ∈ [0,1].

    예측이 없으면 0(페널티 없음 — recall 은 r_M 이 이미 0 으로 처리).
    """
    preds = _parse_pred_segments(completion)
    if not preds:
        return 0.0
    P = _merge_intervals(preds)
    pred_len = sum(e - s for s, e in P)
    if pred_len <= 0:
        return 0.0
    G = _merge_intervals(list(gt_intervals)) if gt_intervals else []
    inter = _intersection_length(P, G) if G else 0.0
    outside = max(0.0, pred_len - inter)
    return outside / pred_len


def iou_reward(completion: str,
               gt_intervals: List[Tuple[float, float]],
               **kwargs) -> float:
    """MUSEG r_M + 시간단위 precision 페널티 (설계 A).

        iou = r_M - λ * outside_ratio

    r_M ∈ [0,1], outside_ratio ∈ [0,1]. blanket overshoot 가 클수록 강하게 감점.
    (GDPO advantage 는 group 내 정규화라 절대 부호/스케일 무관.)
    """
    base = r_M(completion, gt_intervals)
    penalty = LAMBDA_PREC * outside_ratio(completion, gt_intervals)
    return base - penalty


__all__ = ["format_reward", "iou_reward", "decode_vtg_time", "outside_ratio"]
