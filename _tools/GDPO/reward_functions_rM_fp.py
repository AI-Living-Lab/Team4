#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_rM_fp.py
  MUSEG r_M (Segment Matching reward) + false-positive(FP) penalty.

    total_iou_reward = r_M(pred, gt) - K * n_unmatched_pred

  목적: MUSEG r_M 이 multi-seg 출력은 학습시켰지만 precision 이 무너진
        (FP rate 40% -> 55%, precision 60 -> 45) rebound 를 직접 조준.
        r_M = 0.5*(r_G + r_L) 로, r_G(global)는 union(pred) vs union(gt) NGIoU,
        r_L(local)은 정렬 위치순 페어링 NGIoU. **직접 precision 항이 없음** —
        base reward(F1 bestmatch)에 있던 precision 항이 r_M 전환 때 사라진 것이
        FP 폭증의 핵심. unmatched pred 1개당 -K 페널티로 그 precision 항을 복원.
        (주의: r_M 은 "pure union IoU" 가 아니라 union-global + 위치순-local 하이브리드.)

  n_unmatched_pred:
    - 각 pred segment에 대해 모든 GT와의 temporal_iou 중 최댓값(bestmatch)을 구함
    - bestmatch IoU < TAU 이면 unmatched(헛쏨, FP)로 카운트
    - TAU = 0.3  (평가 메트릭 R@0.3 / FP rate per pred seg 정의와 align)
    - bestmatch 로직 출처: reward_functions.py:183-233 (옛 F1 commented precision 항)

  주의: 원본 reward_functions.py는 절대 수정하지 않고 import만 한다.
        (예진 학습이 원본을 import 중일 수 있으므로 안전성 확보)
"""
from typing import List, Tuple

# 원본에서 그대로 재사용 (원본 무수정).
#   r_M           : MUSEG Segment Matching reward (활성 iou_reward 구현체)
#   temporal_iou  : pairwise temporal IoU
#   decode_vtg_time, _SEG_CAPTURE_RE : pred segment 파싱용 (r_M과 동일 방식)
#   format_reward : trainer가 함께 import -> 변경 없이 re-export
from reward_functions import (
    r_M,
    temporal_iou,
    decode_vtg_time,
    _SEG_CAPTURE_RE,
    format_reward,
)

# ---- 하이퍼파라미터 ----
# k 는 env FP_PENALTY_K 로 오버라이드 (k sweep 용). 미설정 시 0.2.
#   주의: 이미 import 된(=실행 중인) 프로세스엔 영향 없음. launch 시점 env 값으로 고정.
import os
FP_PENALTY_K = float(os.environ.get("FP_PENALTY_K", "0.2"))   # unmatched pred 1개당 감점
IOU_MATCH_THRESHOLD = 0.3   # bestmatch IoU < TAU 이면 unmatched(FP)


def _parse_pred_segments(completion: str) -> List[Tuple[float, float]]:
    """r_M과 동일한 방식으로 pred segment 파싱 (start<end 인 것만)."""
    if not isinstance(completion, str):
        return []
    text = completion.replace("<|im_end|>", "").strip()
    preds: List[Tuple[float, float]] = []
    for start_str, end_str in _SEG_CAPTURE_RE.findall(text):
        s = decode_vtg_time(start_str)
        e = decode_vtg_time(end_str)
        if s is not None and e is not None and e > s:
            preds.append((s, e))
    return preds


def n_unmatched_pred(completion: str,
                     gt_intervals: List[Tuple[float, float]],
                     threshold: float = IOU_MATCH_THRESHOLD) -> int:
    """bestmatch temporal IoU < threshold 인 pred segment 개수 (= FP 개수).

    각 pred에 대해 모든 GT와의 temporal_iou 중 최댓값을 구하고, 그 값이
    threshold 미만이면 unmatched(헛쏨)로 카운트. 원본 F1 commented 코드
    (reward_functions.py:222-229, precision 루프)의 bestmatch 로직과 동일.
    """
    if not gt_intervals:
        return 0
    preds = _parse_pred_segments(completion)
    if not preds:
        return 0
    n = 0
    for ps, pe in preds:
        best = 0.0
        for gs, ge in gt_intervals:
            iou = temporal_iou(ps, pe, gs, ge)
            if iou > best:
                best = iou
        if best < threshold:
            n += 1
    return n


def iou_reward(completion: str,
               gt_intervals: List[Tuple[float, float]],
               **kwargs) -> float:
    """MUSEG r_M + FP penalty.

        total = r_M(pred, gt) - K * n_unmatched_pred

    r_M in [0, 1]. FP가 많으면 total은 음수가 될 수 있음(GDPO advantage는
    group 내 정규화이므로 절대 부호/스케일은 무관).
    """
    base = r_M(completion, gt_intervals)
    penalty = FP_PENALTY_K * n_unmatched_pred(completion, gt_intervals)
    return base - penalty


__all__ = ["format_reward", "iou_reward", "decode_vtg_time", "n_unmatched_pred"]
