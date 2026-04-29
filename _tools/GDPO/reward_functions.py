#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions.py
  GDPO 학습용 리워드 함수 (multi-segment temporal grounding).

출력 포맷 가정:
  "From <t0><t0><t0><t5><tdot><t2> to <t0><t0><t1><t3><tdot><t9>. From ..."

리워드:
  1. format_reward — "From X to Y." 패턴 준수도 (0.0 / 0.5 / 1.0)
  2. iou_reward    — GT interval별 best temporal IoU 평균 (0~1)
"""

import re
from typing import List, Tuple


# "From X to Y[.]" 한 segment 매칭
# X, Y: (<t\d>){1,4}<tdot><t\d>  (정수부 1~4자리 + . + 소수 1자리)
_TIME_TOKEN_RE = r"(?:<t\d>){1,4}<tdot><t\d>"
_SEG_CAPTURE_RE = re.compile(
    r"[Ff]rom\s+(" + _TIME_TOKEN_RE + r")\s+to\s+(" + _TIME_TOKEN_RE + r")"
)
_SEG_MATCH_RE = re.compile(
    r"[Ff]rom\s+" + _TIME_TOKEN_RE + r"\s+to\s+" + _TIME_TOKEN_RE + r"\.?"
)


def decode_vtg_time(token_str: str, max_time: float = 60.0) -> float | None:
    """VTG time token 역변환. "<t0><t0><t3><t9><tdot><t0>" → 39.0"""
    if "<tdot>" not in token_str:
        return None
    parts = token_str.split("<tdot>")
    int_part = re.findall(r"<t(\d)>", parts[0])
    dec_part = re.findall(r"<t(\d)>", parts[1]) if len(parts) > 1 else []
    if not int_part:
        return None
    integer_part = int("".join(int_part))
    decimal_part = int(dec_part[0]) if dec_part else 0
    t = integer_part + decimal_part / 10.0
    return min(t, max_time)


def temporal_iou(ps: float, pe: float, gs: float, ge: float) -> float:
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = (pe - ps) + (ge - gs) - inter
    return inter / union if union > 0 else 0.0


def format_reward(completion: str, **kwargs) -> float:
    """'From X to Y.' 멀티-segment 포맷 준수도.

    1.0 — segment들로만 구성 (잡음 없음)
    0.5 — segment 있지만 약간의 잡음
    0.0 — valid segment 없음
    """
    if not isinstance(completion, str):
        return 0.0
    text = completion.replace("<|im_end|>", "").strip()
    if not text:
        return 0.0

    segments = _SEG_MATCH_RE.findall(text)
    if not segments:
        return 0.0

    # segment 전부 제거하고 남는 잡음 측정
    residual = _SEG_MATCH_RE.sub("", text)
    residual_clean = re.sub(r"[\s.]", "", residual)  # 공백/점은 허용

    if not residual_clean:
        return 1.0
    # 잡음 비율 10% 미만이면 부분 점수
    if len(residual_clean) / max(len(text), 1) < 0.1:
        return 0.5
    return 0.0


def iou_reward(
    completion: str,
    gt_intervals: List[Tuple[float, float]],
    **kwargs,
) -> float:
    """각 GT interval에 대해 best-match IoU 평균 (recall-style).

    Args:
        completion: 모델 출력
        gt_intervals: [(start, end), ...] 단위=초

    Returns:
        0~1 평균 IoU. segment 파싱 실패 또는 GT 없으면 0.
    """
    if not gt_intervals:
        return 0.0
    if not isinstance(completion, str):
        return 0.0

    text = completion.replace("<|im_end|>", "").strip()

    # 예측 segment 추출
    preds = []
    for start_str, end_str in _SEG_CAPTURE_RE.findall(text):
        s = decode_vtg_time(start_str)
        e = decode_vtg_time(end_str)
        if s is not None and e is not None and e > s:
            preds.append((s, e))

    if not preds:
        return 0.0

    ious = []
    for gs, ge in gt_intervals:
        best = 0.0
        for ps, pe in preds:
            best = max(best, temporal_iou(ps, pe, gs, ge))
        ious.append(best)

    return sum(ious) / len(ious)
