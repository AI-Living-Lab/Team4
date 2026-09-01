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

import os
import re
from typing import List, Tuple

# r_M = w·r_G + (1-w)·r_L 의 r_G 가중. env RM_RG_WEIGHT 로 오버라이드 (blanket 억제 sweep용).
# 미설정 시 0.5 → 0.5·(r_G+r_L) 로 기존과 100% 동일 (backward compatible).
RM_RG_WEIGHT = float(os.environ.get("RM_RG_WEIGHT", "0.5"))


# "From X to Y[.]" 한 segment 매칭
# X, Y: (<t\d>){1,4}<tdot><t\d>  (정수부 1~4자리 + . + 소수 1자리)
_TIME_TOKEN_RE = r"(?:<t\d>){1,4}<tdot><t\d>"
_SEG_CAPTURE_RE = re.compile(
    r"[Ff]rom\s+(" + _TIME_TOKEN_RE + r")\s+to\s+(" + _TIME_TOKEN_RE + r")"
)
_SEG_MATCH_RE = re.compile(
    r"[Ff]rom\s+" + _TIME_TOKEN_RE + r"\s+to\s+" + _TIME_TOKEN_RE + r"\.?"
)


def decode_vtg_time(token_str: str, max_time: float = 999.9) -> float | None:
    """VTG time token 역변환. "<t0><t0><t3><t9><tdot><t0>" → 39.0"

    max_time: 디코드 값 상한(초). 5토큰 포맷 XXX.Y 의 표현 범위 0.0~999.9 에 맞춤.
              (이전 기본값 60.0 은 >60s 영상의 GT/pred 를 60 으로 뭉개 reward 를 왜곡함.)
    """
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


# ============================================================
# [원본 format_reward]
# ============================================================
# def format_reward(completion: str, **kwargs) -> float:
#     """'From X to Y.' 멀티-segment 포맷 준수도.
#
#     1.0 — segment들로만 구성 (잡음 없음)
#     0.5 — segment 있지만 약간의 잡음
#     0.0 — valid segment 없음
#     """
#     if not isinstance(completion, str):
#         return 0.0
#     text = completion.replace("<|im_end|>", "").strip()
#     if not text:
#         return 0.0
#
#     segments = _SEG_MATCH_RE.findall(text)
#     if not segments:
#         return 0.0
#
#     # segment 전부 제거하고 남는 잡음 측정
#     residual = _SEG_MATCH_RE.sub("", text)
#     residual_clean = re.sub(r"[\s.]", "", residual)  # 공백/점은 허용
#
#     if not residual_clean:
#         return 1.0
#     # 잡음 비율 10% 미만이면 부분 점수
#     if len(residual_clean) / max(len(text), 1) < 0.1:
#         return 0.5
#     return 0.0


# ============================================================
# [strict 버전 — 토큰 단위 채점]
# ============================================================

# Loose matcher — "from X to Y" 패턴에서 X, Y에 어떤 시퀀스의 시간 토큰이 와도 캡쳐
_SEG_LOOSE_RE = re.compile(
    r"[Ff]rom\s+((?:<t\d>|<tdot>)+)\s+to\s+((?:<t\d>|<tdot>)+)"
)


def _score_time_value(s: str) -> float:
    """한 time value(X 또는 Y)에 대한 strict 채점 (0 ~ 1).

    0.5 — <tdot> 존재
    0.5 — canonical 토큰 구성 (<t\\d>{3}<tdot><t\\d>{1})
    """
    score = 0.0
    has_tdot = "<tdot>" in s
    if has_tdot:
        score += 0.5
        parts = s.split("<tdot>")
        if len(parts) == 2:
            int_digits = re.findall(r"<t\d>", parts[0])
            dec_digits = re.findall(r"<t\d>", parts[1])
            if len(int_digits) == 3 and len(dec_digits) == 1:
                score += 0.5
    return score


def format_reward(completion: str, **kwargs) -> float:
    """'From X to Y.' segment-level strict format reward.

    각 segment의 X, Y 시간값을 독립 평가하고 평균.
    Per time value:
      0.5 — <tdot> 존재
      0.5 — canonical 토큰 구성 (<t\\d>{3}<tdot><t\\d>{1})
    """
    if not isinstance(completion, str):
        return 0.0
    text = completion.replace("<|im_end|>", "").strip()
    if not text:
        return 0.0

    matches = _SEG_LOOSE_RE.findall(text)
    if not matches:
        return 0.0

    scores = []
    for x_str, y_str in matches:
        scores.append(_score_time_value(x_str))
        scores.append(_score_time_value(y_str))

    return sum(scores) / len(scores)


# def iou_reward(
#     completion: str,
#     gt_intervals: List[Tuple[float, float]],
#     **kwargs,
# ) -> float:
#     """각 GT interval에 대해 best-match IoU 평균 (recall-style).

#     Args:
#         completion: 모델 출력
#         gt_intervals: [(start, end), ...] 단위=초

#     Returns:
#         0~1 평균 IoU. segment 파싱 실패 또는 GT 없으면 0.
#     """
#     if not gt_intervals:
#         return 0.0
#     if not isinstance(completion, str):
#         return 0.0

#     text = completion.replace("<|im_end|>", "").strip()

#     # 예측 segment 추출
#     preds = []
#     for start_str, end_str in _SEG_CAPTURE_RE.findall(text):
#         s = decode_vtg_time(start_str)
#         e = decode_vtg_time(end_str)
#         if s is not None and e is not None and e > s:
#             preds.append((s, e))

#     if not preds:
#         return 0.0

#     ious = []
#     for gs, ge in gt_intervals:
#         best = 0.0
#         for ps, pe in preds:
#             best = max(best, temporal_iou(ps, pe, gs, ge))
#         ious.append(best)

#     return sum(ious) / len(ious)


# ============================================================
# [F1-style iou_reward — 비활성 (MUSEG r_M으로 교체, 롤백용 보존)]
# ============================================================
# def iou_reward(
#     completion: str,
#     gt_intervals: List[Tuple[float, float]],
#     **kwargs,
# ) -> float:
#     """F1-style temporal IoU.
#
#     Recall:    각 GT의 best-match pred IoU 평균
#     Precision: 각 pred의 best-match GT IoU 평균
#     F1:        2 * P * R / (P + R)
#
#     → over-prediction (pred 길게 빼기, segment 추가하기)에 precision 페널티.
#     """
#     if not gt_intervals:
#         return 0.0
#     if not isinstance(completion, str):
#         return 0.0
#
#     text = completion.replace("<|im_end|>", "").strip()
#
#     preds = []
#     for start_str, end_str in _SEG_CAPTURE_RE.findall(text):
#         s = decode_vtg_time(start_str)
#         e = decode_vtg_time(end_str)
#         if s is not None and e is not None and e > s:
#             preds.append((s, e))
#
#     if not preds:
#         return 0.0
#
#     # Recall: 각 GT마다 best-match pred
#     recalls = []
#     for gs, ge in gt_intervals:
#         best = 0.0
#         for ps, pe in preds:
#             best = max(best, temporal_iou(ps, pe, gs, ge))
#         recalls.append(best)
#     recall = sum(recalls) / len(recalls)
#
#     # Precision: 각 pred마다 best-match GT
#     precisions = []
#     for ps, pe in preds:
#         best = 0.0
#         for gs, ge in gt_intervals:
#             best = max(best, temporal_iou(ps, pe, gs, ge))
#         precisions.append(best)
#     precision = sum(precisions) / len(precisions)
#
#     if precision + recall < 1e-8:
#         return 0.0
#     return 2 * precision * recall / (precision + recall)


# ============================================================
# [Set-IoU recall 버전 — 비활성(롤백용). iou_reward로는 안 쓰임]
# ============================================================

def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """겹치거나 인접한 interval들을 합쳐 union 반환."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals)
    merged = [sorted_iv[0]]
    for s, e in sorted_iv[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def _intersection_length(a: List[Tuple[float, float]],
                          b: List[Tuple[float, float]]) -> float:
    """두 union(merged) interval list 사이의 총 교집합 길이."""
    total = 0.0
    for as_, ae in a:
        for bs, be in b:
            total += max(0.0, min(ae, be) - max(as_, bs))
    return total


def iou_reward_setrecall(
    completion: str,
    gt_intervals: List[Tuple[float, float]],
    **kwargs,
) -> float:
    """[비활성/롤백용] Set-IoU recall: 모든 pred를 union으로 합친 뒤 GT union 대비 coverage.

        recall = |union(preds) ∩ union(GTs)| / |union(GTs)|

    Over-prediction 페널티 없음 (recall-only). 이전 best-match recall 모델과의
    공정 비교를 위해 의도적으로 recall 기반.
    """
    if not gt_intervals:
        return 0.0
    if not isinstance(completion, str):
        return 0.0

    text = completion.replace("<|im_end|>", "").strip()

    preds = []
    for start_str, end_str in _SEG_CAPTURE_RE.findall(text):
        s = decode_vtg_time(start_str)
        e = decode_vtg_time(end_str)
        if s is not None and e is not None and e > s:
            preds.append((s, e))

    if not preds:
        return 0.0

    pred_union = _merge_intervals(preds)
    gt_union = _merge_intervals(list(gt_intervals))
    gt_total = sum(e - s for s, e in gt_union)
    if gt_total <= 0:
        return 0.0

    inter = _intersection_length(pred_union, gt_union)
    return inter / gt_total


# ============================================================
# [MUSEG r_M — Segment Matching reward (활성)]
#   Paper: "MUSEG: Reinforcing Video Temporal Understanding via
#           Timestamp-Aware Multi-Segment Grounding" (arXiv:2505.20715)
#   r_M = (r_G + r_L) / 2
#     r_G — global NGIoU on union(GT) vs union(pred)
#     r_L — sorted sequential pairing NGIoU (G_i ↔ P_i), mismatch=0
# ============================================================

def _enclosing_interval_length(intervals: List[Tuple[float, float]]) -> float:
    """모든 interval을 덮는 최소 enclosing interval의 길이 |C|."""
    if not intervals:
        return 0.0
    lo = min(s for s, _ in intervals)
    hi = max(e for _, e in intervals)
    return max(0.0, hi - lo)


def _union_total_length(intervals: List[Tuple[float, float]]) -> float:
    """interval list의 union 총 길이 (겹침 제거)."""
    return sum(e - s for s, e in _merge_intervals(intervals))


def ngiou_segments(pred: List[Tuple[float, float]],
                    gt: List[Tuple[float, float]]) -> float:
    """NGIoU between two interval sets (union을 단일 region으로 취급).

        NGIoU = ½ · (1 + IoU − |C \\ (G ∪ P)| / |C|)

    IoU = |G ∩ P| / |G ∪ P|. C는 G와 P 모두를 덮는 최소 enclosing interval.
    값 범위: [0, 1]. 한쪽 빈 set이면 0.
    """
    if not pred or not gt:
        return 0.0

    pred_u = _merge_intervals(pred)
    gt_u = _merge_intervals(gt)

    inter_len = _intersection_length(pred_u, gt_u)
    union_len = _union_total_length(list(pred_u) + list(gt_u))
    iou_term = inter_len / union_len if union_len > 0 else 0.0

    c_len = _enclosing_interval_length(list(pred_u) + list(gt_u))
    outside_term = (c_len - union_len) / c_len if c_len > 0 else 0.0

    return 0.5 * (1 + iou_term - outside_term)


def ngiou_pair(g: Tuple[float, float], p: Tuple[float, float]) -> float:
    """단일 (g, p) interval pair NGIoU."""
    return ngiou_segments([p], [g])


def r_M(
    completion: str,
    gt_intervals: List[Tuple[float, float]],
    **kwargs,
) -> float:
    """MUSEG Segment Matching reward (multi-segment temporal grounding).

        r_M = (r_G + r_L) / 2

    r_G (global): union(GT) vs union(pred)의 NGIoU
    r_L (local):  start로 정렬한 뒤 sequential pairing G_i ↔ P_i NGIoU 평균.
                  미스매치(한쪽이 없음)는 NGIoU = 0.
                  분모는 max(|G|, |P|) — over/under-prediction 양쪽에 페널티.

    Hungarian이 아니라 정렬 후 순차 매칭 — MUSEG paper 정의 그대로.
    """
    if not gt_intervals:
        return 0.0
    if not isinstance(completion, str):
        return 0.0

    text = completion.replace("<|im_end|>", "").strip()
    preds = []
    for start_str, end_str in _SEG_CAPTURE_RE.findall(text):
        s = decode_vtg_time(start_str)
        e = decode_vtg_time(end_str)
        if s is not None and e is not None and e > s:
            preds.append((s, e))

    if not preds:
        return 0.0

    r_G = ngiou_segments(preds, list(gt_intervals))

    pred_sorted = sorted(preds, key=lambda x: x[0])
    gt_sorted = sorted(list(gt_intervals), key=lambda x: x[0])
    n = max(len(pred_sorted), len(gt_sorted))
    if n == 0:
        return 0.0
    local_scores = []
    for i in range(n):
        if i < len(pred_sorted) and i < len(gt_sorted):
            local_scores.append(ngiou_pair(gt_sorted[i], pred_sorted[i]))
        else:
            local_scores.append(0.0)
    r_L = sum(local_scores) / n

    return RM_RG_WEIGHT * r_G + (1.0 - RM_RG_WEIGHT) * r_L


# trainer/외부 호출용 export 이름 — 구현은 MUSEG r_M.
iou_reward = r_M
