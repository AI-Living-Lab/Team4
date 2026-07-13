#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_rM_sep3_natural.py
  reward_functions_rM_sep3 의 chronus(natural-text 출력) 변형.

  ★ rM_sep3 와의 유일한 차이 = 출력 파싱 포맷.
     rM_sep3      : 'From <t\\d>{3}<tdot><t\\d> to <t\\d>{3}<tdot><t\\d>.'  (special_token)
     이 파일       : 'second{start}-second{end}. second{start}-second{end}. ...'  (chronus)
     → 채널 5개(format/count/global/local/precision)의 '의미·수식'은 100% 동일,
       세그먼트를 뽑는 정규식(_parse)과 format strict 패턴만 chronus 로 교체.

  채널 5개 (전부 [0,1], completion 전체 기준):
    format    — strict 'second{X.Y}-second{X.Y}' 나열만 (개수 중립)
    count     — 세그먼트 '개수' 매칭 (multi & pred==1 → 0 강벌)         [rM_sep3 동일]
    global    — MUSEG F1 (precision·recall 결합)                       [rM_sep 동일]
    local     — NGIoU best-match / max(n)                             [rM_sep 동일]
    precision — temporal precision = overlap(pred∩gt) / pred_len       [rM_sep3 동일]

  reward_weights 순서 [format, count, global, local, precision] (config 와 1:1).
  wandb 채널명은 load_reward_module 이 fn.__name__ 으로 자동 부여.
  ⚠️ GT interval 은 트레이너(gdpo_trainer_batch_natural.py)가 gpt(chronus) 답변에서
     별도 파싱해 gt_intervals 로 넘겨준다 — 이 모듈은 pred(completion) 만 파싱.
"""
import re
from typing import List, Tuple

# 포맷-무관 유틸은 원본 그대로 재사용 (수식 일관성 보장)
from reward_functions import _merge_intervals, ngiou_pair


# ============================================================
# 파싱 — chronus 'second{start}-second{end}' 세그먼트 추출
# ============================================================
_MAX_TIME = 999.9
# 세그먼트 캡처: second{정수 또는 소수}-second{...}  (구분자 '-', 세그먼트간 '. ')
_NAT_SEG_CAPTURE = re.compile(
    r"second\{\s*(\d+(?:\.\d+)?)\s*\}\s*-\s*second\{\s*(\d+(?:\.\d+)?)\s*\}"
)
# strict: 전체가 'second{X.Y}-second{X.Y}' 의 '. ' 나열로만 구성 (앞뒤/사이 잡음 불허)
_NAT_STRICT_SEG = r"second\{\d+(?:\.\d+)?\}-second\{\d+(?:\.\d+)?\}"
_NAT_STRICT_FULL = re.compile(
    r"^\s*(?:" + _NAT_STRICT_SEG + r")(?:\.\s+(?:" + _NAT_STRICT_SEG + r"))*\.?\s*$"
)


def _decode_nat_time(num_str: str) -> float:
    try:
        return min(float(num_str), _MAX_TIME)
    except (TypeError, ValueError):
        return None


def _parse(completion: str) -> List[Tuple[float, float]]:
    """completion 에서 (start, end) 세그먼트 추출 (start<end 인 것만). chronus 포맷."""
    if not isinstance(completion, str):
        return []
    text = completion.replace("<|im_end|>", "").strip()
    out: List[Tuple[float, float]] = []
    for s_str, e_str in _NAT_SEG_CAPTURE.findall(text):
        s = _decode_nat_time(s_str)
        e = _decode_nat_time(e_str)
        if s is not None and e is not None and e > s:
            out.append((s, e))
    return out


# ============================================================
# 1) format — strict + 개수 중립  (chronus)
# ============================================================
def format_reward(completion: str, **kwargs) -> float:
    if not isinstance(completion, str):
        return 0.0
    text = completion.replace("<|im_end|>", "").strip()
    if not text:
        return 0.0
    return 1.0 if _NAT_STRICT_FULL.match(text) else 0.0


# ============================================================
# 2) count — 세그먼트 '개수' 매칭  (rM_sep3 count_matching_reward 동일 로직)
# ============================================================
def count_matching_reward(completion: str, gt_intervals, **kwargs) -> float:
    if not gt_intervals:
        return 0.0
    n_g = len(gt_intervals)
    if n_g == 0:
        return 0.0
    n_p = len(_parse(completion))
    if n_g >= 2 and n_p == 1:          # multi 문제에서 single-collapse 강벌
        return 0.0
    return max(0.0, 1.0 - abs(n_p - n_g) / max(n_p, n_g, 1))


# ============================================================
# 3) global — MUSEG global matching (F1)  (rM_sep global_reward 동일)
# ============================================================
def global_reward(completion: str, gt_intervals, **kwargs) -> float:
    if not gt_intervals:
        return 0.0
    preds = _parse(completion)
    if not preds:
        return 0.0
    pred_u = _merge_intervals(preds)
    gt = list(gt_intervals)
    all_sel = sum(e - s for s, e in pred_u)
    all_gt = sum(e - s for s, e in gt)
    overlap = 0.0
    for gs, ge in gt:
        for ps, pe in pred_u:
            overlap += max(0.0, min(pe, ge) - max(ps, gs))
    precision = overlap / (all_sel + 1e-6)
    recall = overlap / (all_gt + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6)


# ============================================================
# 4) local — 세그먼트별 NGIoU best-match  (rM_sep local_reward 동일)
# ============================================================
def local_reward(completion: str, gt_intervals, **kwargs) -> float:
    if not gt_intervals:
        return 0.0
    preds = _parse(completion)
    gt = list(gt_intervals)
    if not preds:
        return 0.0
    n = max(len(preds), len(gt))
    if n == 0:
        return 0.0
    used = set()
    total = 0.0
    for p in preds:
        best, best_j = 0.0, -1
        for j, g in enumerate(gt):
            if j in used:
                continue
            score = ngiou_pair(g, p)
            if score > best:
                best, best_j = score, j
        if best_j >= 0:
            used.add(best_j)
            total += best
    return total / n


# ============================================================
# 5) precision — temporal precision = overlap(pred∩gt)/pred_len  (rM_sep3 동일)
# ============================================================
def precision_reward(completion: str, gt_intervals, **kwargs) -> float:
    if not gt_intervals:
        return 0.0
    preds = _parse(completion)
    if not preds:
        return 0.0
    pred_u = _merge_intervals(list(preds))
    gt_u = _merge_intervals([tuple(g) for g in gt_intervals])
    pred_len = sum(e - s for s, e in pred_u)
    if pred_len <= 0:
        return 0.0
    overlap = 0.0
    for gs, ge in gt_u:
        for ps, pe in pred_u:
            overlap += max(0.0, min(pe, ge) - max(ps, gs))
    return min(1.0, overlap / pred_len)


# trainer 가 읽는 채널 스펙 — (name, fn, needs_gt). 순서 = reward_weights.
iou_reward = global_reward   # load_reward_module 레거시 'iou' 슬롯 하위호환
REWARD_CHANNELS = [
    ("format",    format_reward,          False),
    ("count",     count_matching_reward,  True),
    ("global",    global_reward,          True),
    ("local",     local_reward,           True),
    ("precision", precision_reward,       True),
]

__all__ = [
    "format_reward", "count_matching_reward", "global_reward", "local_reward",
    "precision_reward", "iou_reward", "REWARD_CHANNELS", "_parse",
]
