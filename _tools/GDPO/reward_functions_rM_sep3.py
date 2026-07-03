#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_rM_sep3.py
  reward_functions_rM_sep2 의 후속 — 채널 이름/의미 정리 + fp 를 precision(연속)으로 교체.

  채널 5개 (전부 [0,1], completion 전체 기준):
    format    — strict 'From <t\\d>{3}<tdot><t\\d> to ... .' 나열 (개수 중립)  [rM_sep 그대로]
    count     — 세그먼트 '개수' 매칭 (구 len)                              [rM_sep2 로직, 이름만 count]
    global    — MUSEG F1 (precision·recall 결합)                          [rM_sep 그대로]
    local     — NGIoU best-match / max(n)                                [rM_sep 그대로]
    precision — temporal precision = overlap(pred∩gt) / pred_len (구 fp 대체)

  [왜 sep3 인가 — fp → precision]
    구 fp = 1 - n_unmatched_pred/n_pred, 매칭기준 IoU>=0.3 계단.
      · 개수(over-segment)만 잡고 '과길이'는 못 잡음: GT를 덮으면 IoU가 0.3 위에
        남아 과길이여도 fp=1.0 (예: GT 8s를 13.4s로 예측 → IoU 0.60 → 무벌점).
      · single-seg 에선 0.3 계단이라 gradient 가 없음.
    precision = 예측 시간 중 GT에 겹치는 비율(overlap/pred_len) = temporal precision.
      · 과길이(오버런) 를 연속으로 벌: pred_len↑ → precision↓.
      · 헛쏨(over-segment) 도 연속으로 벌: 안 겹치는 pred 구간이 분모만 키움.
      · [0,1], 1 = 예측 시간이 전부 GT 안(딱 맞거나 더 짧음).
      · recall(전 구간 커버) 은 global/local 이 담당 → 이 채널은 precision 전담(연속).

  [이름 변경] len → count, fp → precision.
    wandb 는 rewards/<채널명> 으로 자동 로깅(load_reward_module 이 fn.__name__ = 채널명 부여).
    reward_weights 는 순서 [format, count, global, local, precision] 로 매칭.
  원본 reward_functions*.py 는 수정하지 않고 import 만.
"""
from typing import List, Tuple

# 바뀌지 않는 채널/유틸은 sep 에서 그대로 재사용 (DRY, 일관성 보장)
from reward_functions_rM_sep import (
    format_reward,     # strict + 개수 중립
    global_reward,     # MUSEG F1 (precision·recall 결합)
    local_reward,      # NGIoU best-match / max(n)
    _parse,            # 세그먼트 파싱
)
from reward_functions import _merge_intervals   # 겹침 제거 union


def count_matching_reward(completion: str, gt_intervals, **kwargs) -> float:
    """세그먼트 '개수' 매칭 (구 len_matching_reward — 동작 동일, 이름만 count).
      - multi(GT>=2) & pred==1 → 0.0  (single-collapse 강벌)
      - 그 외          → 1 - |n_pred - n_gt| / max(n_pred, n_gt, 1)
    """
    if not gt_intervals:
        return 0.0
    n_g = len(gt_intervals)
    if n_g == 0:
        return 0.0
    n_p = len(_parse(completion))
    # multi 문제에서 pred=1 collapse 는 무조건 0 (anti-single-collapse)
    if n_g >= 2 and n_p == 1:
        return 0.0
    return max(0.0, 1.0 - abs(n_p - n_g) / max(n_p, n_g, 1))


def precision_reward(completion: str, gt_intervals, **kwargs) -> float:
    """temporal precision = overlap(pred∩gt) / pred_len ∈ [0,1]  (구 fp 대체).

    구 fp(개수·IoU0.3 계단) 와 달리 '예측이 GT 밖으로 얼마나 삐져나갔나' 를 연속으로
    벌한다. pred 겹침은 merge 로 제거(이중계상 방지), gt 도 merge.
      · 과길이/오버런 → pred_len 커져 precision↓
      · 헛쏜 여분 세그먼트 → 겹침 없이 pred_len 만 키워 precision↓
      · 1.0 = 예측 시간이 전부 GT 위(오버런 0). recall 은 global/local 담당.
    """
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


# trainer 가 읽는 채널 스펙 — (name, fn, needs_gt). 가중치 순서 [format, count, global, local, precision].
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
    "precision_reward", "iou_reward", "REWARD_CHANNELS",
]
