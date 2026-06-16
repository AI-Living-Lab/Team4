#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_rM_sep2.py
  reward_functions_rM_sep.py 의 개선판 — len 채널만 교체.
    채널: {format, len, global, local}  (format/global/local 은 sep 그대로 재사용)

  [왜 sep2 인가]
    sep 의 binary len 은 "개수 학습"을 못 하고 데이터 prior(multi:single ≈ 4:6)만
    반영 → multi 문제에서도 pred=1 로 수렴(안전전략). multi 능력(GT>=2 정확히 맞춤)에
    single 정확(GT=1,pred=1)과 같은 점수를 주면 안 된다.

  [len 정책: graded + multi-pred1 floor]
    - 기본: graded = 1 - |n_pred - n_gt| / max(n_pred, n_gt, 1)
    - multi(GT>=2)인데 pred=1 → 0.0 강제(anti-single-collapse). graded(0.33/0.5)보다
      더 강하게 최하위로 밀어 그룹 내 advantage 스프레드를 키움.
    ※ "single=0.6 / multi=1.0 cap" 은 의도적으로 넣지 않음:
       GDPO advantage 는 prompt 그룹 내부에서 정규화되고(per_device_batch=1 → 한 step=
       한 prompt=한 그룹), single 과 multi 가 같은 정규화 그룹에 절대 안 섞이므로 절대
       스케일(cap)은 평균빼기에서 제거되어 효과 0(no-op). "multi 를 single 보다 더 가치
       있게"는 reward 값이 아니라 trainer 의 multi-GT per-sample advantage 가중
       (gdpo_trainer_sep2.py 의 multi_seg_weight) 으로 구현한다.

  trainer 호환: REWARD_CHANNELS 선언 → load_reward_module 이 그대로 사용.
  원본 reward_functions.py / reward_functions_rM_sep.py 는 수정하지 않고 import.
"""
from typing import List, Tuple

# 바뀌지 않는 채널/유틸은 sep 에서 그대로 재사용 (DRY, 일관성 보장)
from reward_functions_rM_sep import (
    format_reward,     # strict + 개수 중립
    global_reward,     # MUSEG F1 (precision 내장)
    local_reward,      # NGIoU best-match / max(n)
    _parse,            # 세그먼트 파싱
)


def len_matching_reward(completion: str, gt_intervals, **kwargs) -> float:
    """graded + multi-pred1 floor.
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


# trainer 가 읽는 채널 스펙 — (name, fn, needs_gt). 가중치는 [format, len, global, local] 순.
iou_reward = global_reward   # load_reward_module 레거시 'iou' 슬롯 하위호환
REWARD_CHANNELS = [
    ("format", format_reward,        False),
    ("len",    len_matching_reward,  True),
    ("global", global_reward,        True),
    ("local",  local_reward,         True),
]

__all__ = [
    "format_reward", "len_matching_reward", "global_reward", "local_reward",
    "iou_reward", "REWARD_CHANNELS",
]
