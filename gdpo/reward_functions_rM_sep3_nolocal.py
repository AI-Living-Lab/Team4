#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_rM_sep3_nolocal.py
  reward_functions_rM_sep3 의 ablation — **local 채널만 제거**한 버전.
  (reward w/o local ablation: 세그먼트별 NGIoU best-match(segment-level) 신호를 뺐을 때의 영향 측정용)

  채널 4개 (sep3 에서 local 제외, 전부 [0,1]):
    format    — strict 포맷 (개수 중립)                  [sep 재사용]
    count     — 세그먼트 '개수' 매칭                       [sep3 재사용]
    global    — MUSEG global matching F1 (set-level)     [sep 재사용]
    precision — temporal precision = overlap/pred_len     [sep3 재사용]

  ⚠️ 원본 reward_functions_rM_sep3.py / reward_functions_rM_sep.py 는 **수정하지 않고 import 만**.
     local 을 뺐으므로 reward_weights 는 4개 [format, count, global, precision] 로 매칭
     (config_sep3_nolocal.yaml). 채널명=로깅명(wandb rewards/<name> + 데이터셋별 rewards/<name>/<tag>).
"""
# 바뀌지 않는 채널은 원본에서 그대로 재사용 (DRY, sep3 와 값 완전 동일)
from reward_functions_rM_sep import (
    format_reward,     # strict + 개수 중립
    global_reward,     # MUSEG F1 (set-level)
)
from reward_functions_rM_sep3 import (
    count_matching_reward,   # 세그먼트 개수 매칭 (구 len)
    precision_reward,        # temporal precision (overlap/pred_len)
)


# 채널 스펙 — (name, fn, needs_gt). local 제거 → 4채널. 가중치 순서 [format, count, global, precision].
iou_reward = global_reward   # load_reward_module 레거시 'iou' 슬롯 하위호환
REWARD_CHANNELS = [
    ("format",    format_reward,          False),
    ("count",     count_matching_reward,  True),
    ("global",    global_reward,          True),
    ("precision", precision_reward,       True),
]

__all__ = [
    "format_reward", "count_matching_reward", "global_reward",
    "precision_reward", "iou_reward", "REWARD_CHANNELS",
]
