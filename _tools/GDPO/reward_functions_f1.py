#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_f1.py
  base F1 (bestmatch) reward 베이스라인용 wrapper.

    iou_reward = iou_reward_f1   (Recall=mean_GT(max IoU), Precision=mean_pred(max IoU), F1=2PR/(P+R))

  목적: 새 v5b base 위에서 r_M / rM_fp 와 "동일 조건, reward만 F1" 비교(§6.5)를 위한
        F1 베이스라인. 원본 reward_functions.py 의 iou_reward_f1 을 **이름으로 명시적 import**
        하여, 원본의 `iou_reward = iou_reward_f1 / r_M` 별칭 토글 상태와 무관하게 항상 F1 을 쓴다.

  주의: 원본 reward_functions.py 는 절대 수정하지 않고 import 만 한다(§3, 예진 학습 live import 안전).
        format_reward 는 원본 그대로 re-export.
"""
# 원본에서 그대로 재사용 (원본 무수정).
#   iou_reward_f1 : base F1 bestmatch reward (Recall/Precision/F1)
#   format_reward : trainer 가 함께 import -> 변경 없이 re-export
from reward_functions import iou_reward_f1 as iou_reward, format_reward  # noqa: F401
