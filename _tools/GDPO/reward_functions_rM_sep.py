#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_rM_sep.py
  학습신호 세분화(separated) 버전 — MUSEG r_M 처럼 하나로 합치지 않고,
  채널을 4개로 분리해 GDPO 가 각 신호를 독립적으로(채널별 group-normalize) 학습하게 한다.

    채널: {format, len, global, local}
      format — 'From X to Y.' strict 구조 준수 (개수 중립: 정형 세그먼트 나열이면 1.0)
      len    — pred 세그먼트 개수 == gt 세그먼트 개수 면 1.0, 아니면 0.0 (binary; env 로 graded 토글)
      global — MUSEG global matching = F1(precision, recall)  ← precision 항 내장
      local  — 세그먼트별 NGIoU best-match 평균 / max(n_pred, n_gt)

  배경 / 의도 (multi-segment 붕괴 대응):
    1) 기존 r_G(NGIoU, precision 없음)는 'GT 전체를 덮는 한 개의 큰 세그먼트'를
       싸게 보상해 1-세그먼트 collapse 의 attractor 가 됐다. → global 을 MUSEG 원본
       F1 로 되돌려 precision 항을 복원, 큰-한방 전략을 무력화한다.
    2) r_G/r_L 을 r_M(0.5 합)으로 뭉치지 않고 따로 둔다. GDPO 의 _compute_gdpo_advantages
       가 채널마다 group 내 표준화 → 합치면 묻히던 r_L(위치) 신호가 단위분산으로 보존된다.
    3) len 채널로 '개수' 신호를 명시한다 (r_L 의 max(n) 분모와 보강).

  ⚠️ binary len 의 한계:
       이미 1-세그먼트로 무너진 그룹에선 모든 롤아웃이 len=0 → std=0 → advantage=0
       → 회복 gradient 없음(GDPO group-norm 의 구조적 한계). reward 설계는 '예방'이고,
       회복엔 clip-higher / temperature / num_generations(탐색 유지)가 함께 필요하다.
       near-collapse 에서 gradient 를 살리고 싶으면 env LEN_REWARD_MODE=graded 사용.

  ⚠️ 이 모듈을 쓰려면 trainer.load_reward_module 이 REWARD_CHANNELS 를 인식해야 한다
     (gdpo_trainer.py 의 load_reward_module 에 'if hasattr(mod, "REWARD_CHANNELS")' 분기 필요).

  원본 reward_functions.py 는 절대 수정하지 않고 import 만 한다.
"""
import os
import re
from typing import List, Tuple

# 원본에서 그대로 재사용 (원본 무수정):
#   decode_vtg_time : VTG time-token → 초
#   _merge_intervals: 겹침 제거 union
#   _SEG_CAPTURE_RE : pred 세그먼트 파싱 (정수 1~4자리 + <tdot> + 소수 1자리)
#   ngiou_pair      : 단일 (gt, pred) NGIoU (local 용, 원본 공식 그대로)
from reward_functions import (
    decode_vtg_time,
    _merge_intervals,
    _SEG_CAPTURE_RE,
    ngiou_pair,
)

# ── len 채널 모드: binary(기본, 요구사항대로) / graded(near-collapse gradient 생존) ──
#   launch 시점 env 값으로 고정 (이미 import 된 프로세스엔 영향 없음).
LEN_REWARD_MODE = os.environ.get("LEN_REWARD_MODE", "binary").lower()


# ============================================================
# 파싱 — IoU/개수 계열은 표준 capture 사용 (포맷 엄격성은 format 채널 전담)
# ============================================================

def _parse(completion: str) -> List[Tuple[float, float]]:
    """completion 에서 (start, end) 세그먼트 추출 (start<end 인 것만)."""
    if not isinstance(completion, str):
        return []
    text = completion.replace("<|im_end|>", "").strip()
    out: List[Tuple[float, float]] = []
    for s_str, e_str in _SEG_CAPTURE_RE.findall(text):
        s = decode_vtg_time(s_str)
        e = decode_vtg_time(e_str)
        if s is not None and e is not None and e > s:
            out.append((s, e))
    return out


# ============================================================
# 1) format — strict + 개수 중립
# ============================================================
# 응답 전체가 'From <t\d>{3}<tdot><t\d> to <t\d>{3}<tdot><t\d>.' 의
# 1개 이상 나열로만 구성돼야 1.0, 아니면 0.0.
#   - 세그먼트가 여러 개여도 전부 정형이면 1.0 → 개수를 벌하지 않음(multi-seg 억제 방지).
#   - 잡음(세그먼트 외 토큰)이 끼면 0.0.
_STRICT_SEG = r"[Ff]rom\s+(?:<t\d>){3}<tdot><t\d>\s+to\s+(?:<t\d>){3}<tdot><t\d>\s*\.\s*"
_STRICT_FULL = re.compile(r"^\s*(?:" + _STRICT_SEG + r")+$")


def format_reward(completion: str, **kwargs) -> float:
    if not isinstance(completion, str):
        return 0.0
    text = completion.replace("<|im_end|>", "").strip()
    if not text:
        return 0.0
    return 1.0 if _STRICT_FULL.match(text) else 0.0


# ============================================================
# 2) len — pred 세그먼트 개수 vs gt 세그먼트 개수
# ============================================================

def len_matching_reward(completion: str, gt_intervals, **kwargs) -> float:
    """개수 일치 보상.
      binary(기본): n_pred == n_gt → 1.0, else 0.0
      graded(env) : 1 - |n_pred - n_gt| / max(n_pred, n_gt, 1)  (near-collapse 에서 gradient 생존)
    """
    if not gt_intervals:
        return 0.0
    n_g = len(gt_intervals)
    n_p = len(_parse(completion))
    if n_g == 0:
        return 0.0
    if LEN_REWARD_MODE == "graded":
        return max(0.0, 1.0 - abs(n_p - n_g) / max(n_p, n_g, 1))
    return 1.0 if n_p == n_g else 0.0


# ============================================================
# 3) global — MUSEG global matching (F1, precision 내장)
# ============================================================
# MUSEG 원본(reward_global_matching)과 동일: pred union 을 merge 한 뒤
#   P = overlap / Σpred,  R = overlap / Σgt,  F1 = 2PR/(P+R)
# precision 항(Σpred 분모) 덕에 over-prediction(큰 한 방/헛쏨)이 직접 감점된다.

def global_reward(completion: str, gt_intervals, **kwargs) -> float:
    if not gt_intervals:
        return 0.0
    preds = _parse(completion)
    if not preds:
        return 0.0
    pred_u = _merge_intervals(preds)              # MUSEG: pred 겹침 제거
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
# 4) local — 세그먼트별 NGIoU best-match
# ============================================================
# 정렬-순차 페어링은 세그먼트 하나만 어긋나도 연쇄 오정렬되어 multi-seg 에 취약하다.
# → greedy best-match (각 pred 를 미사용 gt 중 NGIoU 최대인 것과 매칭) 후
#   분모 max(n_pred, n_gt) 로 평균 = over/under-prediction 양쪽에 개수 페널티.
#   NGIoU pair 공식 자체는 원본(ngiou_pair) 그대로.

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
# trainer 가 읽는 채널 스펙 — (name, fn, needs_gt)
# ============================================================
# load_reward_module 이 REWARD_CHANNELS 를 우선 사용하도록 분기 필요(아래 함수명/형식 고정).
# reward_weights 는 [format, len, global, local] 순서로 매칭됨 (예: [1,1,1,1]).
iou_reward = global_reward   # 하위호환: 레거시 load_reward_module 의 'iou' 슬롯
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
