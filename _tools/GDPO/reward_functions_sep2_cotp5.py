#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_sep2_cotp5.py
  reward_functions_rM_sep2 의 CoT(P5) 적용판 — 래핑이 아니라 카피-수정.

  CoT 완성형 답변:  <think> 자유 추론 </think><answer>From <t..> to <t..>. ...</answer>
  reward 는 전부 <answer> 블록만 파싱(think 의 시간범위가 개수/IoU 를 오염시키지 않게).

  채널(5개, 전부 [0,1] — wandb 표시 일관):
    format — <think>..</think><answer>..</answer> 구조 AND answer 가 strict
             'From <t\\d>{3}<tdot><t\\d> to ... .' 나열이면 1.0, else 0.0 (개수 중립)
    len    — sep2: graded 1-|n_p-n_g|/max(n_p,n_g,1), 단 multi(GT>=2) & pred==1 → 0.0
    global — MUSEG F1 = 2PR/(P+R) (precision 내장 → 큰-한방 억제)
    local  — NGIoU best-match 합 / max(n_p, n_g)
    fp     — precision = 1 - n_unmatched_pred/n_pred  (FP 적을수록 1; rMfp 로직의 [0,1]판)

  count(think N == pred 개수) 로직은 의도적으로 제외(모델 think 미파싱).
  trainer 호환: REWARD_CHANNELS 선언. 원본 reward_functions.py 는 import 만(무수정).
"""
import os
import re
from typing import List, Tuple

from reward_functions import (
    decode_vtg_time,
    _merge_intervals,
    _SEG_CAPTURE_RE,
    ngiou_pair,
    temporal_iou,
)

IOU_MATCH_THR = float(os.environ.get("IOU_MATCH_THR", "0.3"))   # fp 매칭 임계


# ============================================================
# 파싱 — answer 블록만
# ============================================================

def _answer_of(completion: str) -> str:
    """<answer>..</answer> 내용. 없으면 "" (→ 세그먼트 0 → CoT 구조 강제)."""
    if not isinstance(completion, str):
        return ""
    text = completion.replace("<|im_end|>", "")
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    return m.group(1) if m else ""


def _parse(text: str) -> List[Tuple[float, float]]:
    """text 에서 (start, end) 세그먼트 (start<end 만)."""
    if not isinstance(text, str):
        return []
    out: List[Tuple[float, float]] = []
    for s_str, e_str in _SEG_CAPTURE_RE.findall(text):
        s = decode_vtg_time(s_str)
        e = decode_vtg_time(e_str)
        if s is not None and e is not None and e > s:
            out.append((s, e))
    return out


# ============================================================
# 1) format — CoT 구조 + answer strict 나열 (개수 중립)
# ============================================================
_STRICT_SEG = r"[Ff]rom\s+(?:<t\d>){3}<tdot><t\d>\s+to\s+(?:<t\d>){3}<tdot><t\d>\s*\.\s*"
_STRICT_FULL = re.compile(r"^\s*(?:" + _STRICT_SEG + r")+$")
_COT_STRUCT = re.compile(r"^\s*<think>.*?</think>\s*<answer>(.*?)</answer>\s*$", re.S)


def format_reward(completion: str, **kwargs) -> float:
    if not isinstance(completion, str):
        return 0.0
    text = completion.replace("<|im_end|>", "").strip()
    m = _COT_STRUCT.match(text)
    if not m:
        return 0.0
    ans = m.group(1).strip()
    return 1.0 if (ans and _STRICT_FULL.match(ans)) else 0.0


# ============================================================
# 2) len — 개수 일치 (graded + multi-pred1 floor)  [sep2]
# ============================================================

def len_matching_reward(completion: str, gt_intervals, **kwargs) -> float:
    if not gt_intervals:
        return 0.0
    n_g = len(gt_intervals)
    if n_g == 0:
        return 0.0
    n_p = len(_parse(_answer_of(completion)))
    if n_g >= 2 and n_p == 1:          # multi 인데 1개 collapse → 강벌
        return 0.0
    return max(0.0, 1.0 - abs(n_p - n_g) / max(n_p, n_g, 1))


# ============================================================
# 3) global — MUSEG F1 (precision 내장)
# ============================================================

def global_reward(completion: str, gt_intervals, **kwargs) -> float:
    if not gt_intervals:
        return 0.0
    preds = _parse(_answer_of(completion))
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
# 4) local — NGIoU best-match / max(n)
# ============================================================

def local_reward(completion: str, gt_intervals, **kwargs) -> float:
    if not gt_intervals:
        return 0.0
    preds = _parse(_answer_of(completion))
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
# 5) fp — precision = 1 - n_unmatched_pred/n_pred  (rMfp 로직 [0,1]판)
# ============================================================

def fp_reward(completion: str, gt_intervals, **kwargs) -> float:
    if not gt_intervals:
        return 0.0
    preds = _parse(_answer_of(completion))
    if not preds:
        return 0.0
    n_fp = 0
    for ps, pe in preds:
        best = 0.0
        for gs, ge in gt_intervals:
            iou = temporal_iou(ps, pe, gs, ge)
            if iou > best:
                best = iou
        if best < IOU_MATCH_THR:
            n_fp += 1
    return 1.0 - n_fp / len(preds)


# ============================================================
# trainer 채널 스펙 — (name, fn, needs_gt). weights = [format, len, global, local, fp]
# ============================================================
iou_reward = global_reward   # 레거시 'iou' 슬롯 하위호환
REWARD_CHANNELS = [
    ("format", format_reward,       False),
    ("len",    len_matching_reward, True),
    ("global", global_reward,       True),
    ("local",  local_reward,        True),
    ("fp",     fp_reward,           True),
]

__all__ = [
    "format_reward", "len_matching_reward", "global_reward", "local_reward",
    "fp_reward", "iou_reward", "REWARD_CHANNELS",
]
