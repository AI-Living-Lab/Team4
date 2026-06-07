#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_cot_p2.py — P2 CoT 용 reward (CoT 개선 STEP 6, ★시간토큰 파싱판).

★ 수정(2026-06-07): 모델이 think 에 <timestep>X.X</timestep> 데시멀이 아니라 **시간토큰**
   (<t0><t0><t6><tdot><t1> 형태)을 쓴다는 걸 실측 → reward 가 think 의 *시간토큰 범위*를
   파싱하도록 변경. (이전 <timestep> 파싱은 모델 출력과 안 맞아 timestamp 항상 0 이었음.)

가정 포맷 (think 형식은 자유, 단 시간토큰 범위가 있어야 content 보상):
  <think>From <t0><t0><t6><tdot><t1> to <t0><t1><t4><tdot><t7> the people play tennis.</think>
  <answer>From <t0><t0><t6><tdot><t1> to <t0><t1><t4><tdot><t7>.</answer>

3개 reward 채널 (메인 gdpo_trainer 가 가변로딩: format/iou + timestamp):
  1. format_reward     — <think>..</think><answer>..</answer> 구조(0.5) + think 에 시간토큰 범위(0.25) + answer 유효(0.25).
  2. iou_reward        — r_M(answer) − K·FP. answer 블록만 (reward_functions_rM_fp_cot 재사용).
  3. timestamp_reward  — think 의 시간토큰 범위 ↔ GT IoU (content). GT 마다 best-match 평균(recall형).
                         = "추론에 적은 시간이 실제 정답과 일치" 를 보상.

원본 무수정 import. FP_PENALTY_K env 는 iou_reward 가 읽음.
"""
import re
from typing import List, Tuple

from reward_functions import temporal_iou, decode_vtg_time, _SEG_CAPTURE_RE   # noqa: E402
from reward_functions_rM_fp_cot import iou_reward, _extract_block             # noqa: E402  (answer-only r_M−FP)

# 시간토큰 1개: (자릿수)+ <tdot> (소수1). 예: <t0><t0><t6><tdot><t1>
_TOK = r"(?:<t\d>)+<tdot><t\d>"
# think 속 "토큰 to/- 토큰" 범위 (from 접두 유무 무관)
_THINK_RANGE_RE = re.compile(_TOK + r"\s*(?:to|-)\s*" + _TOK)
_PAIR_RE = re.compile(r"(" + _TOK + r")\s*(?:to|-)\s*(" + _TOK + r")")
_STRUCT_RE = re.compile(r"\s*<think>.*?</think>\s*<answer>(.*?)</answer>\s*", re.S)


def _parse_think_segs(completion: str) -> List[Tuple[float, float]]:
    """think 블록의 시간토큰 범위(<t..> to/- <t..>) → [(s,e)] 초 (s<e 만)."""
    if not isinstance(completion, str):
        return []
    think = _extract_block(completion, "think") or ""
    out = []
    for a, b in _PAIR_RE.findall(think):
        s, e = decode_vtg_time(a), decode_vtg_time(b)
        if s is not None and e is not None and e > s:
            out.append((s, e))
    return out


def format_reward(completion: str, **kwargs) -> float:
    """구조(0.5) + think 에 시간토큰 범위(0.25) + answer 유효 세그(0.25). 최대 1.0."""
    if not isinstance(completion, str):
        return 0.0
    text = completion.replace("<|im_end|>", "").strip()
    m = _STRUCT_RE.fullmatch(text)
    if not m:
        return 0.0
    score = 0.5
    if _parse_think_segs(text):
        score += 0.25
    if _SEG_CAPTURE_RE.findall(m.group(1)):
        score += 0.25
    return score


def timestamp_reward(completion: str,
                     gt_intervals: List[Tuple[float, float]],
                     **kwargs) -> float:
    """think 의 시간토큰 범위 ↔ GT IoU. GT 마다 best-match 평균(recall형) ∈ [0,1].
    think 에 시간토큰 범위 없거나 GT 없으면 0 (= 추론에 시간 안 적으면 보상 0)."""
    ts = _parse_think_segs(completion)
    if not ts or not gt_intervals:
        return 0.0
    recalls = []
    for gs, ge in gt_intervals:
        best = 0.0
        for ps, pe in ts:
            best = max(best, temporal_iou(ps, pe, gs, ge))
        recalls.append(best)
    return sum(recalls) / len(recalls)


__all__ = ["format_reward", "iou_reward", "timestamp_reward", "_parse_think_segs"]
