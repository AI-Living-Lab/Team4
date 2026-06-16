#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_cot_p3.py — P3 CoT 용 reward (P2 계승, ★<time> 태그 강제판).

P2 와의 차이 (데이터/프롬프트):
  P2 think:  <think>... From <t..><tdot><t.> to <t..><tdot><t.> ...</think>   (자유 think 안 토큰범위)
  P3 think:  <think>... <time>From <t..><tdot><t.> to <t..><tdot><t.></time>, <time>...</time> ...</think>
             (새 프롬프트가 각 구간을 <time> </time> 로 감싸도록 명시 — unav100_v2_p3.json)

  → 따라서 P3 reward 는 think 의 시간범위를 **<time>...</time> 안에서만** 파싱한다.
    (P2 처럼 think 어디서나 토큰범위를 줍지 않음 → "구간을 <time> 로 감싸라" 라는 새 포맷을
     timestamp/format reward 가 실제로 강제하도록.)

answer 포맷은 P2 와 동일 (From <tok> to <tok>. ...) → iou_reward 는 P2 와 똑같이 재사용.

3개 reward 채널 (메인 gdpo_trainer 가 convention 으로 format/iou + timestamp 로딩):
  1. format_reward     — 구조(0.5) + think 에 유효한 <time> 토큰범위(0.25) + answer 유효(0.25).
  2. iou_reward        — r_M(answer) − K·FP. answer 블록만 (reward_functions_rM_fp_cot 재사용, 무수정).
  3. timestamp_reward  — think 의 <time> 토큰범위 ↔ GT IoU (content). GT 마다 best-match 평균(recall형).

원본 무수정 import. FP_PENALTY_K env 는 iou_reward 가 읽음.
"""
import re
from typing import List, Tuple

from reward_functions import temporal_iou, decode_vtg_time, _SEG_CAPTURE_RE   # noqa: E402
from reward_functions_rM_fp_cot import iou_reward, _extract_block             # noqa: E402  (answer-only r_M−FP)

# 시간토큰 1개: (자릿수)+ <tdot> (소수1). 예: <t0><t0><t6><tdot><t1>
_TOK = r"(?:<t\d>)+<tdot><t\d>"
# <time>...</time> 블록 (think 안의 각 구간 표기).
_TIME_BLOCK_RE = re.compile(r"<time>(.*?)</time>", re.S)
# 한 <time> 블록 속 "토큰 to/- 토큰" 범위 (from 접두 유무 무관).
_PAIR_RE = re.compile(r"(" + _TOK + r")\s*(?:to|-)\s*(" + _TOK + r")")
_STRUCT_RE = re.compile(r"\s*<think>.*?</think>\s*<answer>(.*?)</answer>\s*", re.S)


def _parse_think_segs(completion: str) -> List[Tuple[float, float]]:
    """think 블록의 **<time>...</time> 안** 시간토큰 범위 → [(s,e)] 초 (s<e 만).

    P3 핵심: <time> 태그 밖에 적은 토큰범위는 무시 → '구간을 <time> 로 감싸라' 강제.
    """
    if not isinstance(completion, str):
        return []
    think = _extract_block(completion, "think") or ""
    out = []
    for block in _TIME_BLOCK_RE.findall(think):
        for a, b in _PAIR_RE.findall(block):
            s, e = decode_vtg_time(a), decode_vtg_time(b)
            if s is not None and e is not None and e > s:
                out.append((s, e))
    return out


def format_reward(completion: str, **kwargs) -> float:
    """구조(0.5) + think 에 유효한 <time> 토큰범위(0.25) + answer 유효 세그(0.25). 최대 1.0."""
    if not isinstance(completion, str):
        return 0.0
    text = completion.replace("<|im_end|>", "").strip()
    m = _STRUCT_RE.fullmatch(text)
    if not m:
        return 0.0
    score = 0.5
    if _parse_think_segs(text):          # <time> 안 유효 범위가 1개라도 있어야 가점
        score += 0.25
    if _SEG_CAPTURE_RE.findall(m.group(1)):
        score += 0.25
    return score


def timestamp_reward(completion: str,
                     gt_intervals: List[Tuple[float, float]],
                     **kwargs) -> float:
    """think 의 <time> 토큰범위 ↔ GT IoU. GT 마다 best-match 평균(recall형) ∈ [0,1].
    <time> 범위 없거나 GT 없으면 0 (= 추론을 <time> 로 안 적으면 보상 0)."""
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
