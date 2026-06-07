#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_cot_p4.py — P4(자유 think) CoT 용 reward (CoT 개선 STEP 6, P4판).

가정 포맷 (think 형식 제약 없음, 자연어):
  <think>The relevant event spans 3.3 to 5.6, 9.1 to 10.0 seconds.</think>
  <answer>From <t0><t0><t3><tdot><t3> to <t0><t0><t5><tdot><t6>. ...</answer>

★ P2 와의 차이: **timestep 파싱/content reward 없음 → format + iou 2채널만.**
  (P2[timestep 강제+content] vs P4[자유 think] 비교로 "구조 강제가 도움되나" 확인용.
   P4 는 think 내용을 채점하지 않음 — 비어있지만 않으면 OK.)

채널 (메인 gdpo_trainer 가 timestamp/modality 없으면 2채널로 로드):
  1. format_reward  — <think>..</think><answer>..</answer> 구조 + think 비어있지 않음 + answer 유효.
  2. iou_reward     — r_M(answer) − K·FP. answer 블록만 (reward_functions_rM_fp_cot 재사용).

원본 무수정 import. FP_PENALTY_K env 는 iou_reward 가 읽음.
"""
import re

from reward_functions import _SEG_CAPTURE_RE                          # noqa: E402
from reward_functions_rM_fp_cot import iou_reward, _extract_block     # noqa: E402  (answer-only r_M−FP)

_STRUCT_RE = re.compile(r"\s*<think>.*?</think>\s*<answer>(.*?)</answer>\s*", re.S)


def format_reward(completion: str, **kwargs) -> float:
    """구조(0.5) + think 비어있지 않음(0.25) + answer 유효 세그(0.25). 최대 1.0.
    P1(빈 think) 대비 'think 채우기' 를 유도하되, 내용 형식은 강제하지 않음(P4 핵심)."""
    if not isinstance(completion, str):
        return 0.0
    text = completion.replace("<|im_end|>", "").strip()
    m = _STRUCT_RE.fullmatch(text)
    if not m:
        return 0.0
    score = 0.5
    think = (_extract_block(text, "think") or "").strip()
    if len(think) >= 5:          # 비어있지 않음(자유 추론 존재)
        score += 0.25
    if _SEG_CAPTURE_RE.findall(m.group(1)):
        score += 0.25
    return score


__all__ = ["format_reward", "iou_reward"]
