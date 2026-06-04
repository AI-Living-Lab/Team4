#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_rM_fp_cot.py
  CoT (<think>...</think><answer>...</answer>) 버전 reward.
  base: reward_functions_rM_fp.py (MUSEG r_M + FP penalty) 를 그대로 계승하되,
        completion 이 CoT 포맷이라는 가정 하에 4개 reward 채널을 제공한다.

  완성 포맷 가정 (unav100_v2_cot.json prompt 와 align):
    <think>
    Visual: From <tX>..<tdot><tX> to <tX>..<tdot><tX>. ...
    Audio:  From ... to ... . ...
    Joint:  From ... to ... . ...
    </think><answer>From ... to ... . ...</answer>

  4개 reward (trainer 가 각각 별도 GDPO 채널로 사용):
    1. format_reward    — 기존 strict 토큰 포맷 AND <think>..</think><answer>..</answer> 구조.
                          둘 다 만족해야 1 (구조 깨지면 0, 토큰 흠집이면 부분점수).
    2. iou_reward       — r_M(answer) - K * n_unmatched_pred(answer).
                          **answer 블록만** 파싱 (think 안의 segment 오염 방지).
    3. timestamp_reward — think 의 Joint segment 집합 == answer segment 집합 → 1, 하나라도
                          다르면 0 (집합 동일성).
    4. modality_reward  — think 의 Joint 가 (Audio ∩ Video) 안에 완전히 포함되면 1, 한 조각
                          이라도 교집합 밖이면 0. ("both video and audio" 의미: 교집합 기준)

  주의: 원본 reward_functions.py / reward_functions_rM_fp.py 는 무수정. import 만 한다.
"""
import os
import re
from typing import List, Tuple, Dict

# 원본에서 그대로 재사용 (무수정).
from reward_functions import (
    r_M,                                  # MUSEG Segment Matching reward
    decode_vtg_time,                      # VTG time token 역변환
    _SEG_CAPTURE_RE,                      # "From X to Y" segment 캡쳐 (X,Y 그룹)
    _merge_intervals,                     # interval union merge
    format_reward as _orig_format_reward, # strict 토큰 포맷 점수 [0,1]
)
from reward_functions_rM_fp import (
    n_unmatched_pred,                     # FP 개수 (answer 부분에 대해 호출)
    FP_PENALTY_K,                         # unmatched pred 1개당 감점 (env FP_PENALTY_K)
)

# 교집합/포함 판정 부동소수 허용오차 (timestamp 는 0.1 grid).
_TOL = 1e-3


# ============================================================
# 파싱 유틸
# ============================================================

def _strip(completion: str) -> str:
    if not isinstance(completion, str):
        return ""
    return completion.replace("<|im_end|>", "").strip()


def _extract_block(text: str, tag: str) -> str | None:
    """<tag>...</tag> 안쪽 텍스트. 없으면 None. (첫 매치)"""
    m = re.search(r"<" + tag + r">(.*?)</" + tag + r">", text, re.DOTALL)
    return m.group(1) if m else None


def _parse_segments_from(text: str) -> List[Tuple[float, float]]:
    """주어진 텍스트 안의 'From X to Y' segment 들을 (start, end) 로 (start<end 만)."""
    out: List[Tuple[float, float]] = []
    if not text:
        return out
    for s_str, e_str in _SEG_CAPTURE_RE.findall(text):
        s = decode_vtg_time(s_str)
        e = decode_vtg_time(e_str)
        if s is not None and e is not None and e > s:
            out.append((s, e))
    return out


# think 내부 라벨: 각 줄 시작이 키워드여야 매치 (joint 줄의 "video+audio" 서술에
# 오인되지 않도록 line-start anchor).
_LABEL_PATS = [
    ("visual", re.compile(r"(?im)^[^\S\n]*(?:visual|video)[^\n:]*:")),
    ("audio",  re.compile(r"(?im)^[^\S\n]*audio[^\n:]*:")),
    ("joint",  re.compile(r"(?im)^[^\S\n]*(?:joint|both)[^\n:]*:")),
]


def _parse_think_sections(completion: str) -> Dict[str, List[Tuple[float, float]]]:
    """think 블록을 Visual / Audio / Joint 3개 라벨 구간으로 쪼개 각 segment list 반환.

    라벨을 못 찾은 채널은 빈 list. think 자체가 없으면 모두 빈 list.
    """
    result = {"visual": [], "audio": [], "joint": []}
    text = _strip(completion)
    think = _extract_block(text, "think")
    if think is None:
        return result

    found = []
    for name, pat in _LABEL_PATS:
        m = pat.search(think)
        if m:
            found.append((m.start(), m.end(), name))
    found.sort()

    for i, (st, en, name) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(think)
        result[name] = _parse_segments_from(think[en:end])
    return result


# ============================================================
# interval 집합 연산 (modality reward 용)
# ============================================================

def _intersect_lists(a: List[Tuple[float, float]],
                     b: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """두 interval list 의 교집합 region (merged)."""
    am, bm = _merge_intervals(a), _merge_intervals(b)
    out = []
    for as_, ae in am:
        for bs, be in bm:
            lo, hi = max(as_, bs), min(ae, be)
            if hi - lo > _TOL:
                out.append((lo, hi))
    return _merge_intervals(out)


def _uncovered_length(segs: List[Tuple[float, float]],
                      region: List[Tuple[float, float]]) -> float:
    """segs 중 region 에 덮이지 않은 총 길이."""
    region = _merge_intervals(region)
    total = 0.0
    for s, e in segs:
        covered = 0.0
        for rs, re_ in region:
            lo, hi = max(s, rs), min(e, re_)
            if hi > lo:
                covered += hi - lo
        total += max(0.0, (e - s) - covered)
    return total


def _seg_set(segs: List[Tuple[float, float]]):
    """집합 비교용 정규화 (0.1 grid 로 반올림)."""
    return {(round(s, 1), round(e, 1)) for s, e in segs}


# ============================================================
# 1. format_reward — 기존 토큰 포맷 AND CoT 구조
# ============================================================

def _structure_ok(completion: str) -> bool:
    """<think>...</think><answer>...</answer> 구조 + answer 에 유효 segment 1개 이상."""
    t = _strip(completion)
    if not t:
        return False
    m = re.fullmatch(r"\s*<think>.*?</think>\s*<answer>(.*?)</answer>\s*",
                     t, re.DOTALL)
    if not m:
        return False
    return bool(_SEG_CAPTURE_RE.search(m.group(1)))


def format_reward(completion: str, **kwargs) -> float:
    """기존 strict 토큰 포맷 점수 AND CoT 구조 만족 시에만 1.

        format = orig_token_format([0,1]) * structure(0 or 1)

    → 구조가 깨지면 0, 구조 OK 인데 토큰 흠집이면 부분점수, 둘 다 완벽하면 1.
    """
    struct = 1.0 if _structure_ok(completion) else 0.0
    if struct == 0.0:
        return 0.0
    return _orig_format_reward(completion) * struct


# ============================================================
# 2. iou_reward — r_M + FP penalty (answer 블록만)
# ============================================================

def iou_reward(completion: str,
               gt_intervals: List[Tuple[float, float]],
               **kwargs) -> float:
    """answer 블록에 대해 r_M - K * n_unmatched_pred.

    think 안의 segment 가 r_M / FP 계산을 오염시키지 않도록 answer 만 떼어 채점.
    answer 블록이 없으면 r_M("",gt)=0, penalty=0 → 0.
    """
    if gt_intervals is None:
        gt_intervals = []
    ans = _extract_block(_strip(completion), "answer") or ""
    base = r_M(ans, gt_intervals)
    penalty = FP_PENALTY_K * n_unmatched_pred(ans, gt_intervals)
    return base - penalty


# ============================================================
# 3. timestamp_reward — think.Joint 집합 == answer 집합
# ============================================================

def timestamp_reward(completion: str,
                     gt_intervals: List[Tuple[float, float]] = None,
                     **kwargs) -> float:
    """think 의 Joint segment 집합이 answer segment 집합과 완전히 동일하면 1, 아니면 0.

    하나라도 다르면(개수/값) 0. 둘 중 하나라도 비면 0.
    GT 와 무관 — think↔answer 내부 일관성 reward.
    """
    sections = _parse_think_sections(completion)
    joint = sections["joint"]
    ans = _parse_segments_from(_extract_block(_strip(completion), "answer") or "")
    if not ans or not joint:
        return 0.0
    return 1.0 if _seg_set(joint) == _seg_set(ans) else 0.0


# ============================================================
# 4. modality_reward — think.Joint ⊆ (Audio ∩ Video)
# ============================================================

def modality_reward(completion: str,
                    gt_intervals: List[Tuple[float, float]] = None,
                    **kwargs) -> float:
    """think 의 Joint 구간이 (Audio ∩ Video) 교집합 안에 완전히 포함되면 1, 아니면 0.

    Joint 의 어느 한 조각이라도 (Audio ∩ Video) 밖이면 0.
    Audio/Video 중 하나라도 비면 교집합이 비어 Joint(>0 길이)는 밖 → 0.
    Joint 가 비면 검증 불가(malformed) → 0.
    """
    sections = _parse_think_sections(completion)
    joint = sections["joint"]
    audio = sections["audio"]
    video = sections["visual"]
    if not joint:
        return 0.0
    inter = _intersect_lists(audio, video)
    return 0.0 if _uncovered_length(joint, inter) > _TOL else 1.0


__all__ = [
    "format_reward",
    "iou_reward",
    "timestamp_reward",
    "modality_reward",
    "decode_vtg_time",
    "n_unmatched_pred",
]
