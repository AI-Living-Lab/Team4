#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions.py
  GDPO 학습에 사용할 리워드 함수.
  VS2의 평가 방식(temporal IoU, 라벨 매칭)을 기반으로 구현.

리워드 구성:
  1. format_reward  — 모델 출력이 유효한 JSON인지 (0 or 1)
  2. label_reward   — 예측 라벨이 GT 라벨과 매칭되는 비율 (0~1)
  3. iou_reward     — temporal IoU 기반 정확도 (0~1)
"""

import json
import re
import string
from typing import List, Dict, Optional

# ============================================================
# 라벨 정규화 (eval_mAP_multi.py와 동일)
# ============================================================

_END_PUNCT = set(list(string.punctuation))
_SEP_RE = re.compile(r"[_\-/]+")
_WS_RE = re.compile(r"\s+")


def normalize_label(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    s = raw.strip()
    if not s:
        return ""
    s = s.lower()
    s = _SEP_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    while s and s[0] in _END_PUNCT:
        s = s[1:].lstrip()
    while s and s[-1] in _END_PUNCT:
        s = s[:-1].rstrip()
    s = _WS_RE.sub(" ", s).strip()
    return s


# ============================================================
# JSON 파싱 (eval_mAP_multi.py의 extract_pred_events와 동일)
# ============================================================

def extract_pred_events(text: str) -> Optional[List[Dict]]:
    """모델 출력에서 이벤트 리스트 추출."""
    if not isinstance(text, str):
        return None

    s = text.replace("<|im_end|>", "").strip()

    # 1) "[...]" 배열 파싱
    m = re.search(r"\[[\s\S]*\]", s)
    if m:
        try:
            arr = json.loads(m.group(0).strip())
            if isinstance(arr, list):
                return arr
        except Exception:
            pass

    # 2) JSON object 연속 파싱
    dec = json.JSONDecoder()
    i, n = 0, len(s)
    out = []
    while i < n:
        while i < n and s[i] in " \t\r\n,":
            i += 1
        if i >= n or s[i] not in "[{":
            break
        try:
            obj, j = dec.raw_decode(s, i)
            i = j
            if isinstance(obj, list):
                out.extend([x for x in obj if isinstance(x, dict)])
            elif isinstance(obj, dict):
                out.append(obj)
        except Exception:
            break

    return out if out else None


# ============================================================
# 핵심 리워드 함수들
# ============================================================

def decode_vtg_time(token_str: str, max_time: float = 60.0) -> float | None:
    """VTG-LLM time token 역변환.
    "<t0><t0><t3><t9><tdot><t0>" → 39.0
    형식: 정수부 4자리 digit + <tdot> + 소수부 1자리 digit → 6토큰 고정
    """
    has_dot = "<tdot>" in token_str

    if has_dot:
        parts = token_str.split("<tdot>")
        int_part = re.findall(r"<t(\d)>", parts[0])
        dec_part = re.findall(r"<t(\d)>", parts[1]) if len(parts) > 1 else []
    else:
        int_part = re.findall(r"<t(\d)>", token_str)
        dec_part = []

    if not int_part:
        return None

    integer_part = int("".join(int_part))
    decimal_part = int(dec_part[0]) if dec_part else 0
    t = integer_part + decimal_part / 10.0
    return min(t, max_time)


def parse_time_value(val) -> float | None:
    """시간 값을 파싱. time token 형식과 일반 float 모두 지원."""
    if val is None:
        return None
    s = str(val)
    # time token 형식 감지
    if "<t" in s:
        return decode_vtg_time(s)
    # 일반 float
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def temporal_iou(ps: float, pe: float, gs: float, ge: float) -> float:
    """두 시간 구간의 IoU 계산."""
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = (pe - ps) + (ge - gs) - inter
    return inter / union if union > 0 else 0.0


def format_reward(completion: str, **kwargs) -> float:
    """모델 출력이 유효한 JSON 이벤트 리스트인지 확인.

    Returns:
        1.0 — time token 형식 사용
        0.5 — JSON 파싱 성공 + 필수 필드 존재 (일반 숫자)
        0.0 — 파싱 실패
    """
    events = extract_pred_events(completion)
    if events is None or len(events) == 0:
        return 0.0

    _time_token_re = re.compile(r"^(<t\d>){1,4}<tdot><t\d>$")
    has_valid_json = False
    has_time_token = False

    for ev in events:
        has_label = isinstance(ev.get("event") or ev.get("label"), str)
        start_str = str(ev.get("start", ""))
        end_str = str(ev.get("end", ""))

        if has_label and start_str and end_str:
            has_valid_json = True

        valid_start = bool(_time_token_re.match(start_str))
        valid_end = bool(_time_token_re.match(end_str))
        if has_label and valid_start and valid_end:
            has_time_token = True

    if has_time_token:
        return 1.0
    elif has_valid_json:
        return 0.5
    return 0.0


def label_reward(completion: str, gt_events: List[Dict], **kwargs) -> float:
    """예측 라벨이 GT 라벨과 얼마나 매칭되는지.

    Args:
        completion: 모델 출력 문자열
        gt_events: [{"label": str, "timestamps": [s, e]}, ...]

    Returns:
        0~1 (매칭된 GT 라벨 비율)
    """
    pred_events = extract_pred_events(completion)
    if not pred_events or not gt_events:
        return 0.0

    gt_labels = {normalize_label(ev["label"]) for ev in gt_events}
    gt_labels.discard("")

    if not gt_labels:
        return 0.0

    pred_labels = set()
    for ev in pred_events:
        raw = ev.get("event") or ev.get("label") or ""
        norm = normalize_label(raw)
        if norm:
            pred_labels.add(norm)

    matched = gt_labels & pred_labels
    if not matched:
        return 0.0
    precision = len(matched) / len(pred_labels)
    recall = len(matched) / len(gt_labels)
    return 2 * precision * recall / (precision + recall)


def iou_reward(completion: str, gt_events: List[Dict], **kwargs) -> float:
    """temporal IoU 기반 정확도 리워드.

    각 GT 이벤트에 대해 가장 높은 IoU를 가진 예측과 매칭 후 평균.

    Args:
        completion: 모델 출력 문자열
        gt_events: [{"label": str, "timestamps": [s, e]}, ...]

    Returns:
        0~1 (평균 best-match IoU)
    """
    pred_events = extract_pred_events(completion)
    if not pred_events or not gt_events:
        return 0.0

    # 예측 파싱 (time token 및 일반 float 모두 지원)
    preds = []
    for ev in pred_events:
        raw_label = ev.get("event") or ev.get("label") or ""
        norm = normalize_label(raw_label)
        start = parse_time_value(ev.get("start"))
        end = parse_time_value(ev.get("end"))
        if start is not None and end is not None and end > start:
            preds.append({"label": norm, "start": start, "end": end})

    if not preds:
        return 0.0

    # 각 GT 이벤트에 대해 best IoU
    ious = []
    for gt in gt_events:
        gt_label = normalize_label(gt["label"])
        gs, ge = float(gt["timestamps"][0]), float(gt["timestamps"][1])

        best_iou = 0.0
        for pred in preds:
            if pred["label"] == gt_label:
                iou = temporal_iou(pred["start"], pred["end"], gs, ge)
                best_iou = max(best_iou, iou)
        ious.append(best_iou)

    return sum(ious) / len(ious) if ious else 0.0


def combined_reward(
    completion: str,
    gt_events: List[Dict],
    weights: Dict[str, float] = None,
    **kwargs,
) -> float:
    """세 가지 리워드의 가중 합.

    Args:
        weights: {"format": 0.1, "label": 0.3, "iou": 0.6} 형태
                 기본값은 포맷 10%, 라벨 30%, IoU 60%
    """
    if weights is None:
        weights = {"format": 0.1, "label": 0.3, "iou": 0.6}

    r_format = format_reward(completion)
    r_label = label_reward(completion, gt_events)
    r_iou = iou_reward(completion, gt_events)

    return (
        weights.get("format", 0.1) * r_format
        + weights.get("label", 0.3) * r_label
        + weights.get("iou", 0.6) * r_iou
    )