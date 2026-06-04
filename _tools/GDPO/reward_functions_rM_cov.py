#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_functions_rM_cov.py  ★골격(SKELETON) — 2026-06-02, eval 결과 보고 확정★

  MUSEG r_M + coverage-excess 페널티 (FP 페널티 대체).

    total = r_M(pred, gt) - GAMMA * coverage_excess(pred, gt)   [+ short_bonus (보류)]

  배경(왜 FP 페널티가 아니라 coverage-excess 인가):
    - FP 페널티(rM_fp)는 "unmatched 세그 *개수*"를 깎음 → "개수 줄여라" → 단일 lumping 강화.
      옛 base mpen 분석에서 R²=0.86 길이편향 극대화 / 멀티세그 포기로 확인됨(FINDINGS).
      라이브 v5b k=0.1 도 멀티 29%→10% 로 같은 붕괴 재현.
    - coverage-excess 는 "GT 대비 덮은 *총 시간*"의 초과분을 깎음 → "통짜로 길게 뱉지 마라".
      길이편향(통짜 hacking)을 *직접* 조준. FP 페널티와 정반대 방향(개수↓가 아니라 길이↓).

  coverage_excess:
    ratio = sum(len(pred_seg)) / max(sum(len(gt_seg)), eps)
    excess = max(0, ratio - COVERAGE_THRESH)     # GT 총길이의 THRESH 배까지는 무벌점
    → 통짜로 영상 전체를 덮으면 ratio 가 크게 튀어 excess 커짐.

  ⚠️ 미확정 / 보류:
    - GAMMA, COVERAGE_THRESH 값은 가설. 작은 sweep 필요(예: γ∈{0.3,0.5}, thr∈{1.3,1.5}).
    - short_bonus(짧은 GT 가산항)는 **이번 k=0.1 run eval 의 짧은 GT R@0.3** 보고 결정.
      R@0.3≈0 이면(=탐지 자체 실패) "맞추면 보너스"는 트리거 안 돼 무효 → 다른 형태 필요.
      유의미하게 올랐으면 가산형 보너스 설계. 그래서 지금은 _short_bonus()=0 placeholder.
    - GRPO advantage 는 group(8 generation) 내 상대값 → coverage 다양성이 있을 때만 신호.
      무너진 정책 말고 cold-start SFT 에서 새로 시작할 것(이 reward 의 전제).

  사용(확정 후): trainer 가 이 모듈을 import 하도록 새 trainer 또는 import 스위치 필요.
    원본 reward_functions.py 는 무수정. (rM_fp 와 동일 패턴)
"""
from typing import List, Tuple
import os

from reward_functions import (
    r_M,
    temporal_iou,
    decode_vtg_time,
    _SEG_CAPTURE_RE,
    format_reward,
)

# ---- 하이퍼파라미터 (env override, 미확정) ----
COVERAGE_GAMMA = float(os.environ.get("COVERAGE_GAMMA", "0.5"))   # excess 1.0 당 감점
COVERAGE_THRESH = float(os.environ.get("COVERAGE_THRESH", "1.5")) # GT 총길이의 N배까지 무벌점
# ★ excess(=ratio-thresh) 상한. outlier(ratio 8~10) 한두 개가 GDPO group 분산을 키워
#   gradient 를 납치하는 hijack 차단용. 기본 inf=cap 없음(=기존 linear, 돌던 g05 런 호환).
#   variance 관점: per-sample 페널티 기여 상한 = group std 폭주 방지. 단 너무 낮으면(예 2.0,
#   ratio 3.5 포화) 초반 다-통짜 구간에서 페널티 평평→분산0→신호죽음 = toothless. ~3.0 권장.
COVERAGE_CAP = float(os.environ.get("COVERAGE_CAP", "inf"))
SHORT_BONUS_ALPHA = float(os.environ.get("SHORT_BONUS_ALPHA", "0.0"))  # 0=비활성(eval 후 결정)
SHORT_THRESH = float(os.environ.get("SHORT_THRESH", "5.0"))
_EPS = 0.1


def _parse_pred_segments(completion: str) -> List[Tuple[float, float]]:
    """rM_fp 와 동일 파싱 (start<end 인 것만)."""
    if not isinstance(completion, str):
        return []
    text = completion.replace("<|im_end|>", "").strip()
    preds: List[Tuple[float, float]] = []
    for start_str, end_str in _SEG_CAPTURE_RE.findall(text):
        s = decode_vtg_time(start_str)
        e = decode_vtg_time(end_str)
        if s is not None and e is not None and e > s:
            preds.append((s, e))
    return preds


def coverage_excess(completion: str,
                    gt_intervals: List[Tuple[float, float]],
                    thresh: float = COVERAGE_THRESH,
                    cap: float = COVERAGE_CAP) -> float:
    """pred 총 커버리지가 GT 총길이의 thresh 배를 넘는 초과분(>=0), cap 으로 상한.

    통짜 hacking(영상 전체 한 구간)을 직접 잡는다. 위치/겹침은 보지 않고 *총 길이*만
    비교(이 항의 역할은 '과대 커버리지' 억제 한 가지; 위치 정확도는 r_M 이 담당).
    cap: excess 상한(기본 inf=무제한). outlier ratio 가 GDPO group 분산을 hijack 못 하게 자름.
    """
    if not gt_intervals:
        return 0.0
    preds = _parse_pred_segments(completion)
    if not preds:
        return 0.0
    total_pred = sum(e - s for s, e in preds)
    total_gt = sum(e - s for s, e in gt_intervals)
    ratio = total_pred / max(total_gt, _EPS)
    return min(cap, max(0.0, ratio - thresh))


def _short_gt_bonus(completion: str,
                    gt_intervals: List[Tuple[float, float]],
                    alpha: float = SHORT_BONUS_ALPHA,
                    short_thresh: float = SHORT_THRESH) -> float:
    """★보류★ 짧은 GT(<short_thresh) 를 IoU>0.3 로 맞추면 +alpha (GT당 1회).

    alpha=0 이면 비활성. 이번 k=0.1 run eval 의 짧은 GT R@0.3 이 0 을 벗어났을 때만
    alpha>0 으로 켤 것(탐지 못 하면 트리거 안 돼 무효 — 옵션 A 의 본질적 약점).
    """
    if alpha <= 0.0 or not gt_intervals:
        return 0.0
    preds = _parse_pred_segments(completion)
    if not preds:
        return 0.0
    bonus = 0.0
    for gs, ge in gt_intervals:
        if (ge - gs) < short_thresh:
            for ps, pe in preds:
                if temporal_iou(ps, pe, gs, ge) > 0.3:
                    bonus += alpha
                    break
    return bonus


def iou_reward(completion: str,
               gt_intervals: List[Tuple[float, float]],
               **kwargs) -> float:
    """MUSEG r_M − GAMMA·coverage_excess  [+ short_bonus(보류, 기본 0)].

    GDPO advantage 는 group 내 정규화이므로 절대 부호/스케일 무관.
    """
    base = r_M(completion, gt_intervals)
    cov_pen = COVERAGE_GAMMA * coverage_excess(completion, gt_intervals)
    bonus = _short_gt_bonus(completion, gt_intervals)   # 기본 alpha=0 → 0
    return base - cov_pen + bonus


__all__ = ["format_reward", "iou_reward", "decode_vtg_time",
           "coverage_excess", "_short_gt_bonus"]
