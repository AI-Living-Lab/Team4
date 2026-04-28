"""Shared eval helpers — Union-IoU + FP_rate + FN_rate (공통 라이브러리).

cdh 브랜치 eval/eval_miou_multiseg.py (origin/cdh 최신 f7920fd) 의 Union-IoU + FP_rate 로직 포팅 +
FN_rate 추가.

모든 base-model baseline eval 스크립트 (chronusomni, crab_plus, arc_hunyuan, avicuna)
에서 동일하게 import 해 쓰도록 통일.

주요 지표:
  mIoU_union_%       — 각 GT 마다 (겹치는 pred 들의 합집합) 과의 IoU 평균
  Recall_%@{0.3,0.5,0.7}
  FP_rate_%          — 어떤 GT 와도 안 겹치는 pred 의 비율 (over-prediction)
  FN_rate_%          — 어떤 pred 와도 안 겹치는 GT 의 비율 (under-prediction)
"""

from typing import List, Tuple, Dict, Any
import numpy as np


# ---------- interval helpers ----------
def merge_intervals(intervals):
    """Sort & merge overlapping 1D intervals → list of [start, end]."""
    if not intervals:
        return []
    s = sorted([list(x) for x in intervals])
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def intervals_total_len(intervals):
    return sum(max(0.0, b - a) for a, b in intervals)


def intervals_intersect(a_list, b_list):
    """두 병합된 구간 리스트의 교집합."""
    out, i, j = [], 0, 0
    while i < len(a_list) and j < len(b_list):
        s = max(a_list[i][0], b_list[j][0])
        e = min(a_list[i][1], b_list[j][1])
        if e > s:
            out.append([s, e])
        if a_list[i][1] < b_list[j][1]:
            i += 1
        else:
            j += 1
    return out


def compute_union_iou(gt_seg, pred_segs):
    """Union-IoU: GT 한 개 vs 그와 겹치는 pred 들의 합집합 간 IoU."""
    gs, ge = float(gt_seg[0]), float(gt_seg[1])
    overlapping = [
        p for p in pred_segs
        if min(float(p[1]), ge) > max(float(p[0]), gs)
    ]
    if not overlapping:
        return 0.0
    U = merge_intervals(overlapping)
    G = [[gs, ge]]
    inter_len = intervals_total_len(intervals_intersect(G, U))
    union_len = intervals_total_len(merge_intervals(G + U))
    return inter_len / union_len if union_len > 0 else 0.0


def overlaps_any(seg, seg_list) -> bool:
    """seg ([s,e]) 이 seg_list 중 하나라도 positive overlap 하는가."""
    s, e = float(seg[0]), float(seg[1])
    for q in seg_list:
        if min(float(q[1]), e) > max(float(q[0]), s):
            return True
    return False


# ---------- batch aggregator ----------
def score_sample(gt_segments, pred_segments):
    """한 sample 의 (per-GT Union-IoU 리스트, FP 수, pred 총 수, FN 수, GT 총 수) 리턴.

    pred_segments 가 비어있으면 모든 GT 는 IoU 0, FN 전부, FP 0, pred 0.
    """
    gt = [[float(s[0]), float(s[1])] for s in gt_segments]
    pred = [[float(s[0]), float(s[1])] for s in pred_segments]

    per_gt_iou = []
    for g in gt:
        per_gt_iou.append(compute_union_iou(g, pred))

    n_fp = 0
    for p in pred:
        if not overlaps_any(p, gt):
            n_fp += 1
    n_fn = 0
    for g in gt:
        if not overlaps_any(g, pred):
            n_fn += 1
    return per_gt_iou, n_fp, len(pred), n_fn, len(gt)


def summarize(all_ious, total_preds, fp_preds, total_gts, fn_gts, parse_ok, parse_fail,
              n_samples, thresholds=(0.3, 0.5, 0.7)):
    """최종 메트릭 dict."""
    arr = np.array(all_ious) if all_ious else np.zeros(0)
    n = len(arr)
    miou = float(arr.mean()) if n > 0 else 0.0
    recall = {th: float((arr >= th).mean()) if n > 0 else 0.0 for th in thresholds}
    fp_rate = fp_preds / total_preds if total_preds > 0 else 0.0
    fn_rate = fn_gts / total_gts if total_gts > 0 else 0.0
    fsr = parse_ok / max(n_samples, 1)
    return {
        "mIoU_union_%": round(miou * 100, 4),
        "Recall_%": {str(th): round(v * 100, 4) for th, v in recall.items()},
        "FP_rate_%": round(fp_rate * 100, 4),
        "FN_rate_%": round(fn_rate * 100, 4),
        "FSR_%": round(fsr * 100, 4),
        "n_samples": n_samples,
        "n_gt_segments": total_gts,
        "n_pred_segments": total_preds,
        "n_fp_segments": fp_preds,
        "n_fn_gt_segments": fn_gts,
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
    }


def print_report(title: str, summary: Dict[str, Any]):
    SEP = "=" * 56
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)
    print(f"  n_samples:        {summary['n_samples']}")
    print(f"  GT segments:      {summary['n_gt_segments']}")
    print(f"  Pred segments:    {summary['n_pred_segments']}")
    print(f"  parse_ok:         {summary['parse_ok']}  ({summary['FSR_%']:.2f}% FSR)")
    print(f"  parse_fail:       {summary['parse_fail']}")
    print(f"  mIoU(union):      {summary['mIoU_union_%']:.2f}%")
    for th, v in summary["Recall_%"].items():
        print(f"  Recall @ IoU={th}: {v:.2f}%")
    print(f"  FP_rate:          {summary['FP_rate_%']:.2f}%  "
          f"({summary['n_fp_segments']}/{summary['n_pred_segments']})")
    print(f"  FN_rate:          {summary['FN_rate_%']:.2f}%  "
          f"({summary['n_fn_gt_segments']}/{summary['n_gt_segments']})")
    print(SEP)
