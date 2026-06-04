#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reeval_from_results.py — 이미 있는 test_results_rank0.json(추론 결과) 만으로
Team4 평가 3종 산출물(eval_miou_summary / segment_analysis / gtlen_analysis)을
재생성한다. GPU 불필요(추론 재실행 X, CPU 파싱·집계만).

outputs/base 의 base 모델들은 출력 포맷이 제각각이라(아크/아비쿠나/크로너스/
뮤즈/살몬), pred_parsers.get_parser(pred_format) 로 포맷별 파서를 갈아끼우고,
나머지 mIoU(union)·F1·GT길이 분해 로직은 Team4 원본 스크립트 함수
(eval_miou_multiseg / analyze_segments / lib_tvg)를 그대로 import 해 수치 일치를 보장.

GT 출처(gt_source):
  embedded : results 각 항목의 gt_segments 사용 (arc/avicuna/chronus/museg)
  ref      : results 각 항목의 ref(토큰 GT) 파싱 (salmonn)

사용:
  python reeval_from_results.py --results <dir>/test_results_rank0.json \
      --pred_format museg --gt_source embedded --label MUSEG --out_dir <dir>
"""
import argparse
import json
import os
import statistics
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval_miou_multiseg as M   # compute_union_iou 등
import analyze_segments as A     # iou, f1_at, sample_f1 등
import lib_tvg as L              # tiou, best_ious_for_sample
from pred_parsers import get_parser, parse_tokens


# -------------------------------------------------------------- 공통: 샘플 추출
def load_samples(results, pred_format, gt_source, duration_key):
    """results → [(gt_segs, pred_segs)] (둘 다 [[s,e]] 초). 파싱은 포맷별 파서."""
    parse_pred = get_parser(pred_format)
    samples = []
    for r in results:
        dur = r.get(duration_key)
        pred_segs = parse_pred(r.get("pred", ""), dur)
        if gt_source == "embedded":
            gt = [[float(s), float(e)] for s, e in (r.get("gt_segments") or [])]
        elif gt_source == "ref":
            gt = parse_tokens(r.get("ref", ""))
        else:
            raise ValueError(f"bad gt_source {gt_source}")
        samples.append((gt, pred_segs))
    return samples


# -------------------------------------------------------------- 1) eval_miou
def build_miou(samples, n_results):
    """eval_miou_multiseg.main 과 동일 union-IoU 집계 (파싱만 외부 주입)."""
    all_ious, parse_ok, parse_fail = [], 0, 0
    total_preds, fp_preds = 0, 0
    for gt_segments, pred_segments in samples:
        if not pred_segments:
            all_ious.extend(0.0 for _ in gt_segments)
            parse_fail += 1
            continue
        parse_ok += 1
        for p in pred_segments:
            total_preds += 1
            if not any(min(p[1], g[1]) > max(p[0], g[0]) for g in gt_segments):
                fp_preds += 1
        for gt_seg in gt_segments:
            all_ious.append(M.compute_union_iou(gt_seg, pred_segments))

    arr = np.array(all_ious)
    n = len(arr)
    miou = float(np.mean(arr)) if n else 0.0
    recall = {th: (float(np.mean(arr >= th)) if n else 0.0) for th in (0.3, 0.5, 0.7)}
    fp_rate = fp_preds / total_preds if total_preds else 0.0
    fn_gt = int(np.sum(arr == 0.0)) if n else 0
    fn_rate = fn_gt / n if n else 0.0
    return {
        "mIoU_union_%": round(miou * 100, 4),
        "Recall": {str(k): round(v * 100, 4) for k, v in recall.items()},
        "FP_rate_%": round(fp_rate * 100, 4),
        "FN_rate_%": round(fn_rate * 100, 4),
        "n_pred_segments": total_preds,
        "n_fp_segments": fp_preds,
        "n_gt_segments": n,
        "n_fn_gt_segments": fn_gt,
        "n_samples": n_results,
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
    }


# -------------------------------------------------------------- 2) segment 분석
def build_segment(samples, label, fp_iou_thr=0.3):
    """analyze_segments.main 과 동일 리포트 (파싱만 외부 주입)."""
    n = len(samples)
    pred_seg_counts, gt_seg_counts = [], []
    pred_len_total, gt_len_total, pred_seg_lens_flat = [], [], []
    parse_fail = 0
    fp_per, tp_per = [], []
    cnt_buckets = Counter()
    under_seg = over_seg = equal_seg = 0
    f1_all, f1_single, f1_multi = [], [], []
    f1_by_thr = {t: [] for t in A.F1_THRS}

    for gt_segs, pred_segs in samples:
        if not pred_segs:
            parse_fail += 1
        pred_seg_counts.append(len(pred_segs))
        gt_seg_counts.append(len(gt_segs))
        cnt_buckets[min(len(pred_segs), 4)] += 1
        if len(pred_segs) < len(gt_segs):
            under_seg += 1
        elif len(pred_segs) > len(gt_segs):
            over_seg += 1
        else:
            equal_seg += 1
        pred_len_total.append(sum(max(0.0, e - s) for s, e in pred_segs))
        gt_len_total.append(sum(max(0.0, e - s) for s, e in gt_segs))
        pred_seg_lens_flat.extend(max(0.0, e - s) for s, e in pred_segs)

        n_fp = n_tp = 0
        for ps in pred_segs:
            if not gt_segs:
                n_fp += 1
                continue
            best = max(A.iou(ps, gs) for gs in gt_segs)
            if best < fp_iou_thr:
                n_fp += 1
            else:
                n_tp += 1
        fp_per.append(n_fp)
        tp_per.append(n_tp)

        sf = A.sample_f1(pred_segs, gt_segs)
        f1_all.append(sf)
        (f1_multi if len(gt_segs) >= 2 else f1_single).append(sf)
        for t in A.F1_THRS:
            f1_by_thr[t].append(A.f1_at(pred_segs, gt_segs, t))

    sm, smd = A.safe_mean, A.safe_median
    multi_seg = sum(1 for c in pred_seg_counts if c >= 2)
    total_pred = sum(pred_seg_counts)
    total_gt = sum(gt_seg_counts)
    total_fp, total_tp = sum(fp_per), sum(tp_per)
    ratio = [p / g for p, g in zip(pred_len_total, gt_len_total) if g > 0]

    return {
        "label": label,
        "n_samples": n,
        "parse_fail": parse_fail,
        "parse_fail_pct": round(100 * parse_fail / max(n, 1), 2),
        "pred_segments": {
            "total": total_pred,
            "mean_per_sample": round(sm(pred_seg_counts), 3),
            "median_per_sample": smd(pred_seg_counts),
            "count_buckets_pct": {
                "0": round(100 * cnt_buckets[0] / max(n, 1), 2),
                "1": round(100 * cnt_buckets[1] / max(n, 1), 2),
                "2": round(100 * cnt_buckets[2] / max(n, 1), 2),
                "3": round(100 * cnt_buckets[3] / max(n, 1), 2),
                "4+": round(100 * cnt_buckets[4] / max(n, 1), 2),
            },
            "multi_seg_pct": round(100 * multi_seg / max(n, 1), 2),
            "mean_seg_len_sec": round(sm(pred_seg_lens_flat), 3),
            "median_seg_len_sec": round(smd(pred_seg_lens_flat), 3),
        },
        "gt_segments": {
            "total": total_gt,
            "mean_per_sample": round(sm(gt_seg_counts), 3),
            "median_per_sample": smd(gt_seg_counts),
        },
        "pred_vs_gt_count": {
            "under_seg_pct": round(100 * under_seg / max(n, 1), 2),
            "equal_seg_pct": round(100 * equal_seg / max(n, 1), 2),
            "over_seg_pct": round(100 * over_seg / max(n, 1), 2),
            "mean_pred_minus_gt": round(
                sm([p - g for p, g in zip(pred_seg_counts, gt_seg_counts)]), 3
            ),
        },
        "covered_time_ratio_pred_over_gt": {
            "mean": round(sm(ratio), 3),
            "median": round(smd(ratio), 3),
            "n_with_gt": len(ratio),
        },
        "fp_analysis": {
            "fp_iou_threshold": fp_iou_thr,
            "total_fp_segments": total_fp,
            "total_tp_segments": total_tp,
            "fp_rate_per_pred_seg": round(total_fp / max(total_pred, 1), 4),
            "precision_pred_seg": round(total_tp / max(total_pred, 1), 4),
            "mean_fp_per_sample": round(sm(fp_per), 3),
        },
        "f1_multiseg": {
            "thresholds": A.F1_THRS,
            "f1_all_pct": round(100 * sm(f1_all), 2),
            "f1_single_pct": round(100 * sm(f1_single), 2),
            "f1_multi_pct": round(100 * sm(f1_multi), 2),
            "n_single": len(f1_single),
            "n_multi": len(f1_multi),
            "f1_by_threshold_pct": {
                str(t): round(100 * sm(f1_by_thr[t]), 2) for t in A.F1_THRS
            },
        },
    }


# -------------------------------------------------------------- 3) gtlen 분해
GTLEN_BUCKETS = [(0, 2, "<2s"), (2, 5, "2-5s"), (5, 10, "5-10s"),
                 (10, 20, "10-20s"), (20, 1e9, "20+s")]


def _bucket_of(gtlen):
    for lo, hi, name in GTLEN_BUCKETS:
        if lo <= gtlen < hi:
            return name
    return "20+s"


def build_gtlen(samples, label, n_results):
    """posteval_gtlen.main 과 동일 (matching 대신 이미 정렬된 samples 사용)."""
    buck = {name: {"n": 0, "iou_sum": 0.0, "ge3": 0, "ge5": 0}
            for *_, name in GTLEN_BUCKETS}
    n_single = n_multi = 0
    iou_single = iou_multi = 0.0
    pred_nseg_hist = {}
    pred_nseg_sum = gt_nseg_sum = n_sample = n_decimal = 0

    for gt_raw, pred_segs in samples:
        gt_segs = [(float(s), float(e)) for s, e in gt_raw]
        pred_segs = [(float(s), float(e)) for s, e in pred_segs]
        n_sample += 1
        n_gt = len(gt_segs)
        gt_nseg_sum += n_gt
        np_ = len(pred_segs)
        pred_nseg_sum += np_
        pred_nseg_hist[np_] = pred_nseg_hist.get(np_, 0) + 1
        if any(abs((x * 10) % 10) > 0.01 for s, e in pred_segs for x in (s, e)):
            n_decimal += 1
        bests = L.best_ious_for_sample(gt_segs, pred_segs)
        sample_miou = sum(bests) / len(bests) if bests else 0.0
        if n_gt >= 2:
            n_multi += 1
            iou_multi += sample_miou
        else:
            n_single += 1
            iou_single += sample_miou
        for (s, e), iou in zip(gt_segs, bests):
            b = buck[_bucket_of(e - s)]
            b["n"] += 1
            b["iou_sum"] += iou
            b["ge3"] += (iou >= 0.3)
            b["ge5"] += (iou >= 0.5)

    def pct(x, d):
        return round(100.0 * x / d, 1) if d else 0.0

    return {
        "label": label, "n_results": n_results, "n_matched": n_sample,
        "gtlen_buckets": {name: {
            "n": buck[name]["n"],
            "mIoU": round(100 * buck[name]["iou_sum"] / buck[name]["n"], 1) if buck[name]["n"] else 0.0,
            "R@0.3": pct(buck[name]["ge3"], buck[name]["n"]),
            "R@0.5": pct(buck[name]["ge5"], buck[name]["n"]),
        } for *_, name in GTLEN_BUCKETS},
        "single": {"n": n_single, "mIoU": round(100 * iou_single / n_single, 1) if n_single else 0.0},
        "multi": {"n": n_multi, "mIoU": round(100 * iou_multi / n_multi, 1) if n_multi else 0.0},
        "pred_nseg_mean": round(pred_nseg_sum / n_sample, 2) if n_sample else 0.0,
        "gt_nseg_mean": round(gt_nseg_sum / n_sample, 2) if n_sample else 0.0,
        "pred_nseg_hist": dict(sorted(pred_nseg_hist.items())),
        "decimal_usage_pct": pct(n_decimal, n_sample),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--pred_format", required=True,
                    choices=["arc_hunyuan", "avicuna", "chronusomni", "museg",
                             "salmonn", "tokens"])
    ap.add_argument("--gt_source", default="embedded",
                    choices=["embedded", "ref"])
    ap.add_argument("--duration_key", default="duration")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--label", default="model")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.results) as f:
        results = json.load(f)
    samples = load_samples(results, args.pred_format, args.gt_source,
                           args.duration_key)

    miou = build_miou(samples, len(results))
    seg = build_segment(samples, args.label)
    gtlen = build_gtlen(samples, args.label, len(results))

    for name, obj in [("eval_miou_summary.json", miou),
                      ("segment_analysis.json", seg),
                      ("gtlen_analysis.json", gtlen)]:
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {path}")

    print(f"\n[{args.label}]  pred_format={args.pred_format} gt={args.gt_source}")
    print(f"  mIoU(union) = {miou['mIoU_union_%']}%   "
          f"R@0.3/0.5/0.7 = {miou['Recall']['0.3']}/{miou['Recall']['0.5']}/{miou['Recall']['0.7']}")
    print(f"  parse_ok/fail = {miou['parse_ok']}/{miou['parse_fail']}   "
          f"pred/GT seg = {miou['n_pred_segments']}/{miou['n_gt_segments']}   "
          f"FP/FN% = {miou['FP_rate_%']}/{miou['FN_rate_%']}")
    print(f"  multi_seg_pct = {seg['pred_segments']['multi_seg_pct']}   "
          f"f1_multi = {seg['f1_multiseg']['f1_multi_pct']}")


if __name__ == "__main__":
    main()
