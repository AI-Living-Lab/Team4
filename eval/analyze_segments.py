#!/usr/bin/env python3
"""
analyze_segments.py
  test_results.json (pred) + chunk JSON (gt_segments) → segment-distribution report.

  Reports per model:
    - Pred segment count distribution (mean, median, count==0/1/2/3+ buckets)
    - Multi-segment ratio (samples with >=2 pred segments)
    - Average pred segment count per sample
    - Pred-vs-GT segment count comparison (mean diff, % under-seg / over-seg / equal)
    - Total covered time ratio: sum(pred len) / sum(gt len) per sample
    - Per-segment length stats (mean / median pred segment seconds)
    - FP rate: fraction of pred segments whose best-matching GT IoU < 0.3
    - Parse-failure rate

Usage:
  python analyze_segments.py --results <test_results.json> --test_json <chunk.json> \
                             --max_time 110 --out <out.json> [--label F1|rM]
"""
import argparse
import json
import os
import re
import statistics
from collections import Counter


def decode_vtg_time(token_str, max_time=60.0):
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
    return min(integer_part + decimal_part / 10.0, max_time)


SEG_RE = re.compile(
    r"[Ff]rom\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)\s+to\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)"
)


def parse_multi_segments(raw, max_time):
    out = []
    for m in SEG_RE.finditer(raw or ""):
        s = decode_vtg_time(m.group(1), max_time)
        e = decode_vtg_time(m.group(2), max_time)
        if s is None or e is None:
            continue
        if e <= s:
            e = min(s + 1.0, max_time)
        out.append([s, e])
    return out


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


# segment-level F1 (MUSEG/E.T.Bench): pred↔GT 1:1 greedy 매칭, FN+FP 둘 다 벌.
F1_THRS = [0.1, 0.3, 0.5, 0.7]


def f1_at(preds, gts, tau):
    if not preds or not gts:
        return 0.0 if (preds or gts) else 1.0
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)),
                   reverse=True)
    up, ug, tp = set(), set(), 0
    for ov, i, j in pairs:
        if ov < tau or i in up or j in ug:
            continue
        up.add(i); ug.add(j); tp += 1
    fp, fn = len(preds) - tp, len(gts) - tp
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0


def sample_f1(preds, gts):
    return statistics.mean(f1_at(preds, gts, t) for t in F1_THRS)


def safe_mean(xs):
    return statistics.mean(xs) if xs else 0.0


def safe_median(xs):
    return statistics.median(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--max_time", type=float, default=110.0)
    ap.add_argument("--fp_iou_thr", type=float, default=0.3,
                    help="A pred segment is FP if its best-matching GT IoU < this.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="model")
    args = ap.parse_args()

    with open(args.results, "r") as f:
        results = json.load(f)
    with open(args.test_json, "r") as f:
        test_data = json.load(f)

    if len(results) != len(test_data):
        print(f"[WARN] {args.label}: results({len(results)}) != test({len(test_data)}); "
              f"truncating to min.")
    n = min(len(results), len(test_data))

    pred_seg_counts = []
    gt_seg_counts = []
    pred_len_total = []
    gt_len_total = []
    pred_seg_lens_flat = []
    parse_fail = 0
    fp_count_per_sample = []
    tp_count_per_sample = []
    cnt_buckets = Counter()  # 0 / 1 / 2 / 3 / 4+ pred segments
    under_seg = 0  # n_pred < n_gt
    over_seg = 0
    equal_seg = 0
    raw_pred_lens_chars = []
    f1_all = []      # segment-level F1 (avg over thresholds), per sample
    f1_single = []   # GT가 단일세그인 sample
    f1_multi = []    # GT가 멀티세그인 sample
    f1_by_thr = {t: [] for t in F1_THRS}

    for i in range(n):
        res = results[i]
        gt = test_data[i]
        pred_text = res.get("pred", "")
        raw_pred_lens_chars.append(len(pred_text or ""))
        gt_segs = [list(map(float, s)) for s in gt.get("gt_segments", [])]
        pred_segs = parse_multi_segments(pred_text, max_time=args.max_time)

        if not pred_segs:
            parse_fail += 1

        pred_seg_counts.append(len(pred_segs))
        gt_seg_counts.append(len(gt_segs))

        bk = min(len(pred_segs), 4)
        cnt_buckets[bk] += 1

        if len(pred_segs) < len(gt_segs):
            under_seg += 1
        elif len(pred_segs) > len(gt_segs):
            over_seg += 1
        else:
            equal_seg += 1

        pred_len_total.append(sum(max(0.0, e - s) for s, e in pred_segs))
        gt_len_total.append(sum(max(0.0, e - s) for s, e in gt_segs))
        pred_seg_lens_flat.extend(max(0.0, e - s) for s, e in pred_segs)

        # FP: pred segment whose best IoU against any GT < threshold
        n_fp = 0
        n_tp = 0
        for ps in pred_segs:
            if not gt_segs:
                n_fp += 1
                continue
            best = max(iou(ps, gs) for gs in gt_segs)
            if best < args.fp_iou_thr:
                n_fp += 1
            else:
                n_tp += 1
        fp_count_per_sample.append(n_fp)
        tp_count_per_sample.append(n_tp)

        # segment-level F1 (MUSEG 방식)
        sf = sample_f1(pred_segs, gt_segs)
        f1_all.append(sf)
        (f1_multi if len(gt_segs) >= 2 else f1_single).append(sf)
        for t in F1_THRS:
            f1_by_thr[t].append(f1_at(pred_segs, gt_segs, t))

    multi_seg_samples = sum(1 for c in pred_seg_counts if c >= 2)
    total_pred_segs = sum(pred_seg_counts)
    total_gt_segs = sum(gt_seg_counts)
    total_fp = sum(fp_count_per_sample)
    total_tp = sum(tp_count_per_sample)

    length_ratio_per_sample = []
    for p, g in zip(pred_len_total, gt_len_total):
        if g > 0:
            length_ratio_per_sample.append(p / g)

    report = {
        "label": args.label,
        "n_samples": n,
        "parse_fail": parse_fail,
        "parse_fail_pct": round(100 * parse_fail / max(n, 1), 2),

        "pred_segments": {
            "total": total_pred_segs,
            "mean_per_sample": round(safe_mean(pred_seg_counts), 3),
            "median_per_sample": safe_median(pred_seg_counts),
            "count_buckets_pct": {
                "0": round(100 * cnt_buckets[0] / max(n, 1), 2),
                "1": round(100 * cnt_buckets[1] / max(n, 1), 2),
                "2": round(100 * cnt_buckets[2] / max(n, 1), 2),
                "3": round(100 * cnt_buckets[3] / max(n, 1), 2),
                "4+": round(100 * cnt_buckets[4] / max(n, 1), 2),
            },
            "multi_seg_pct": round(100 * multi_seg_samples / max(n, 1), 2),
            "mean_seg_len_sec": round(safe_mean(pred_seg_lens_flat), 3),
            "median_seg_len_sec": round(safe_median(pred_seg_lens_flat), 3),
        },

        "gt_segments": {
            "total": total_gt_segs,
            "mean_per_sample": round(safe_mean(gt_seg_counts), 3),
            "median_per_sample": safe_median(gt_seg_counts),
        },

        "pred_vs_gt_count": {
            "under_seg_pct": round(100 * under_seg / max(n, 1), 2),
            "equal_seg_pct": round(100 * equal_seg / max(n, 1), 2),
            "over_seg_pct":  round(100 * over_seg  / max(n, 1), 2),
            "mean_pred_minus_gt": round(safe_mean(
                [p - g for p, g in zip(pred_seg_counts, gt_seg_counts)]
            ), 3),
        },

        "covered_time_ratio_pred_over_gt": {
            "mean": round(safe_mean(length_ratio_per_sample), 3),
            "median": round(safe_median(length_ratio_per_sample), 3),
            "n_with_gt": len(length_ratio_per_sample),
        },

        "fp_analysis": {
            "fp_iou_threshold": args.fp_iou_thr,
            "total_fp_segments": total_fp,
            "total_tp_segments": total_tp,
            "fp_rate_per_pred_seg": round(
                total_fp / max(total_pred_segs, 1), 4
            ),
            "precision_pred_seg": round(
                total_tp / max(total_pred_segs, 1), 4
            ),
            "mean_fp_per_sample": round(safe_mean(fp_count_per_sample), 3),
        },

        "raw_pred_text_len_chars": {
            "mean": round(safe_mean(raw_pred_lens_chars), 2),
            "median": safe_median(raw_pred_lens_chars),
        },

        # segment-level F1 (MUSEG/E.T.Bench 방식, τ=0.1/0.3/0.5/0.7 평균). 멀티세그 목표 핵심 지표.
        "f1_multiseg": {
            "thresholds": F1_THRS,
            "f1_all_pct": round(100 * safe_mean(f1_all), 2),
            "f1_single_pct": round(100 * safe_mean(f1_single), 2),
            "f1_multi_pct": round(100 * safe_mean(f1_multi), 2),
            "n_single": len(f1_single),
            "n_multi": len(f1_multi),
            "f1_by_threshold_pct": {str(t): round(100 * safe_mean(f1_by_thr[t]), 2) for t in F1_THRS},
        },
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {args.out}")


if __name__ == "__main__":
    main()
