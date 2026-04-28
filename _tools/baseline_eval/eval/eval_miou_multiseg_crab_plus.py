"""Crab+ × UnAV-100 eval — Union-IoU + FP_rate + FN_rate.

Parser: `<event>{label}, (s e)(s e)...</event>`. fuzzy label 일치 체크. 10s cap breakdown 포함.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import score_sample, summarize, print_report

EVENT_RE = re.compile(r"<event>\s*([^,<]+?)\s*,\s*((?:\(\d+\s+\d+\))+)\s*</event>", re.IGNORECASE)
PAIR_RE = re.compile(r"\((\d+)\s+(\d+)\)")


def normalize_label(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).replace("-", " ").replace("_", " ")


def fuzzy_match(pred_label: str, query_label: str) -> bool:
    p, q = normalize_label(pred_label), normalize_label(query_label)
    if p == q:
        return True
    p_toks, q_toks = set(p.split()), set(q.split())
    if p_toks & q_toks:
        min_side = min(len(p_toks), len(q_toks))
        return len(p_toks & q_toks) / max(min_side, 1) >= 0.5
    return False


def parse_segments(raw: str, query_label: str, max_time: float = 60.0):
    out = []
    for m in EVENT_RE.finditer(raw):
        lab = m.group(1)
        if not fuzzy_match(lab, query_label):
            continue
        for a, b in PAIR_RE.findall(m.group(2)):
            s, e = float(a), float(b)
            if e < s:
                s, e = e, s
            if e <= s:
                continue
            s, e = max(0.0, min(s, max_time)), max(0.0, min(e, max_time))
            out.append([s, e])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--max_time", type=float, default=60.0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(args.test_json) as f:
        test_data = json.load(f)
    with open(args.results) as f:
        raw_results = json.load(f)

    print(f"[1/3] Test: {len(test_data)}, Results: {len(raw_results)}")

    all_ious = []
    total_preds = fp_preds = total_gts = fn_gts = 0
    parse_ok = parse_fail = hallucination = 0

    # 10s cap breakdown
    le10_ious, le10_preds, le10_fp, le10_gts, le10_fn = [], 0, 0, 0, 0
    gt10_ious, gt10_preds, gt10_fp, gt10_gts, gt10_fn = [], 0, 0, 0, 0

    for result, gt_item in zip(raw_results, test_data):
        gt_segments = gt_item.get("gt_segments", [])
        gt_label = gt_item.get("gt_label", "")
        if not gt_segments:
            continue
        gt_max_end = max(float(s[1]) for s in gt_segments)
        is_le10 = gt_max_end <= 10.0

        if result.get("error"):
            parse_fail += 1
            for _ in gt_segments:
                all_ious.append(0.0)
                (le10_ious if is_le10 else gt10_ious).append(0.0)
            total_gts += len(gt_segments); fn_gts += len(gt_segments)
            if is_le10:
                le10_gts += len(gt_segments); le10_fn += len(gt_segments)
            else:
                gt10_gts += len(gt_segments); gt10_fn += len(gt_segments)
            continue

        raw_pred = result.get("pred", "")
        all_events = EVENT_RE.findall(raw_pred)
        pred_segs = parse_segments(raw_pred, gt_label, max_time=args.max_time)
        if all_events and not pred_segs:
            hallucination += 1
        if not pred_segs:
            parse_fail += 1
            for _ in gt_segments:
                all_ious.append(0.0)
                (le10_ious if is_le10 else gt10_ious).append(0.0)
            total_gts += len(gt_segments); fn_gts += len(gt_segments)
            if is_le10:
                le10_gts += len(gt_segments); le10_fn += len(gt_segments)
            else:
                gt10_gts += len(gt_segments); gt10_fn += len(gt_segments)
            continue

        parse_ok += 1
        per_gt, n_fp, n_pred, n_fn, n_gt = score_sample(gt_segments, pred_segs)
        all_ious.extend(per_gt)
        total_preds += n_pred; fp_preds += n_fp
        total_gts += n_gt; fn_gts += n_fn
        if is_le10:
            le10_ious.extend(per_gt); le10_preds += n_pred; le10_fp += n_fp
            le10_gts += n_gt; le10_fn += n_fn
        else:
            gt10_ious.extend(per_gt); gt10_preds += n_pred; gt10_fp += n_fp
            gt10_gts += n_gt; gt10_fn += n_fn

    summary = summarize(all_ious, total_preds, fp_preds, total_gts, fn_gts,
                        parse_ok, parse_fail, len(raw_results))
    summary["parser"] = "crab_plus (<event>{L}, (s e)...</event>, fuzzy label)"
    summary["hallucination"] = hallucination

    def _split(ious, pred, fp, gt, fn):
        arr = np.array(ious) if ious else np.zeros(0)
        return {
            "n_gt": gt, "n_pred": pred,
            "mIoU_union_%": round(float(arr.mean()) * 100, 4) if len(arr) else 0.0,
            "FP_rate_%": round(fp / pred * 100, 4) if pred else 0.0,
            "FN_rate_%": round(fn / gt * 100, 4) if gt else 0.0,
        }
    summary["GT_max_end_le10s"] = _split(le10_ious, le10_preds, le10_fp, le10_gts, le10_fn)
    summary["GT_max_end_gt10s"] = _split(gt10_ious, gt10_preds, gt10_fp, gt10_gts, gt10_fn)

    print_report("Crab+ × UnAV-100 — Union-IoU", summary)
    print(f"  [GT ≤ 10s]  n_gt={summary['GT_max_end_le10s']['n_gt']}  "
          f"mIoU={summary['GT_max_end_le10s']['mIoU_union_%']:.2f}%  "
          f"FP={summary['GT_max_end_le10s']['FP_rate_%']:.2f}%  FN={summary['GT_max_end_le10s']['FN_rate_%']:.2f}%")
    print(f"  [GT > 10s]  n_gt={summary['GT_max_end_gt10s']['n_gt']}  "
          f"mIoU={summary['GT_max_end_gt10s']['mIoU_union_%']:.2f}%  "
          f"FP={summary['GT_max_end_gt10s']['FP_rate_%']:.2f}%  FN={summary['GT_max_end_gt10s']['FN_rate_%']:.2f}%")
    print(f"  hallucination: {hallucination}")

    summary_path = os.path.join(out_dir, "eval_miou_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {summary_path}")


if __name__ == "__main__":
    main()
