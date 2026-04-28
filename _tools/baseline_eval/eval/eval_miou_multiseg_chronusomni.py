"""ChronusOmni × UnAV-100 eval — Union-IoU + FP_rate + FN_rate.

Parser: `From second{X} to second{Y}` 또는 `second{X}-second{Y}` (hybrid 및 native 둘 다).
cdh 표준 인터페이스: --results --test_json --max_time --out_dir
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import score_sample, summarize, print_report

SEG_RE = re.compile(
    r"second\{([\d.]+)\}\s*(?:to|-)\s*second\{([\d.]+)\}",
    re.IGNORECASE,
)


def parse_segments(raw: str, max_time: float = 60.0):
    out = []
    for m in SEG_RE.finditer(raw):
        try:
            s, e = float(m.group(1)), float(m.group(2))
        except ValueError:
            continue
        if e < s:
            s, e = e, s
        if e <= s:
            continue
        s, e = min(s, max_time), min(e, max_time)
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
    parse_ok = parse_fail = 0

    for result, gt_item in zip(raw_results, test_data):
        gt_segments = gt_item.get("gt_segments", [])
        if not gt_segments:
            continue
        if result.get("error"):
            parse_fail += 1
            for _ in gt_segments:
                all_ious.append(0.0)
            total_gts += len(gt_segments)
            fn_gts += len(gt_segments)
            continue
        pred_segs = parse_segments(result.get("pred", ""), max_time=args.max_time)
        if not pred_segs:
            parse_fail += 1
            for _ in gt_segments:
                all_ious.append(0.0)
            total_gts += len(gt_segments)
            fn_gts += len(gt_segments)
            continue
        parse_ok += 1
        per_gt, n_fp, n_pred, n_fn, n_gt = score_sample(gt_segments, pred_segs)
        all_ious.extend(per_gt)
        total_preds += n_pred
        fp_preds += n_fp
        total_gts += n_gt
        fn_gts += n_fn

    summary = summarize(all_ious, total_preds, fp_preds, total_gts, fn_gts,
                        parse_ok, parse_fail, len(raw_results))
    summary["parser"] = "chronusomni (second{X} to second{Y} / -)"
    print_report("ChronusOmni × UnAV-100 — Union-IoU", summary)

    summary_path = os.path.join(out_dir, "eval_miou_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {summary_path}")


if __name__ == "__main__":
    main()
