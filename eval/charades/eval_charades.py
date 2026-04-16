#!/usr/bin/env python3
"""
Charades-STA evaluation: parse vS2 predictions and compute R@1 at IoU thresholds.

Usage:
  python eval_charades.py \
    --results  results/charades_test_XXXX/test_results.json \
    --test_json charades_sta_test.json \
    --out_dir  results/charades_test_XXXX
"""
import argparse, json, os, re
from collections import defaultdict


def parse_time_tokens(text):
    """
    Parse PU-VALOR style time tokens from model output.
    Handles formats like:
      - From <t0><t0><t2><t4><tdot><t3> to <t0><t0><t3><t0><tdot><t4>.
      - From 0024.3 to 0030.4.
      - Mixed: <t0><t0>24<tdot>3 etc.
    Returns (start_sec, end_sec) or None.
    """
    if text is None:
        return None

    # Try to find "From ... to ..." pattern
    m = re.search(r'[Ff]rom\s+(.+?)\s+to\s+(.+?)[\.\s,]', text)
    if not m:
        return None

    start_str = m.group(1).strip()
    end_str = m.group(2).strip()

    start = _tokens_to_seconds(start_str)
    end = _tokens_to_seconds(end_str)

    if start is None or end is None:
        return None
    return start, end


def _tokens_to_seconds(s):
    """Convert a time token string to seconds."""
    # Replace <tN> with digit N, <tdot> with '.'
    s = re.sub(r'<t(\d)>', r'\1', s)
    s = re.sub(r'<tdot>', '.', s)
    # Remove any remaining non-numeric/dot characters
    s = re.sub(r'[^0-9.]', '', s)
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def compute_iou(pred_start, pred_end, gt_start, gt_end):
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    inter = max(0, inter_end - inter_start)
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    if union <= 0:
        return 0.0
    return inter / union


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="test_results.json from inference")
    parser.add_argument("--test_json", default="/home/aix23102/audiolm/vS2_eunji/eval/charades/charades_sta_test.json")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)
    with open(args.test_json) as f:
        test_data = json.load(f)

    # Build GT lookup by index
    gt_map = {i: (d["gt_start"], d["gt_end"]) for i, d in enumerate(test_data)}

    thresholds = [0.3, 0.5, 0.7]
    correct = {t: 0 for t in thresholds}
    total = 0
    parse_ok = 0
    parse_fail = 0
    failed_examples = []
    iou_sum_all = 0.0       # parse 실패 = IoU 0 포함
    iou_sum_parsed = 0.0    # parse 성공만

    for i, res in enumerate(results):
        pred_text = res.get("pred", "")
        gt_start = res.get("gt_start") or test_data[i]["gt_start"] if i < len(test_data) else None
        gt_end = res.get("gt_end") or test_data[i]["gt_end"] if i < len(test_data) else None

        if gt_start is None or gt_end is None:
            continue

        total += 1
        parsed = parse_time_tokens(pred_text)

        if parsed is None:
            parse_fail += 1
            # parse 실패 시 IoU = 0
            iou_sum_all += 0.0
            if len(failed_examples) < 50:
                failed_examples.append({"index": i, "pred": pred_text[:300]})
            continue

        parse_ok += 1
        pred_start, pred_end = parsed
        if pred_start > pred_end:
            pred_start, pred_end = pred_end, pred_start

        iou = compute_iou(pred_start, pred_end, gt_start, gt_end)
        iou_sum_all += iou
        iou_sum_parsed += iou

        for t in thresholds:
            if iou >= t:
                correct[t] += 1

    # Results
    miou_all = iou_sum_all / total if total > 0 else 0
    miou_parsed = iou_sum_parsed / parse_ok if parse_ok > 0 else 0

    print(f"\n{'='*50}")
    print(f"Charades-STA Evaluation Results")
    print(f"{'='*50}")
    print(f"Total samples:  {total}")
    print(f"Parse OK:       {parse_ok} ({100*parse_ok/total:.1f}%)")
    print(f"Parse failed:   {parse_fail} ({100*parse_fail/total:.1f}%)")
    print()
    print(f"mIoU:           {100*miou_all:.2f}%  (among parsed: {100*miou_parsed:.2f}%)")
    for t in thresholds:
        r1 = 100 * correct[t] / total if total > 0 else 0
        r1_parsed = 100 * correct[t] / parse_ok if parse_ok > 0 else 0
        print(f"R@1, IoU={t:.1f}:  {r1:.2f}%  (among parsed: {r1_parsed:.2f}%)")

    summary = {
        "total": total,
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
        "mIoU": round(100 * miou_all, 2),
        "mIoU_parsed_only": round(100 * miou_parsed, 2),
        "metrics": {f"R@1_IoU={t}": round(100 * correct[t] / total, 2) if total > 0 else 0 for t in thresholds},
        "metrics_parsed_only": {f"R@1_IoU={t}": round(100 * correct[t] / parse_ok, 2) if parse_ok > 0 else 0 for t in thresholds},
    }

    out_dir = args.out_dir or os.path.dirname(args.results)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "eval_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        if failed_examples:
            with open(os.path.join(out_dir, "failed_parses.json"), "w") as f:
                json.dump(failed_examples, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {out_dir}/eval_summary.json")


if __name__ == "__main__":
    main()
