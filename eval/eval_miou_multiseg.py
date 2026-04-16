#!/usr/bin/env python3
"""
eval_miou_multiseg.py
  - Multi-segment QA 형식의 inference 결과 → mIoU + R@1 계산
  - 모델 응답: "From <t...> to <t...>. From <t...> to <t...>."
  - GT: gt_segments 필드 (여러 구간)
  - 각 GT segment에 대해 prediction 중 best IoU를 계산
"""
import argparse
import json
import os
import re
import numpy as np


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
    t = integer_part + decimal_part / 10.0
    return min(t, max_time)


def parse_multi_segments(raw, max_time=60.0):
    """Parse 'From X to Y. From X to Y.' format into list of segments."""
    segments = []
    pattern = r"[Ff]rom\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)\s+to\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)"
    for m in re.finditer(pattern, raw):
        start = decode_vtg_time(m.group(1), max_time)
        end = decode_vtg_time(m.group(2), max_time)
        if start is not None and end is not None:
            if end <= start:
                end = min(start + 1.0, max_time)
            segments.append([start, end])
    return segments


def compute_tiou(seg1, seg2):
    inter_s = max(seg1[0], seg2[0])
    inter_e = min(seg1[1], seg2[1])
    inter = max(0.0, inter_e - inter_s)
    union = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - inter
    return inter / (union + 1e-8) if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--max_time", type=float, default=60.0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.test_json, "r") as f:
        test_data = json.load(f)

    with open(args.results, "r") as f:
        raw_results = json.load(f)

    print(f"[1/3] Test: {len(test_data)}, Results: {len(raw_results)}")

    # Match results to test data
    # Results have "pred" field, test_data has "gt_segments"
    all_ious = []
    parse_ok = 0
    parse_fail = 0

    for i, (result, gt_item) in enumerate(zip(raw_results, test_data)):
        gt_segments = gt_item.get("gt_segments", [])
        gt_label = gt_item.get("gt_label", "")
        raw_pred = result.get("pred", "")

        pred_segments = parse_multi_segments(raw_pred, max_time=args.max_time)

        if not pred_segments:
            # No valid predictions
            for _ in gt_segments:
                all_ious.append(0.0)
            parse_fail += 1
            continue

        parse_ok += 1

        # For each GT segment, find best matching prediction
        for gt_seg in gt_segments:
            best_iou = max(compute_tiou(pred_seg, gt_seg) for pred_seg in pred_segments)
            all_ious.append(best_iou)

    all_ious = np.array(all_ious)
    n = len(all_ious)
    miou = float(np.mean(all_ious)) if n > 0 else 0.0

    iou_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    recall_at = {}
    for th in iou_thresholds:
        recall_at[th] = float(np.mean(all_ious >= th)) if n > 0 else 0.0

    SEP = "=" * 52
    print(f"\n{SEP}")
    print("  UnAV-100 Multi-Segment — mIoU + R@1")
    print(SEP)
    print(f"  Samples:     {len(raw_results)}")
    print(f"  GT segments: {n}")
    print(f"  Parse OK:    {parse_ok} ({parse_ok*100/max(len(raw_results),1):.1f}%)")
    print(f"  Parse fail:  {parse_fail} ({parse_fail*100/max(len(raw_results),1):.1f}%)")
    print(f"  mIoU:        {miou * 100:.2f}%")
    print()
    for th, val in sorted(recall_at.items()):
        print(f"  R@1 @ IoU={th:.1f}:  {val * 100:.2f}%")
    print(f"\n{SEP}\n")

    summary = {
        "mIoU_%": round(miou * 100, 4),
        "R@1": {str(k): round(v * 100, 4) for k, v in recall_at.items()},
        "n_gt_segments": n,
        "n_samples": len(raw_results),
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
    }

    summary_path = os.path.join(out_dir, "eval_miou_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {summary_path}")


if __name__ == "__main__":
    main()
