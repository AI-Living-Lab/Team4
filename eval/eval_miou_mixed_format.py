#!/usr/bin/env python3
"""
eval_miou_mixed_format.py
  - 혼합 포맷(time token + 자연어 숫자)의 inference 결과 → mIoU + R@1 계산
  - 모델 응답 예시: "From 0<t0>00<tdot>0 to 0004.6."
  - time token과 일반 숫자를 모두 처리
"""
import argparse
import json
import os
import re
import numpy as np


def normalize_time_str(s):
    """Replace <tN> with N and <tdot> with . to get a plain number string."""
    s = re.sub(r"<t(\d)>", r"\1", s)
    s = s.replace("<tdot>", ".")
    return s


def extract_number(s):
    """Extract a float from a normalized string like '00004.6' or '0.0'."""
    m = re.search(r"(\d+\.?\d*)", s)
    if m:
        return float(m.group(1))
    return None


def parse_segments_mixed(raw, max_time=60.0):
    """Parse mixed-format segments from model output."""
    segments = []

    # Pattern 1: "From X to Y" or "From X - Y"
    # X and Y can contain <tN>, <tdot>, regular digits, dots
    token_or_digit = r"[0-9<>t\.\s]*?(?:<t\d>|[\d\.])[0-9<>tdot\.\s]*?"
    pattern = r"[Ff]rom\s+((?:[0-9]|<t\d>|<tdot>|\.)+)\s+(?:to|-)\s+((?:[0-9]|<t\d>|<tdot>|\.)+)"

    for m in re.finditer(pattern, raw):
        start_raw = normalize_time_str(m.group(1))
        end_raw = normalize_time_str(m.group(2))
        start = extract_number(start_raw)
        end = extract_number(end_raw)
        if start is not None and end is not None:
            start = min(start, max_time)
            end = min(end, max_time)
            if end <= start:
                end = min(start + 1.0, max_time)
            segments.append([start, end])

    # Pattern 2: "happens in X - Y seconds" or "at X - Y"
    if not segments:
        pattern2 = r"(?:in|at)\s+((?:[0-9]|<t\d>|<tdot>|\.)+)\s*-\s*((?:[0-9]|<t\d>|<tdot>|\.)+)"
        for m in re.finditer(pattern2, raw):
            start_raw = normalize_time_str(m.group(1))
            end_raw = normalize_time_str(m.group(2))
            start = extract_number(start_raw)
            end = extract_number(end_raw)
            if start is not None and end is not None:
                start = min(start, max_time)
                end = min(end, max_time)
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

    with open(args.test_json) as f:
        test_data = json.load(f)
    with open(args.results) as f:
        raw_results = json.load(f)

    print(f"[1/3] Test: {len(test_data)}, Results: {len(raw_results)}")

    all_ious = []
    parse_ok = 0
    parse_fail = 0

    for i, (result, gt_item) in enumerate(zip(raw_results, test_data)):
        pred_text = result.get("pred", "")
        gt_segs = gt_item.get("gt_segments", [])

        pred_segs = parse_segments_mixed(pred_text, args.max_time)

        if not pred_segs:
            parse_fail += 1
            all_ious.extend([0.0] * len(gt_segs))
            continue

        parse_ok += 1
        for gt_seg in gt_segs:
            best_iou = max(compute_tiou(gt_seg, ps) for ps in pred_segs)
            all_ious.append(best_iou)

    all_ious = np.array(all_ious)
    n_gt = len(all_ious)
    miou = all_ious.mean() * 100 if n_gt > 0 else 0.0

    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    recall = {}
    for th in thresholds:
        recall[str(th)] = float((all_ious >= th).sum() / max(n_gt, 1) * 100)

    summary = {
        "mIoU_%": round(miou, 4),
        "R@1": {k: round(v, 4) for k, v in recall.items()},
        "n_gt_segments": n_gt,
        "n_samples": len(raw_results),
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
    }

    print(f"\n{'='*52}")
    print(f"  UnAV-100 Multi-Segment (mixed format parser)")
    print(f"{'='*52}")
    print(f"  Samples:     {len(raw_results)}")
    print(f"  GT segments: {n_gt}")
    print(f"  Parse OK:    {parse_ok} ({parse_ok*100/max(len(raw_results),1):.1f}%)")
    print(f"  Parse fail:  {parse_fail} ({parse_fail*100/max(len(raw_results),1):.1f}%)")
    print(f"  mIoU:        {miou:.2f}%")
    for th in thresholds:
        print(f"  R@1 @ IoU={th}:  {recall[str(th)]:.2f}%")
    print(f"{'='*52}")

    out_path = os.path.join(out_dir, "eval_miou_mixed_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
