#!/usr/bin/env python3
"""
eval_miou_multiseg.py
  - Multi-segment QA 형식의 inference 결과 → Union-IoU 기반 mIoU + Recall@θ 계산
  - 모델 응답: "From <t...> to <t...>. From <t...> to <t...>."
  - GT: gt_segments 필드 (여러 구간)
  - 각 GT segment에 대해, 그와 겹치는 pred segments들의 합집합(Union)을 구해
    GT vs Union IoU를 계산 → 쪼개진 예측/과장된 예측 모두 합리적으로 반영
"""
import argparse
import datetime as _dt
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
                # 비정상 예측은 보수적으로 최소 단위(0.1초)만 부여 — 점수 이득 최소화
                end = min(start + 0.1, max_time)
            segments.append([start, end])
    return segments


def compute_tiou(seg1, seg2):
    inter_s = max(seg1[0], seg2[0])
    inter_e = min(seg1[1], seg2[1])
    inter = max(0.0, inter_e - inter_s)
    union = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - inter
    return inter / (union + 1e-8) if union > 0 else 0.0


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
    """Union-IoU: GT 한 개 vs 그와 겹치는 pred들의 합집합 사이의 IoU."""
    overlapping = [
        p for p in pred_segs
        if min(p[1], gt_seg[1]) > max(p[0], gt_seg[0])
    ]
    if not overlapping:
        return 0.0
    U = merge_intervals(overlapping)
    G = [list(gt_seg)]
    inter_len = intervals_total_len(intervals_intersect(G, U))
    union_len = intervals_total_len(merge_intervals(G + U))
    return inter_len / union_len if union_len > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--max_time", type=float, default=60.0)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--progress_log", default=None,
                    help="When set, append a single-line JSON snapshot of metrics to this JSONL file.")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress console report. Summary JSON is still written.")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.test_json, "r") as f:
        test_data = json.load(f)

    with open(args.results, "r") as f:
        raw_results = json.load(f)

    if not args.quiet:
        print(f"[1/3] Test: {len(test_data)}, Results: {len(raw_results)}")

    # Match results to test data
    # Results have "pred" field, test_data has "gt_segments"
    all_ious = []
    parse_ok = 0
    parse_fail = 0
    total_preds = 0
    fp_preds = 0

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

        # Count FP predictions: preds that overlap with no GT at all
        for p in pred_segments:
            total_preds += 1
            has_overlap = any(
                min(p[1], g[1]) > max(p[0], g[0]) for g in gt_segments
            )
            if not has_overlap:
                fp_preds += 1

        # For each GT segment, compute Union-IoU against all overlapping predictions
        for gt_seg in gt_segments:
            all_ious.append(compute_union_iou(gt_seg, pred_segments))

    all_ious = np.array(all_ious)
    n = len(all_ious)
    miou = float(np.mean(all_ious)) if n > 0 else 0.0

    iou_thresholds = [0.3, 0.5, 0.7]
    recall_at = {}
    for th in iou_thresholds:
        recall_at[th] = float(np.mean(all_ious >= th)) if n > 0 else 0.0

    fp_rate = fp_preds / total_preds if total_preds > 0 else 0.0
    fn_gt = int(np.sum(all_ious == 0.0)) if n > 0 else 0
    fn_rate = fn_gt / n if n > 0 else 0.0

    if not args.quiet:
        SEP = "=" * 52
        print(f"\n{SEP}")
        print("  Multi-Segment — Union-IoU mIoU + Recall@θ")
        print(SEP)
        print(f"  Samples:     {len(raw_results)}")
        print(f"  GT segments: {n}")
        print(f"  Parse OK:    {parse_ok} ({parse_ok*100/max(len(raw_results),1):.1f}%)")
        print(f"  Parse fail:  {parse_fail} ({parse_fail*100/max(len(raw_results),1):.1f}%)")
        print(f"  mIoU(union): {miou * 100:.2f}%")
        print()
        for th, val in sorted(recall_at.items()):
            print(f"  Recall @ IoU={th:.1f}:  {val * 100:.2f}%")
        print()
        print(f"  FP_rate:     {fp_rate * 100:.2f}%  ({fp_preds}/{total_preds})")
        print(f"  FN_rate:     {fn_rate * 100:.2f}%  ({fn_gt}/{n})")
        print(f"\n{SEP}\n")

    summary = {
        "mIoU_union_%": round(miou * 100, 4),
        "Recall": {str(k): round(v * 100, 4) for k, v in recall_at.items()},
        "FP_rate_%": round(fp_rate * 100, 4),
        "FN_rate_%": round(fn_rate * 100, 4),
        "n_pred_segments": total_preds,
        "n_fp_segments": fp_preds,
        "n_gt_segments": n,
        "n_fn_gt_segments": fn_gt,
        "n_samples": len(raw_results),
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
    }

    summary_path = os.path.join(out_dir, "eval_miou_summary.json")
    tmp_summary = summary_path + ".tmp"
    with open(tmp_summary, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    os.replace(tmp_summary, summary_path)
    if not args.quiet:
        print(f"[SAVED] {summary_path}")

    if args.progress_log:
        snapshot = {"timestamp": _dt.datetime.now().isoformat(timespec="seconds"), **summary}
        with open(args.progress_log, "a") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        if not args.quiet:
            print(f"[PROGRESS] appended snapshot -> {args.progress_log}")


if __name__ == "__main__":
    main()
