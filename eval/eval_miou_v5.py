#!/usr/bin/env python3
"""
eval_miou_v5.py
  v5 학습 (4+1 또는 3+2 토큰화) 결과의 mIoU/R@1 평가.
  eval_miou_multiseg.py 의 변종 — decimal-digit 수가 가변적이라
  decode_vtg_time 가 모든 decimal token 을 사용해 시간 복원.

GT: gt_segments (raw seconds)
Pred: "From <t...> to <t...>. From <t...> to <t...>." (multi-segment)
"""
import argparse
import json
import os
import re


def decode_vtg_time(token_str, max_time=999.99):
    """variable decimal precision 지원.

    예) "<t0><t0><t3><tdot><t5><t6>" (3+2) → 003.56 = 3.56
    """
    has_dot = "<tdot>" in token_str
    if has_dot:
        parts = token_str.split("<tdot>", 1)
        int_part = re.findall(r"<t(\d)>", parts[0])
        dec_part = re.findall(r"<t(\d)>", parts[1]) if len(parts) > 1 else []
    else:
        int_part = re.findall(r"<t(\d)>", token_str)
        dec_part = []
    if not int_part:
        return None
    integer_part = int("".join(int_part))
    if dec_part:
        decimal_int = int("".join(dec_part))
        n_dec = len(dec_part)
        t = integer_part + decimal_int / (10 ** n_dec)
    else:
        t = float(integer_part)
    return min(t, max_time)


def parse_multi_segments(raw, max_time=999.99):
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
    ap.add_argument("--results", required=True, help="test_results_rank0.json")
    ap.add_argument("--test_json", required=True, help="unav_test_t2_tok{4_1|3_2}.json")
    ap.add_argument("--max_time", type=float, default=60.0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.test_json, "r") as f:
        test_data = json.load(f)
    with open(args.results, "r") as f:
        raw_results = json.load(f)

    print(f"[v5_eval] test={len(test_data)}, results={len(raw_results)}")

    all_per_gt_ious = []  # 각 GT segment 의 best-pred IoU
    parse_ok = parse_fail = 0
    n_eval = min(len(test_data), len(raw_results))

    for i in range(n_eval):
        gt_item = test_data[i]
        result = raw_results[i]
        pred_str = result.get("pred", "")
        gt_segments = gt_item.get("gt_segments", [])
        if not gt_segments:
            continue

        pred_segments = parse_multi_segments(pred_str, args.max_time)
        if not pred_segments:
            parse_fail += 1
            # 파싱 실패 시 모든 GT 에 IoU=0
            for _ in gt_segments:
                all_per_gt_ious.append(0.0)
            continue
        parse_ok += 1

        # 각 GT 에 대해 best matching pred 의 IoU
        for gt in gt_segments:
            best_iou = 0.0
            for pred in pred_segments:
                iou = compute_tiou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
            all_per_gt_ious.append(best_iou)

    if not all_per_gt_ious:
        print("[v5_eval] no GT segments scored; aborting")
        return

    mIoU = sum(all_per_gt_ious) / len(all_per_gt_ious)
    R03 = sum(1 for x in all_per_gt_ious if x >= 0.3) / len(all_per_gt_ious)
    R05 = sum(1 for x in all_per_gt_ious if x >= 0.5) / len(all_per_gt_ious)
    R07 = sum(1 for x in all_per_gt_ious if x >= 0.7) / len(all_per_gt_ious)

    summary = {
        "n_samples": n_eval,
        "n_gt_segments_scored": len(all_per_gt_ious),
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
        "mIoU_%": mIoU * 100,
        "R@1": {"0.3": R03 * 100, "0.5": R05 * 100, "0.7": R07 * 100},
        "max_time": args.max_time,
    }
    out_path = os.path.join(out_dir, "eval_miou_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[v5_eval] mIoU={mIoU*100:.2f}%  R@0.3={R03*100:.2f}%  R@0.5={R05*100:.2f}%  R@0.7={R07*100:.2f}%")
    print(f"[v5_eval] parse_ok={parse_ok}/{n_eval} ({100*parse_ok/max(n_eval,1):.1f}%)")
    print(f"[v5_eval] summary → {out_path}")


if __name__ == "__main__":
    main()
