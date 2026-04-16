#!/usr/bin/env python3
"""
eval_miou_single.py
  - Single QA 형식의 inference 결과 → mIoU + R@1 계산
  - 각 test sample에 gt_label, gt_segment가 포함되어 있음
  - 모델 응답: "From <t...> to <t...>." 형식

Usage:
  python eval/eval_miou_single.py \
    --results  eval/results/.../test_results.json \
    --test_json data/unav100_test_grounding.json \
    --max_time 60.0 \
    --out_dir  eval/results/...
"""
import argparse
import json
import os
import re
import numpy as np


def decode_vtg_time(token_str: str, max_time: float = 60.0):
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


def parse_from_to(raw: str, max_time: float = 60.0):
    """Parse 'From <t...> to <t...>.' format."""
    # "From X to Y" 패턴
    m = re.search(r"[Ff]rom\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)\s+to\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)", raw)
    if m:
        start = decode_vtg_time(m.group(1), max_time)
        end = decode_vtg_time(m.group(2), max_time)
        if start is not None and end is not None:
            if end <= start:
                end = min(start + 1.0, max_time)
            return [start, end]

    # fallback: 아무 time token 2개 추출
    tokens = re.findall(r"((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)", raw)
    if len(tokens) >= 2:
        start = decode_vtg_time(tokens[0], max_time)
        end = decode_vtg_time(tokens[1], max_time)
        if start is not None and end is not None:
            if end <= start:
                end = min(start + 1.0, max_time)
            return [start, end]

    return None


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

    # 1. Test data 로드 (GT 포함)
    with open(args.test_json, "r") as f:
        test_data = json.load(f)
    print(f"[1/3] Test data: {len(test_data)} samples")

    # 2. Inference 결과 로드
    with open(args.results, "r") as f:
        raw_results = json.load(f)
    print(f"[2/3] Inference results: {len(raw_results)} items")

    if len(raw_results) != len(test_data):
        print(f"[WARN] result count ({len(raw_results)}) != test count ({len(test_data)})")

    # 3. 파싱 + IoU 계산
    print("[3/3] Parsing & computing IoU ...")
    all_ious = []
    parse_ok = 0
    parse_fail = 0
    failed_samples = []
    predictions = []

    for i, (result, gt_item) in enumerate(zip(raw_results, test_data)):
        gt_seg = gt_item["gt_segment"]
        gt_label = gt_item["gt_label"]
        raw_pred = result.get("pred", "")

        pred_seg = parse_from_to(raw_pred, max_time=args.max_time)

        if pred_seg is None:
            all_ious.append(0.0)
            parse_fail += 1
            if len(failed_samples) < 50:
                failed_samples.append({
                    "index": i,
                    "gt_label": gt_label,
                    "gt_segment": gt_seg,
                    "pred": raw_pred[:200],
                })
        else:
            iou = compute_tiou(pred_seg, gt_seg)
            all_ious.append(iou)
            parse_ok += 1
            predictions.append({
                "index": i,
                "gt_label": gt_label,
                "gt_segment": gt_seg,
                "pred_segment": pred_seg,
                "iou": round(iou, 4),
            })

    all_ious = np.array(all_ious)
    n = len(all_ious)
    miou = float(np.mean(all_ious)) if n > 0 else 0.0

    iou_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    recall_at = {}
    for th in iou_thresholds:
        recall_at[th] = float(np.mean(all_ious >= th)) if n > 0 else 0.0

    # 결과 출력
    SEP = "=" * 52
    print(f"\n{SEP}")
    print("  UnAV-100 Single QA — mIoU + R@1")
    print(SEP)
    print(f"  Total samples:  {n}")
    print(f"  Parse OK:       {parse_ok} ({parse_ok*100/n:.1f}%)")
    print(f"  Parse fail:     {parse_fail} ({parse_fail*100/n:.1f}%)")
    print(f"  mIoU:           {miou * 100:.2f}%")
    print()
    for th, val in sorted(recall_at.items()):
        print(f"  R@1 @ IoU={th:.1f}:  {val * 100:.2f}%")
    print(f"\n{SEP}\n")

    # 저장
    summary = {
        "mIoU_%": round(miou * 100, 4),
        "R@1": {str(k): round(v * 100, 4) for k, v in recall_at.items()},
        "n_samples": n,
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
    }

    summary_path = os.path.join(out_dir, "eval_miou_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pred_path = os.path.join(out_dir, "predictions_miou.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    failed_path = os.path.join(out_dir, "failed_parses_miou.json")
    with open(failed_path, "w") as f:
        json.dump(failed_samples, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {pred_path}")
    print(f"[SAVED] {failed_path} ({len(failed_samples)} items)")


if __name__ == "__main__":
    main()
