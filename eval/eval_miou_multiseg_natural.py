#!/usr/bin/env python3
"""
eval_miou_multiseg_natural.py
  - 베이스 모델(video_salmonn2_plus_7B, time-token 학습 안 됨)이 자연어로 뱉은
    "From X to Y" 류 응답을 파싱하여 eval_miou_multiseg.py와 동일한 metric 계산.
  - 평가 로직(Union-IoU mIoU + Recall@θ + FP_rate)은 token 버전과 완전히 동일.
  - 차이는 parse_multi_segments_natural 하나뿐.
"""
import argparse
import json
import os
import re
import numpy as np


# X / X.Y / M:SS(.s) / H:MM:SS(.s) — 초 단위 float로 변환
_NUM = r"(\d+(?::\d+){0,2}(?:\.\d+)?)"
# 선택적 단위 (영/한)
_UNIT = r"(?:\s*(?:s|secs?|seconds?|초))?"

# (X, Y) 쌍 추출 패턴
_PAIR_PATTERNS = [
    # from X to Y
    re.compile(rf"[Ff]rom\s+{_NUM}{_UNIT}\s+to\s+{_NUM}{_UNIT}"),
    # between X and|to Y
    re.compile(rf"[Bb]etween\s+{_NUM}{_UNIT}\s+(?:and|to)\s+{_NUM}{_UNIT}"),
    # X - Y seconds  (대시 범위, 단위 필수로 오탐 차단)
    re.compile(rf"{_NUM}\s*[-–—]\s*{_NUM}\s+(?:s|secs?|seconds?|초)\b"),
    # starts ... at|from X (and) (continues until|ends at|until) Y
    re.compile(
        rf"start(?:s|ing|ed)?\s+(?:\w+\s+){{0,4}}(?:at|from)\s+{_NUM}{_UNIT}"
        rf"[^.]{{0,80}}?"
        rf"(?:continue[s]?\s+(?:to\s+do\s+so\s+)?until|until|ends?\s+at|continues?\s+(?:until|to))"
        rf"\s+{_NUM}{_UNIT}",
        re.IGNORECASE,
    ),
]

# 단일 숫자 -> [0, N]: "(in) the first N seconds"
_FIRST_N_PATTERN = re.compile(rf"(?:in\s+)?the\s+first\s+{_NUM}\s+(?:s|secs?|seconds?|초)\b", re.IGNORECASE)

# 전구간 표현: "throughout (the entire/whole) video"
_THROUGHOUT_PATTERN = re.compile(r"throughout\s+(?:the\s+)?(?:entire\s+|whole\s+)?video", re.IGNORECASE)


def _to_seconds(s):
    """ "12.5" / "1:23" / "1:23:45" / "0:01:23.5" -> float seconds. None on failure. """
    s = s.strip()
    try:
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 2:
                # MM:SS(.s)
                return float(parts[0]) * 60.0 + float(parts[1])
            elif len(parts) == 3:
                # HH:MM:SS(.s)
                return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
            return None
        return float(s)
    except Exception:
        return None


def parse_multi_segments_natural(raw, max_time=60.0):
    """ 자연어 응답에서 [start, end] 구간 리스트 추출.
        중복(반올림 0.01s 단위) 제거, end<=start는 0.1s로 보정. """
    segments = []
    seen = set()

    def _add(s, e):
        if s is None or e is None:
            return
        s = max(0.0, min(s, max_time))
        e = max(0.0, min(e, max_time))
        if e <= s:
            e = min(s + 0.1, max_time)
        key = (round(s, 2), round(e, 2))
        if key in seen:
            return
        seen.add(key)
        segments.append([s, e])

    # (X, Y) 쌍 패턴
    for pat in _PAIR_PATTERNS:
        for m in pat.finditer(raw):
            _add(_to_seconds(m.group(1)), _to_seconds(m.group(2)))

    # "first N seconds" -> [0, N]
    for m in _FIRST_N_PATTERN.finditer(raw):
        _add(0.0, _to_seconds(m.group(1)))

    # "throughout the (entire) video" -> [0, max_time]
    if _THROUGHOUT_PATTERN.search(raw):
        _add(0.0, max_time)

    return segments


# ---------- 아래는 eval_miou_multiseg.py와 동일 ----------

def compute_tiou(seg1, seg2):
    inter_s = max(seg1[0], seg2[0])
    inter_e = min(seg1[1], seg2[1])
    inter = max(0.0, inter_e - inter_s)
    union = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - inter
    return inter / (union + 1e-8) if union > 0 else 0.0


def merge_intervals(intervals):
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
    ap.add_argument("--dump_parse_fail", type=int, default=0,
                    help="파싱 실패 샘플 처음 N개의 raw pred를 별도 파일로 저장 (디버깅용)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.test_json, "r") as f:
        test_data = json.load(f)

    with open(args.results, "r") as f:
        raw_results = json.load(f)

    print(f"[1/3] Test: {len(test_data)}, Results: {len(raw_results)}")

    all_ious = []
    parse_ok = 0
    parse_fail = 0
    total_preds = 0
    fp_preds = 0
    fail_samples = []

    for i, (result, gt_item) in enumerate(zip(raw_results, test_data)):
        gt_segments = gt_item.get("gt_segments", [])
        raw_pred = result.get("pred", "")

        pred_segments = parse_multi_segments_natural(raw_pred, max_time=args.max_time)

        if not pred_segments:
            for _ in gt_segments:
                all_ious.append(0.0)
            parse_fail += 1
            if len(fail_samples) < args.dump_parse_fail:
                fail_samples.append({"idx": i, "pred": raw_pred[:500]})
            continue

        parse_ok += 1

        for p in pred_segments:
            total_preds += 1
            has_overlap = any(
                min(p[1], g[1]) > max(p[0], g[0]) for g in gt_segments
            )
            if not has_overlap:
                fp_preds += 1

        for gt_seg in gt_segments:
            all_ious.append(compute_union_iou(gt_seg, pred_segments))

    all_ious = np.array(all_ious)
    n = len(all_ious)
    miou = float(np.mean(all_ious)) if n > 0 else 0.0

    iou_thresholds = [0.3, 0.5, 0.7]
    recall_at = {th: float(np.mean(all_ious >= th)) if n > 0 else 0.0
                 for th in iou_thresholds}

    fp_rate = fp_preds / total_preds if total_preds > 0 else 0.0

    SEP = "=" * 52
    print(f"\n{SEP}")
    print("  Multi-Segment (NATURAL) — Union-IoU mIoU + Recall@θ")
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
    print(f"\n{SEP}\n")

    summary = {
        "mIoU_union_%": round(miou * 100, 4),
        "Recall": {str(k): round(v * 100, 4) for k, v in recall_at.items()},
        "FP_rate_%": round(fp_rate * 100, 4),
        "n_pred_segments": total_preds,
        "n_fp_segments": fp_preds,
        "n_gt_segments": n,
        "n_samples": len(raw_results),
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
    }

    summary_path = os.path.join(out_dir, "eval_miou_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {summary_path}")

    if args.dump_parse_fail and fail_samples:
        fail_path = os.path.join(out_dir, "parse_fail_samples.json")
        with open(fail_path, "w") as f:
            json.dump(fail_samples, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {fail_path}")


if __name__ == "__main__":
    main()
