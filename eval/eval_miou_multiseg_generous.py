#!/usr/bin/env python3
"""
eval_miou_multiseg_generous.py
  - eval_miou_multiseg.py 와 같은 지표를 계산하지만 파서가 더 관대 ("generous").
  - 기존 strict 파서가 못 잡은 패턴을 추가 인식:
      (1) `<tD>×N . <tD>×M` 의 평문 '.' / ':' separator 도 `<tdot>` 와 동등 처리.
      (2) 정수부/소수부 자릿수가 비표준이어도 최대한 디코딩 (자릿수 자동 결정).
      (3) `<t*>` 토큰 없는 plain decimal `From 0.5 to 1.2` 형태도 fallback 으로 허용.
  - 출력 파일명: eval_miou_summary_generous.json (기본). --summary_name 으로 변경 가능.
"""
import argparse
import datetime as _dt
import json
import os
import re
import numpy as np


# === 기존 strict 와 동일한 helpers ===

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


# === Generous parser ===

# `<tD>` 시퀀스 + 옵션 separator (<tdot>|.|:) + 옵션 fractional `<tD>` 시퀀스
_VTG_NUM = (
    r"((?:<t\d>)+)"                     # int part
    r"(?:(<tdot>|\.|:)((?:<t\d>)+))?"   # optional [sep, frac]
)
_VTG_FROM_TO = re.compile(
    rf"[Ff]rom\s+{_VTG_NUM}\s+to\s+{_VTG_NUM}"
)

# vtg 토큰이 전혀 없을 때만 발동되는 평문 fallback
_PLAIN_FROM_TO = re.compile(
    r"[Ff]rom\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)?\s+to\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)?",
    re.IGNORECASE,
)


def _decode_vtg_with_sep(int_part_str, sep, frac_part_str, max_time=9999.9):
    """vtg 토큰 시퀀스 → seconds.

    int_part_str: '<t0><t0><t1><t5>' 형태
    sep         : None | '<tdot>' | '.' | ':'
    frac_part_str: '<t6>' 형태 또는 None
    """
    int_digits = re.findall(r"<t(\d)>", int_part_str)
    if not int_digits:
        return None
    integer_part = int("".join(int_digits))
    decimal_part = 0
    if sep is not None and frac_part_str:
        frac_digits = re.findall(r"<t(\d)>", frac_part_str)
        if frac_digits:
            # strict 파서와 동일하게 첫 자리만 사용 (0.1초 정밀도)
            decimal_part = int(frac_digits[0])
    t = integer_part + decimal_part / 10.0
    return min(t, max_time)


def parse_multi_segments_generous(raw, max_time=9999.9):
    """관대한 파서. strict 가 인식하는 패턴 + (1)(2)(3) 확장."""
    segments = []

    # 1차: vtg 토큰 + 관대한 separator 허용
    for m in _VTG_FROM_TO.finditer(raw):
        s_int, s_sep, s_frac, e_int, e_sep, e_frac = m.groups()
        start = _decode_vtg_with_sep(s_int, s_sep, s_frac, max_time)
        end = _decode_vtg_with_sep(e_int, e_sep, e_frac, max_time)
        if start is None or end is None:
            continue
        if end <= start:
            end = min(start + 0.1, max_time)
        segments.append([start, end])

    # 2차 fallback: vtg 토큰이 전혀 없는 응답만 plain decimal 시도 (overgen 방지)
    if not segments and "<t" not in raw:
        for m in _PLAIN_FROM_TO.finditer(raw):
            try:
                start = float(m.group(1))
                end = float(m.group(2))
                start = min(start, max_time)
                end = min(end, max_time)
                if end <= start:
                    end = min(start + 0.1, max_time)
                segments.append([start, end])
            except ValueError:
                continue

    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--max_time", type=float, default=9999.9)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--progress_log", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--summary_name", default="eval_miou_summary_generous.json")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.test_json, "r") as f:
        test_data = json.load(f)
    with open(args.results, "r") as f:
        raw_results = json.load(f)

    # results 가 부분 평가 (chunk 진행 중) 인 경우 test_data 도 잘라서 정렬 유지
    n_eval = min(len(test_data), len(raw_results))

    if not args.quiet:
        print(f"[1/3] Test: {len(test_data)}, Results: {len(raw_results)}, Eval n={n_eval}  (GENEROUS parser)")

    all_ious = []
    parse_ok = 0
    parse_fail = 0
    total_preds = 0
    fp_preds = 0

    for i in range(n_eval):
        result = raw_results[i]
        gt_item = test_data[i]
        gt_segments = gt_item.get("gt_segments", [])
        raw_pred = result.get("pred", "")

        pred_segments = parse_multi_segments_generous(raw_pred, max_time=args.max_time)

        if not pred_segments:
            for _ in gt_segments:
                all_ious.append(0.0)
            parse_fail += 1
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
    recall_at = {th: (float(np.mean(all_ious >= th)) if n > 0 else 0.0)
                 for th in iou_thresholds}

    fp_rate = fp_preds / total_preds if total_preds > 0 else 0.0
    fn_gt = int(np.sum(all_ious == 0.0)) if n > 0 else 0
    fn_rate = fn_gt / n if n > 0 else 0.0

    if not args.quiet:
        SEP = "=" * 52
        print(f"\n{SEP}")
        print("  Multi-Segment — Union-IoU mIoU + Recall@θ  [GENEROUS]")
        print(SEP)
        print(f"  Samples:     {n_eval}")
        print(f"  GT segments: {n}")
        print(f"  Parse OK:    {parse_ok} ({parse_ok*100/max(n_eval,1):.1f}%)")
        print(f"  Parse fail:  {parse_fail} ({parse_fail*100/max(n_eval,1):.1f}%)")
        print(f"  mIoU(union): {miou * 100:.2f}%")
        print()
        for th, val in sorted(recall_at.items()):
            print(f"  Recall @ IoU={th:.1f}:  {val * 100:.2f}%")
        print()
        print(f"  FP_rate:     {fp_rate * 100:.2f}%  ({fp_preds}/{total_preds})")
        print(f"  FN_rate:     {fn_rate * 100:.2f}%  ({fn_gt}/{n})")
        print(f"\n{SEP}\n")

    summary = {
        "parser": "generous",
        "mIoU_union_%": round(miou * 100, 4),
        "Recall": {str(k): round(v * 100, 4) for k, v in recall_at.items()},
        "FP_rate_%": round(fp_rate * 100, 4),
        "FN_rate_%": round(fn_rate * 100, 4),
        "n_pred_segments": total_preds,
        "n_fp_segments": fp_preds,
        "n_gt_segments": n,
        "n_fn_gt_segments": fn_gt,
        "n_samples": n_eval,
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
    }

    summary_path = os.path.join(out_dir, args.summary_name)
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


if __name__ == "__main__":
    main()
