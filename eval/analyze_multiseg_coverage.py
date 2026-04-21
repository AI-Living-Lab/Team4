#!/usr/bin/env python3
"""
Compare two models on multi-segment prediction behavior:
  1) Fraction of samples where pred has >=2 segments
  2) Among samples where GT is multi-segment but pred is single-segment,
     how well the single pred covers the GT union (coverage & IoU-to-union).
"""
import argparse
import json
import re


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
    t = int("".join(int_part)) + (int(dec_part[0]) / 10.0 if dec_part else 0)
    return min(t, max_time)


def parse_multi_segments(raw, max_time=60.0):
    segs = []
    pat = r"[Ff]rom\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)\s+to\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)"
    for m in re.finditer(pat, raw):
        s = decode_vtg_time(m.group(1), max_time)
        e = decode_vtg_time(m.group(2), max_time)
        if s is not None and e is not None:
            if e <= s:
                e = min(s + 1.0, max_time)
            segs.append([s, e])
    return segs


def gt_union_span(gts):
    return [min(s for s, _ in gts), max(e for _, e in gts)]


def gt_total_length(gts):
    """Merge overlapping GT segments and sum lengths."""
    segs = sorted([tuple(s) for s in gts])
    merged = []
    for s, e in segs:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return sum(e - s for s, e in merged), merged


def coverage_of_gt(pred, merged_gts):
    """Fraction of GT total length that is inside pred interval."""
    total = sum(e - s for s, e in merged_gts)
    if total == 0:
        return 0.0
    covered = 0.0
    for s, e in merged_gts:
        inter = max(0.0, min(pred[1], e) - max(pred[0], s))
        covered += inter
    return covered / total


def iou_1d(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def analyze(results_path, test_path, name, max_time=60.0):
    with open(results_path) as f:
        preds = json.load(f)
    with open(test_path) as f:
        test = json.load(f)

    total = len(preds)
    n_parse_ok = 0
    n_pred_multi = 0
    n_pred_single = 0
    n_pred_zero = 0
    n_gt_multi = 0
    n_gt_single = 0

    # single-pred vs multi-gt analysis
    sp_mg_covs = []          # coverage of GT by single pred
    sp_mg_iou_union = []     # IoU vs union bounding interval
    sp_mg_contains_all = 0   # pred interval covers GT bounding span fully
    sp_mg_count = 0

    # multi-pred vs multi-gt: number of preds distribution
    pred_seg_counts = []
    gt_seg_counts = []

    # Single pred vs single GT: direct IoU
    sp_sg_ious = []

    for p, g in zip(preds, test):
        gts = g.get("gt_segments", [])
        raw = p.get("pred", "")
        pred_segs = parse_multi_segments(raw, max_time=max_time)

        gt_seg_counts.append(len(gts))
        pred_seg_counts.append(len(pred_segs))

        if len(pred_segs) == 0:
            n_pred_zero += 1
            continue
        n_parse_ok += 1

        if len(pred_segs) >= 2:
            n_pred_multi += 1
        else:
            n_pred_single += 1

        if len(gts) >= 2:
            n_gt_multi += 1
        else:
            n_gt_single += 1

        if len(pred_segs) == 1 and len(gts) >= 2:
            sp_mg_count += 1
            _, merged = gt_total_length(gts)
            cov = coverage_of_gt(pred_segs[0], merged)
            union_span = gt_union_span(gts)
            iou_u = iou_1d(pred_segs[0], union_span)
            contains = pred_segs[0][0] <= union_span[0] and pred_segs[0][1] >= union_span[1]
            sp_mg_covs.append(cov)
            sp_mg_iou_union.append(iou_u)
            if contains:
                sp_mg_contains_all += 1

        if len(pred_segs) == 1 and len(gts) == 1:
            sp_sg_ious.append(iou_1d(pred_segs[0], gts[0]))

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Total samples: {total}")
    print(f"  Parse OK: {n_parse_ok}  |  Parse fail: {n_pred_zero}")
    print(f"")
    print(f"  -- Prediction segment count --")
    print(f"    Pred = 0 segs:  {n_pred_zero:4d} ({n_pred_zero*100/total:.1f}%)")
    print(f"    Pred = 1 seg:   {n_pred_single:4d} ({n_pred_single*100/total:.1f}%)")
    print(f"    Pred >= 2 segs: {n_pred_multi:4d} ({n_pred_multi*100/total:.1f}%)")
    avg_pred = sum(pred_seg_counts) / max(total, 1)
    avg_gt = sum(gt_seg_counts) / max(total, 1)
    print(f"    avg preds/sample: {avg_pred:.2f}  |  avg GTs/sample: {avg_gt:.2f}")
    print(f"")
    print(f"  -- GT composition --")
    gt_multi_total = sum(1 for c in gt_seg_counts if c >= 2)
    gt_single_total = sum(1 for c in gt_seg_counts if c == 1)
    print(f"    GT = 1 seg:   {gt_single_total} ({gt_single_total*100/total:.1f}%)")
    print(f"    GT >= 2 segs: {gt_multi_total} ({gt_multi_total*100/total:.1f}%)")
    print(f"")
    print(f"  -- Single-pred × Multi-GT (n={sp_mg_count}) --")
    if sp_mg_count > 0:
        avg_cov = sum(sp_mg_covs) / sp_mg_count
        avg_iou_u = sum(sp_mg_iou_union) / sp_mg_count
        print(f"    Avg coverage of GT segments:     {avg_cov*100:.1f}%")
        print(f"    Avg IoU vs GT union interval:    {avg_iou_u*100:.1f}%")
        print(f"    Pred fully contains GT span:     {sp_mg_contains_all} ({sp_mg_contains_all*100/sp_mg_count:.1f}%)")
        cov_50 = sum(1 for c in sp_mg_covs if c >= 0.5)
        cov_80 = sum(1 for c in sp_mg_covs if c >= 0.8)
        cov_99 = sum(1 for c in sp_mg_covs if c >= 0.99)
        print(f"    Coverage >= 50%: {cov_50} ({cov_50*100/sp_mg_count:.1f}%)")
        print(f"    Coverage >= 80%: {cov_80} ({cov_80*100/sp_mg_count:.1f}%)")
        print(f"    Coverage = 100%: {cov_99} ({cov_99*100/sp_mg_count:.1f}%)")
    print(f"")
    if sp_sg_ious:
        avg_ss = sum(sp_sg_ious) / len(sp_sg_ious)
        print(f"  -- Single-pred × Single-GT reference --")
        print(f"    n={len(sp_sg_ious)}  avg IoU: {avg_ss*100:.1f}%")

    return {
        "name": name,
        "pred_multi_pct": n_pred_multi * 100 / total,
        "pred_single_pct": n_pred_single * 100 / total,
        "sp_mg_count": sp_mg_count,
        "sp_mg_avg_coverage": sum(sp_mg_covs)/sp_mg_count if sp_mg_count else 0,
        "sp_mg_avg_iou_union": sum(sp_mg_iou_union)/sp_mg_count if sp_mg_count else 0,
        "sp_mg_contains_all_pct": sp_mg_contains_all*100/sp_mg_count if sp_mg_count else 0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_results", required=True)
    ap.add_argument("--b_results", required=True)
    ap.add_argument("--a_name", default="Model A")
    ap.add_argument("--b_name", default="Model B")
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--max_time", type=float, default=60.0)
    args = ap.parse_args()

    a = analyze(args.a_results, args.test_json, args.a_name, args.max_time)
    b = analyze(args.b_results, args.test_json, args.b_name, args.max_time)

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'':30s} {a['name']:>13s}  {b['name']:>13s}")
    print(f"  {'pred multi (>=2 seg) %':30s} {a['pred_multi_pct']:13.1f}  {b['pred_multi_pct']:13.1f}")
    print(f"  {'pred single (=1 seg)  %':30s} {a['pred_single_pct']:13.1f}  {b['pred_single_pct']:13.1f}")
    print(f"  {'SP×MG avg coverage %':30s} {a['sp_mg_avg_coverage']*100:13.1f}  {b['sp_mg_avg_coverage']*100:13.1f}")
    print(f"  {'SP×MG avg IoU-to-union %':30s} {a['sp_mg_avg_iou_union']*100:13.1f}  {b['sp_mg_avg_iou_union']*100:13.1f}")
    print(f"  {'SP×MG contains full span %':30s} {a['sp_mg_contains_all_pct']:13.1f}  {b['sp_mg_contains_all_pct']:13.1f}")
