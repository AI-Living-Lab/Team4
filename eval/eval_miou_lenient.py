#!/usr/bin/env python3
"""
eval_miou_lenient.py
  - eval_miou_subgroup.py와 동일하지만, time-token segment를 더 느슨하게 파싱.
  - 변경점: "From " prefix를 필수로 요구하지 않음. 즉 단순히
    <t...> to <t...> 형태 (앞에 "From"이 있든 없든) 를 모두 segment로 인정.
  - 이외(decode, IoU, subgroup) 로직은 strict 버전과 동일.
  - parse_fail의 주된 원인이 단순 prefix 누락임을 검증하기 위한 진단용 스크립트.
"""
import argparse, json, os, re
from collections import defaultdict


# Strict (original) — kept here for side-by-side reporting
STRICT_PAT = r"[Ff]rom\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)\s+to\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)"
# Lenient — drop the "From " prefix requirement. Pair two timestamps via " to ".
LENIENT_PAT = r"((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)\s+to\s+((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)"


def decode_vtg_time(token_str, max_time=9999.0):
    has_dot = "<tdot>" in token_str
    if has_dot:
        parts = token_str.split("<tdot>")
        i_part = re.findall(r"<t(\d)>", parts[0])
        d_part = re.findall(r"<t(\d)>", parts[1]) if len(parts) > 1 else []
    else:
        i_part = re.findall(r"<t(\d)>", token_str)
        d_part = []
    if not i_part:
        return None
    integer = int("".join(i_part))
    decimal = int(d_part[0]) if d_part else 0
    return min(integer + decimal / 10.0, max_time)


def parse_segments(raw, pattern, max_time=9999.0):
    segs = []
    for m in re.finditer(pattern, raw):
        s = decode_vtg_time(m.group(1), max_time)
        e = decode_vtg_time(m.group(2), max_time)
        if s is None or e is None:
            continue
        if e <= s:
            e = min(s + 1.0, max_time)
        segs.append([s, e])
    return segs


def compute_tiou(a, b):
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / (union + 1e-8) if union > 0 else 0.0


def summarize(ious, name, n_samples, parse_ok, parse_fail, thresholds=(0.1, 0.3, 0.5, 0.7, 0.9)):
    import statistics
    miou = statistics.fmean(ious) if ious else 0.0
    recall = {th: (sum(1 for x in ious if x >= th) / len(ious) if ious else 0.0) for th in thresholds}
    print(f"--- {name} ---")
    print(f"  samples:     {n_samples}")
    print(f"  gt_segments: {len(ious)}")
    print(f"  parse_ok:    {parse_ok}  parse_fail: {parse_fail}")
    print(f"  mIoU:        {miou*100:.2f}%")
    for th in thresholds:
        print(f"  R@1 IoU={th}: {recall[th]*100:.2f}%")
    print()
    return {
        "n_samples": n_samples, "n_gt": len(ious),
        "parse_ok": parse_ok, "parse_fail": parse_fail,
        "mIoU_%": round(miou*100, 4),
        "R@1": {str(k): round(v*100, 4) for k, v in recall.items()},
    }


def evaluate(test, results, pattern, max_time):
    groups = defaultdict(lambda: {"ious": [], "n_samples": 0, "parse_ok": 0, "parse_fail": 0})
    overall = {"ious": [], "n_samples": 0, "parse_ok": 0, "parse_fail": 0}
    for gt_item, res in zip(test, results):
        gt_segs = gt_item.get("gt_segments", [])
        if not gt_segs:
            continue
        n_gt = gt_item.get("n_gt_segments", len(gt_segs))
        group_key = "single" if n_gt == 1 else "multi"

        overall["n_samples"] += 1
        groups[group_key]["n_samples"] += 1

        pred_segs = parse_segments(res.get("pred", ""), pattern, max_time=max_time)
        if not pred_segs:
            for _ in gt_segs:
                overall["ious"].append(0.0)
                groups[group_key]["ious"].append(0.0)
            overall["parse_fail"] += 1
            groups[group_key]["parse_fail"] += 1
            continue

        overall["parse_ok"] += 1
        groups[group_key]["parse_ok"] += 1
        for gs in gt_segs:
            best = max(compute_tiou(ps, gs) for ps in pred_segs)
            overall["ious"].append(best)
            groups[group_key]["ious"].append(best)
    return overall, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="merged_results.json")
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--max_time", type=float, default=9999.0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)

    test = json.load(open(args.test_json))
    results = json.load(open(args.results))
    print(f"test={len(test)}  results={len(results)}")
    assert len(test) == len(results), f"length mismatch {len(test)} vs {len(results)}"

    summary = {}
    for tag, pat in [("strict", STRICT_PAT), ("lenient", LENIENT_PAT)]:
        print()
        print("=" * 52)
        print(f"PARSER: {tag}    pattern={pat}")
        print("=" * 52)
        overall, groups = evaluate(test, results, pat, args.max_time)
        summary[tag] = {}
        summary[tag]["overall"] = summarize(overall["ious"], "OVERALL", overall["n_samples"], overall["parse_ok"], overall["parse_fail"])
        for k in ("single", "multi"):
            if k in groups:
                summary[tag][k] = summarize(groups[k]["ious"], f"SUBGROUP: {k}-seg", groups[k]["n_samples"], groups[k]["parse_ok"], groups[k]["parse_fail"])

    out_path = os.path.join(out_dir, "eval_miou_lenient_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
