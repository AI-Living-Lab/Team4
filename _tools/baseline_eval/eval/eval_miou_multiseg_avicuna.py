"""Avicuna × UnAV-100 eval — Union-IoU + FP_rate + FN_rate.

Parser: native `from XX to YY` pct (0-99). pct × duration/100 → seconds.
cdh 표준 인터페이스.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import score_sample, summarize, print_report


SEG_RE = re.compile(r"(?:from\s+)?(\d{1,3})\s*(?:to|and|\-|~)\s*(\d{1,3})", re.IGNORECASE)


def parse_pct_segments(raw: str):
    out = []
    for m in SEG_RE.finditer(raw):
        a, b = int(m.group(1)), int(m.group(2))
        a, b = min(a, 99), min(b, 99)
        if a < b:
            out.append((a, b))
    return out


def pct_to_sec(pa, pb, dur, max_time):
    s = pa * dur / 100.0
    e = pb * dur / 100.0
    if e < s: s, e = e, s
    s = max(0.0, min(s, max_time))
    e = max(0.0, min(e, max_time))
    return [s, e] if e > s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="JSONL (Avicuna 기존 predictions.jsonl)")
    ap.add_argument("--max_time", type=float, default=60.0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.predictions))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    all_ious = []
    total_preds = fp_preds = total_gts = fn_gts = 0
    parse_ok = parse_fail = 0
    n_samples = 0

    with open(args.predictions) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            n_samples += 1
            gt_segments = r.get("gt_segments", [])
            if not gt_segments: continue

            if r.get("error", "").startswith("feature missing") or r.get("error") == "no duration" or r.get("error", "").startswith("gen_error"):
                parse_fail += 1
                for _ in gt_segments:
                    all_ious.append(0.0)
                total_gts += len(gt_segments)
                fn_gts += len(gt_segments)
                continue

            dur = float(r["duration"])
            pcts = parse_pct_segments(r.get("raw", ""))
            if not pcts:
                parse_fail += 1
                for _ in gt_segments:
                    all_ious.append(0.0)
                total_gts += len(gt_segments)
                fn_gts += len(gt_segments)
                continue

            pred_segs = [pct_to_sec(a, b, dur, args.max_time) for a, b in pcts]
            pred_segs = [p for p in pred_segs if p is not None]
            if not pred_segs:
                parse_fail += 1
                for _ in gt_segments:
                    all_ious.append(0.0)
                total_gts += len(gt_segments)
                fn_gts += len(gt_segments)
                continue

            parse_ok += 1
            per_gt, n_fp, n_pred, n_fn, n_gt = score_sample(gt_segments, pred_segs)
            all_ious.extend(per_gt)
            total_preds += n_pred; fp_preds += n_fp
            total_gts += n_gt; fn_gts += n_fn

    summary = summarize(all_ious, total_preds, fp_preds, total_gts, fn_gts,
                        parse_ok, parse_fail, n_samples)
    summary["parser"] = "avicuna (from XX to YY pct, native)"
    print_report("Avicuna × UnAV-100 — Union-IoU", summary)

    out_path = os.path.join(out_dir, "eval_miou_summary_union.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
