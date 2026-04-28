"""Avicuna × LongVALE eval — Union-IoU + FP_rate + FN_rate.

LongVALE 는 single-segment grounding (query 당 1 GT span). Union-IoU 는 Best-IoU 와 수학적으로
동일한 결과를 내나, 우리 baseline 전체 스키마 통일을 위해 score_sample 경유.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import score_sample, summarize, print_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    all_ious = []
    total_preds = fp_preds = total_gts = fn_gts = 0
    parse_ok = parse_fail = n_samples = 0
    n_feature_missing = n_gen_error = 0

    with open(args.predictions) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n_samples += 1

            if r.get("error") == "feature missing":
                n_feature_missing += 1
                parse_fail += 1
                all_ious.append(0.0)
                total_gts += 1; fn_gts += 1
                continue
            if r.get("error", "").startswith("gen_error"):
                n_gen_error += 1
                parse_fail += 1
                all_ious.append(0.0)
                total_gts += 1; fn_gts += 1
                continue

            ps_pct = r.get("pred_start_pct")
            pe_pct = r.get("pred_end_pct")
            dur = float(r["duration"])
            gs = max(0.0, min(dur, float(r["gt_start"])))
            ge = max(0.0, min(dur, float(r["gt_end"])))
            gt_segments = [[gs, ge]]

            if ps_pct is None or pe_pct is None:
                parse_fail += 1
                all_ious.append(0.0)
                total_gts += 1; fn_gts += 1
                continue

            parse_ok += 1
            ps = max(0.0, min(dur, ps_pct * dur / 100.0))
            pe = max(0.0, min(dur, pe_pct * dur / 100.0))
            if pe < ps:
                ps, pe = pe, ps
            pred_segs = [[ps, pe]] if pe > ps else []

            if not pred_segs:
                parse_fail += 1
                all_ious.append(0.0)
                total_gts += 1; fn_gts += 1
                parse_ok -= 1  # 사실은 parse_ok 로 카운트 말아야
                continue

            per_gt, n_fp, n_pred, n_fn, n_gt = score_sample(gt_segments, pred_segs)
            all_ious.extend(per_gt)
            total_preds += n_pred; fp_preds += n_fp
            total_gts += n_gt; fn_gts += n_fn

    summary = summarize(all_ious, total_preds, fp_preds, total_gts, fn_gts,
                        parse_ok, parse_fail, n_samples)
    summary["parser"] = "avicuna (pred_start_pct/pred_end_pct pre-parsed, single-seg)"
    summary["feature_missing"] = n_feature_missing
    summary["gen_error"] = n_gen_error
    print_report("Avicuna × LongVALE — Union-IoU (single-seg 동등)", summary)

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.predictions)), "eval_miou_summary.json"
    )
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
