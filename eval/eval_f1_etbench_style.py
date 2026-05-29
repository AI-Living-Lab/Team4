"""
ETBench-style F1@thr metric for multi-pred × multi-gt temporal grounding.

Per-sample F1 (ETBench tal_eval):
  iou = IoU_matrix(gt[G], pred[P])  # shape [G, P]
  rec = (iou.amax(dim=1) >= thr).float().mean()   # % GTs covered
  prc = (iou.amax(dim=0) >= thr).float().mean()   # % preds matched
  F1  = 2*P*R / (P+R)                              # 0 if iou.max() < thr
Aggregate: F1@thr = mean(F1_sample) over ALL samples (failed → 0).

Parses predictions using parse_multi_segments from eval_miou_v5 (token-based,
"From <ts> to <te>." pattern).
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_miou_v5 import parse_multi_segments


def temporal_iou_matrix(gt, pred):
    """gt: [G,2], pred: [P,2] → iou: [G,P]"""
    G = gt.shape[0]
    P = pred.shape[0]
    g = gt.unsqueeze(1).expand(G, P, 2)
    p = pred.unsqueeze(0).expand(G, P, 2)
    inter_s = torch.maximum(g[..., 0], p[..., 0])
    inter_e = torch.minimum(g[..., 1], p[..., 1])
    inter = torch.clamp(inter_e - inter_s, min=0.0)
    union = (g[..., 1] - g[..., 0]) + (p[..., 1] - p[..., 0]) - inter
    return inter / (union + 1e-8)


def compute_f1_thr(results, test, thresholds, max_time=9999.0):
    n = min(len(results), len(test))
    per_thr_sum = {t: 0.0 for t in thresholds}
    cnt_failed = 0

    for i in range(n):
        gt_segs = test[i].get("gt_segments", [])
        if not gt_segs:
            continue
        pred_str = results[i].get("pred", "")
        pred_segs = parse_multi_segments(pred_str, max_time)

        if not pred_segs:
            cnt_failed += 1
            continue

        gt = torch.tensor(gt_segs, dtype=torch.float32)
        pr = torch.tensor(pred_segs, dtype=torch.float32)
        iou = temporal_iou_matrix(gt, pr)

        max_iou = iou.max().item()
        for thr in thresholds:
            if max_iou < thr:
                continue
            rec = (iou.amax(dim=1) >= thr).float().mean().item()
            prc = (iou.amax(dim=0) >= thr).float().mean().item()
            if rec + prc == 0:
                continue
            per_thr_sum[thr] += 2 * prc * rec / (prc + rec)

    f1_per_thr = {t: per_thr_sum[t] / n for t in thresholds}
    f1_mean = sum(f1_per_thr.values()) / len(thresholds)
    return dict(
        n_samples=n,
        n_parse_failed=cnt_failed,
        F1_per_thr={f"F1@{t}": round(v * 100, 4) for t, v in f1_per_thr.items()},
        F1_mean_pct=round(f1_mean * 100, 4),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--max_time", type=float, default=9999.0)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--thresholds", default="0.1,0.3,0.5,0.7,0.9")
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",")]
    results = json.load(open(args.results))
    test = json.load(open(args.test_json))
    out = compute_f1_thr(results, test, thresholds, max_time=args.max_time)

    print(f"[etbench_f1] n={out['n_samples']}, parse_failed={out['n_parse_failed']}")
    for k, v in out["F1_per_thr"].items():
        print(f"  {k}: {v:.4f}%")
    print(f"  F1_mean: {out['F1_mean_pct']:.4f}%")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "eval_f1_etbench_style.json"), "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
