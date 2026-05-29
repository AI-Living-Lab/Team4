#!/usr/bin/env python3
"""Reassemble v5_b sub500 variant shards (with run_name-based subdir) and compute mIoU."""
import argparse
import json
import os
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="baseline / v1 / v3")
    ap.add_argument("--test_json", required=True, help="modified prompt JSON (for reassembly only)")
    ap.add_argument("--test_json_raw", default="/home/aix23102/audiolm/vS2_eunji/data/puvalor_test_v5_t1t2_sub500.json",
                    help="raw JSON with gt_segments")
    ap.add_argument("--results_base",
                    default="/data0/aix23102/checkpoints_open_aligner/salmonn2plus_v5_b/eval_sub500_variants",
                    help="parent results dir")
    ap.add_argument("--n_shards", type=int, default=4)
    args = ap.parse_args()

    rb = os.path.join(args.results_base, args.variant)
    N = args.n_shards

    # Inference saves at: {rb}/shard{r}/{variant}_s{r}/test_results_rank{r}.json
    shards = []
    for r in range(N):
        p = os.path.join(rb, f"shard{r}", f"{args.variant}_s{r}", f"test_results_rank{r}.json")
        if not os.path.exists(p):
            print(f"[ERROR] missing: {p}")
            return 1
        shards.append(json.load(open(p)))

    full = json.load(open(args.test_json))
    n = len(full)
    out = [None] * n
    for i in range(n):
        r = i % N
        pos = i // N
        if pos < len(shards[r]):
            out[i] = shards[r][pos]
    keep_idx = [i for i, x in enumerate(out) if x is not None]
    ordered_results = [out[i] for i in keep_idx]

    raw_full = json.load(open(args.test_json_raw))
    ordered_test = [raw_full[i] for i in keep_idx]

    json.dump(ordered_results, open(os.path.join(rb, "results_ordered.json"), "w"), ensure_ascii=False)
    json.dump(ordered_test, open(os.path.join(rb, "test_ordered.json"), "w"), ensure_ascii=False)
    print(f"[reassemble] {args.variant}: {len(ordered_results)} samples")

    # Run mIoU eval
    cmd = [
        "python3", "/home/aix23102/audiolm/vS2_eunji/eval/eval_miou_v5.py",
        "--results", os.path.join(rb, "results_ordered.json"),
        "--test_json", os.path.join(rb, "test_ordered.json"),
        "--max_time", "9999.0",
        "--out_dir", rb,
    ]
    print(f"[eval] {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    return rc

if __name__ == "__main__":
    sys.exit(main())
