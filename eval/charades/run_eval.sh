#!/usr/bin/env bash
# ============================================================
# Charades-STA eval: parse predictions → R@1 at IoU 0.3/0.5/0.7
# ============================================================
set -euo pipefail

CKPT=${1:?Usage: $0 <checkpoint_step>}

BASE=/home/aix23102/audiolm/vS2_eunji/eval/charades
RESULTS=$BASE/results/charades_test_${CKPT}/test_results.json
OUT_DIR=$BASE/results/charades_test_${CKPT}

python $BASE/eval_charades.py \
  --results   "$RESULTS" \
  --test_json "$BASE/charades_sta_test.json" \
  --out_dir   "$OUT_DIR"
