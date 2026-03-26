#!/usr/bin/env bash
# ============================================================
# run_eval.sh
#   test_results.json 파싱 → UnAV-100 mAP 계산
#   run_inference.sh 실행 후 사용
# ============================================================
set -euo pipefail

CKPT=71379

BASE=/home/aix23102/audiolm/vS2_eunji
UNAV_JSON=/home/aix23102/audiolm/CCNet/data/unav_100/annotations/unav100_annotations.json

RESULTS=$BASE/eval/results/unav100_test_uf_$CKPT/test_results.json
OUT_DIR=$BASE/eval/results/unav100_test_uf_$CKPT

python $BASE/eval/parse_and_eval.py \
  --results      "$RESULTS"   \
  --unav_json    "$UNAV_JSON" \
  --split        test         \
  --max_time     60.0         \
  --out_dir      "$OUT_DIR"
