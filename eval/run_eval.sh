#!/usr/bin/env bash
# ============================================================
# run_eval.sh
#   test_results.json 파싱 → UnAV-100 mAP 계산
#   run_inference.sh 실행 후 사용
# ============================================================
set -euo pipefail

# 경로 설정 로드
SCRIPT_DIR=$(cd $(dirname $0); pwd)
if [ -f "$SCRIPT_DIR/../paths.env" ]; then
    source "$SCRIPT_DIR/../paths.env"
else
    echo "[WARNING] paths.env not found. Copy paths.env.example to paths.env and fill in the paths."
fi

CKPT=71379

UNAV_JSON=${UNAV_ANNO}

RESULTS=${BASE_DIR}/eval/results/unav100_test_uf_$CKPT/test_results.json
OUT_DIR=${BASE_DIR}/eval/results/unav100_test_uf_$CKPT

python ${BASE_DIR}/eval/parse_and_eval.py \
  --results      "$RESULTS"   \
  --unav_json    "$UNAV_JSON" \
  --split        test         \
  --max_time     60.0         \
  --out_dir      "$OUT_DIR"
