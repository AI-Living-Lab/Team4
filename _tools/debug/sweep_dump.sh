#!/bin/bash
# ============================================================
# TTI debug interleave — config sweep (모델 로드 없이 dump 만)
#
#   BASE_INTERVAL × VIDEO_MAX_FRAMES 조합별로 _tools/debug/smoke_dump.py 호출.
#   결과: <OUT_BASE>/<config_tag>/NNN_<sample_tag>.{json,txt}
#
# 사용법:
#   bash _tools/debug/sweep_dump.sh \
#     [BASE_MODEL=/workspace/checkpoints/base/video_salmonn2_plus_7B_time_tokens] \
#     [DATASET=data/debug_interleave_samples.json] \
#     [OUT_BASE=_debug_out/sweep] \
#     [INTERVALS="0.1 0.2 0.5 1.0"] \
#     [MAX_FRAMES="64 128 256"] \
#     [SAMPLE_LIMIT=-1]
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source /workspace/setup.sh
conda activate salmonn2plus

BASE_MODEL="${BASE_MODEL:-/workspace/checkpoints/base/video_salmonn2_plus_7B_time_tokens}"
DATASET="${DATASET:-data/debug_interleave_samples.json}"
OUT_BASE="${OUT_BASE:-_debug_out/sweep}"
INTERVALS="${INTERVALS:-0.1 0.2 0.5 1.0}"
MAX_FRAMES="${MAX_FRAMES:-64 128 256}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:--1}"
TTI_TIME_FORMAT="${TTI_TIME_FORMAT:-off}"

cd "$BASE_DIR"

echo "=================================================="
echo "  BASE_MODEL       : $BASE_MODEL"
echo "  DATASET          : $DATASET"
echo "  OUT_BASE         : $OUT_BASE"
echo "  INTERVALS        : $INTERVALS"
echo "  MAX_FRAMES       : $MAX_FRAMES"
echo "  SAMPLE_LIMIT     : $SAMPLE_LIMIT"
echo "  TTI_TIME_FORMAT  : $TTI_TIME_FORMAT"
echo "=================================================="

for INTERVAL in $INTERVALS; do
    for MAXF in $MAX_FRAMES; do
        TAG="interval${INTERVAL}_maxf${MAXF}"
        OUT_DIR="${OUT_BASE}/${TAG}"
        echo ""
        echo "---- [sweep] $TAG -> $OUT_DIR ----"
        python _tools/debug/smoke_dump.py \
            --model_base "$BASE_MODEL" \
            --dataset_use "$DATASET" \
            --out_dir "$OUT_DIR" \
            --base_interval "$INTERVAL" \
            --video_max_frames "$MAXF" \
            --sample_limit "$SAMPLE_LIMIT" \
            --tti_time_format "$TTI_TIME_FORMAT" \
            2>&1 | tail -n $((8 + 2))   # size + "... ok" lines
    done
done

echo ""
echo "[sweep] 완료. 결과: $OUT_BASE/*/"
ls -d "$OUT_BASE"/*/ 2>/dev/null | while read d; do
    n=$(ls "$d"*.json 2>/dev/null | wc -l)
    echo "  $d  ($n samples)"
done
