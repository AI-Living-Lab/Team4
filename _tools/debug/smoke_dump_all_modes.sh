#!/bin/bash
# ============================================================
# 3개 TTI 모드 (off / special_token / natural_text) 전부 돌려서
# data/debug_interleave_samples.json 의 모든 샘플 dump.
#
# 출력: _debug_out/smoke_<mode>/NNN_<sample_tag>.{json,txt}
#
# 사용법:
#   bash _tools/debug/smoke_dump_all_modes.sh \
#     [BASE_MODEL=/workspace/checkpoints/base/video_salmonn2_plus_7B_time_tokens] \
#     [DATASET=data/debug_interleave_samples.json] \
#     [OUT_BASE=_debug_out] \
#     [SAMPLE_LIMIT=-1]   # -1 = 전체
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source /workspace/setup.sh
conda activate salmonn2plus

# ---- KEY=VALUE 인자 파싱 ----
BASE_MODEL="/workspace/checkpoints/base/video_salmonn2_plus_7B_time_tokens"
DATASET="data/debug_interleave_samples.json"
OUT_BASE="_debug_out"
SAMPLE_LIMIT="-1"

for arg in "$@"; do
    case "$arg" in
        BASE_MODEL=*)    BASE_MODEL="${arg#*=}" ;;
        DATASET=*)       DATASET="${arg#*=}" ;;
        OUT_BASE=*)      OUT_BASE="${arg#*=}" ;;
        SAMPLE_LIMIT=*)  SAMPLE_LIMIT="${arg#*=}" ;;
        *)
            echo "[에러] 지원하지 않는 인자: $arg"
            echo "지원: BASE_MODEL, DATASET, OUT_BASE, SAMPLE_LIMIT"
            exit 1
            ;;
    esac
done

cd "$BASE_DIR"

echo "=================================================="
echo "  BASE_MODEL    : $BASE_MODEL"
echo "  DATASET       : $DATASET"
echo "  OUT_BASE      : $OUT_BASE"
echo "  SAMPLE_LIMIT  : $SAMPLE_LIMIT"
echo "=================================================="

for MODE in off special_token natural_text; do
    OUT_DIR="${OUT_BASE}/smoke_${MODE}"
    echo ""
    echo "---- [mode=$MODE] -> $OUT_DIR ----"
    python _tools/debug/smoke_dump.py \
        --model_base "$BASE_MODEL" \
        --dataset_use "$DATASET" \
        --out_dir "$OUT_DIR" \
        --tti_time_format "$MODE" \
        --sample_limit "$SAMPLE_LIMIT"
done

echo ""
echo "=================================================="
echo "  완료. 결과 요약:"
echo "=================================================="
for MODE in off special_token natural_text; do
    DIR="${OUT_BASE}/smoke_${MODE}"
    n_json=$(ls "$DIR"/*.json 2>/dev/null | wc -l)
    n_txt=$(ls "$DIR"/*.txt 2>/dev/null | wc -l)
    echo "  ${DIR}  ${n_json} json / ${n_txt} txt"
done
