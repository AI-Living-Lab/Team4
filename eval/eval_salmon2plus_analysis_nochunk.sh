#!/bin/bash
# ============================================================
# SALMONN2+ GDPO TTI(special) Eval — 단일 JSON 직접 평가 (청크 X)
#
#   결과 5파일:
#     test_results_rank0.json / eval_miou_summary.json / inference.log
#     segment_analysis.json / gtlen_analysis.json
#
#   기본 대상:
#     ckpt   /workspace/checkpoints/gdpo/sft_7b_puvalor_off_v2_rl_tti_rMfp_v2/checkpoint-100
#     base   /workspace/checkpoints/base/salmonn2p_7b_puvalor_off_v2
#     test   /workspace/data/test/unav100_v2_500.json   (TTI special_token)
#     out    /workspace/outputs/gdpo/<id>/checkpoint-100/fps5_tti/unav100_v2_500/
# ============================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /workspace/setup.sh
source "$SCRIPT_DIR/../paths.env"
conda activate "${CONDA_ENV:-salmonn2p}"

# 저장 루트: 팀 공용 /workspace/outputs (OUT_ROOT 로 변경 가능)
EVAL_DIR="${OUT_ROOT:-/workspace/outputs}"

export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

# ---- 이 런 고정값 (CLI override 가능) ----
STAGE=gdpo
CKPT_MODEL_ID=sft_7b_puvalor_off_v2_rl_tti_rMfp_v2
CKPT_STEP=100
BASE_MODEL_ID=base/salmonn2p_7b_puvalor_off_v2
TEST_JSON=/workspace/data/test/unav100_v2_500.json
TESTSET_TAG=""                       # 비면 TEST_JSON 파일명에서 자동
TTI_TIME_FORMAT_CLI=""               # 비면 config 값(special_token) 사용
GPUS=0
CONFIG=config.yaml

for arg in "$@"; do
    case "$arg" in
        STAGE=*)            STAGE="${arg#*=}" ;;
        CKPT_MODEL_ID=*)    CKPT_MODEL_ID="${arg#*=}" ;;
        CKPT_STEP=*)        CKPT_STEP="${arg#*=}" ;;
        BASE_MODEL_ID=*)    BASE_MODEL_ID="${arg#*=}" ;;
        TEST_JSON=*)        TEST_JSON="${arg#*=}" ;;
        TESTSET_TAG=*)      TESTSET_TAG="${arg#*=}" ;;
        TTI_TIME_FORMAT=*)  TTI_TIME_FORMAT_CLI="${arg#*=}" ;;
        GPUS=*)             GPUS="${arg#*=}" ;;
        CONFIG=*)           CONFIG="${arg#*=}" ;;
        *) echo "[에러] 지원하지 않는 인자: $arg"; exit 1 ;;
    esac
done

[ -f "$TEST_JSON" ] || { echo "[에러] TEST_JSON 없음: $TEST_JSON"; exit 1; }
[ -z "$TESTSET_TAG" ] && TESTSET_TAG=$(basename "$TEST_JSON" .json)

# ---- config.yaml 로드 (flat KEY: VALUE) ----
CONFIG_DIR="${SCRIPT_DIR}/${CONFIG}"
[ -f "$CONFIG_DIR" ] || { echo "[에러] config 없음: $CONFIG_DIR"; exit 1; }
while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"; line="${line%%#*}"
    [[ -z "${line// }" ]] && continue
    key="${line%%:*}"; val="${line#*:}"
    key="$(echo "$key" | awk '{$1=$1;print}')"
    val="$(echo "$val" | awk '{$1=$1;print}')"
    val="${val#\"}"; val="${val%\"}"; val="${val#\'}"; val="${val%\'}"
    [[ -z "$key" ]] && continue
    eval "$key=\"\$val\""
done < "$CONFIG_DIR"
[ -n "$TTI_TIME_FORMAT_CLI" ] && TTI_TIME_FORMAT="$TTI_TIME_FORMAT_CLI"

NUM_GPUS=$(echo "$GPUS" | awk -F',' '{print NF}')
export CUDA_VISIBLE_DEVICES=$GPUS
export ARNOLD_WORKER_GPU=$NUM_GPUS

# ---- EVAL_TAG = fps<N>_<format> ----
FPS_INT=$(awk -v b="${BASE_INTERVAL:-0.2}" 'BEGIN { printf "%d", (1.0/b)+0.5 }')
case "${TTI_TIME_FORMAT:-off}" in
    off) FORMAT_TAG=off ;; natural_text) FORMAT_TAG=natural ;;
    special_token) FORMAT_TAG=tti ;; from_to) FORMAT_TAG=fromto ;;
    *) FORMAT_TAG="${TTI_TIME_FORMAT:-off}" ;;
esac
EVAL_TAG="fps${FPS_INT}_${FORMAT_TAG}"

# ---- 경로 ----
BASE_CODE="${BASE_DIR}/video_SALMONN2_plus"
MODEL_BASE="${CKPT_DIR}/${BASE_MODEL_ID}"
if [[ "$CKPT_STEP" == checkpoint-* ]]; then CKPT_FOLDER="$CKPT_STEP"; else CKPT_FOLDER="checkpoint-${CKPT_STEP}"; fi
LORA_CKPT="${CKPT_DIR}/${STAGE}/${CKPT_MODEL_ID}/${CKPT_FOLDER}"
[ -f "$LORA_CKPT/adapter_config.json" ] || { echo "[에러] 어댑터 없음: $LORA_CKPT"; exit 1; }

OUT_DIR="${EVAL_DIR}/${STAGE}/${CKPT_MODEL_ID}/${CKPT_FOLDER}/${EVAL_TAG}/${TESTSET_TAG}"
mkdir -p "$OUT_DIR"
RESULTS="$OUT_DIR/test_results_rank0.json"
INFER_LOG="$OUT_DIR/inference.log"

cat <<EOF
==================================================
  단일 JSON eval (no-chunk)
  STAGE/MODEL    : $STAGE / $CKPT_MODEL_ID / $CKPT_FOLDER
  LORA_CKPT      : $LORA_CKPT
  MODEL_BASE     : $MODEL_BASE
  TEST_JSON      : $TEST_JSON  (tag=$TESTSET_TAG)
  EVAL_TAG       : $EVAL_TAG  (BASE_INTERVAL=$BASE_INTERVAL, TTI=${TTI_TIME_FORMAT:-off})
  GPUS           : $GPUS  (count=$NUM_GPUS)
  OUT_DIR        : $OUT_DIR
  START          : $(date -Iseconds)
==================================================
EOF

# ---- [1/4] 추론 (단일 torchrun) ----
cd "$BASE_CODE"
FIRST_GPU=$(echo "$GPUS" | awk -F',' '{print $1}')
MASTER_PORT=$((12900 + FIRST_GPU))
echo "[1/4] inference (greedy, tti=${TTI_TIME_FORMAT:-off})" | tee "$INFER_LOG"

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    qwenvl/train/train_qwen.py \
    --model_base "$MODEL_BASE" \
    --run_test True \
    --pred_rank 0 \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path "$MODEL_BASE" \
    --dataset_use "$TEST_JSON" \
    --bf16 \
    --output_dir "$OUT_DIR" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --max_pixels "$MAX_PIXELS" \
    --min_pixels "$MIN_PIXELS" \
    --video_max_frame_pixels "$VIDEO_MAX_FRAME_PIXELS" \
    --video_min_frame_pixels "$VIDEO_MIN_FRAME_PIXELS" \
    --eval_strategy "$EVAL_STRATEGY" \
    --save_strategy "$SAVE_STRATEGY" \
    --learning_rate "$LEARNING_RATE" \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --run_name "." \
    --report_to "$REPORT_TO" \
    --video_min_frames "$VIDEO_MIN_FRAMES" \
    --video_max_frames "$VIDEO_MAX_FRAMES" \
    --base_interval "$BASE_INTERVAL" \
    --lora_ckpt "$LORA_CKPT" \
    --no_audio "$NO_AUDIO" \
    --tti_time_format "${TTI_TIME_FORMAT:-off}" \
    2>&1 | tee -a "$INFER_LOG"

[ -f "$RESULTS" ] || { echo "[에러] 결과 없음: $RESULTS"; exit 1; }

# LoRA merge 산출물(generation_0, 수 GB) 정리
rm -rf "$OUT_DIR/generation_0" 2>/dev/null || true

# ---- [2/4] mIoU/R@1 ----
echo "[2/4] mIoU/R@1 → eval_miou_summary.json"
python3 "${BASE_DIR}/eval/eval_miou_multiseg.py" \
    --results "$RESULTS" --test_json "$TEST_JSON" \
    --max_time "$MAX_TIME" --out_dir "$OUT_DIR"

# ---- [3/4] segment 분석 ----
echo "[3/4] segment 분석 → segment_analysis.json"
python3 "${SCRIPT_DIR}/analyze_segments.py" \
    --results "$RESULTS" --test_json "$TEST_JSON" \
    --max_time "$MAX_TIME" --fp_iou_thr 0.3 --label "$CKPT_MODEL_ID" \
    --out "$OUT_DIR/segment_analysis.json" || echo "[warn] segment 분석 실패(무시)"

# ---- [4/4] GT길이별 R@0.3 분해 ----
echo "[4/4] gtlen 분해 → gtlen_analysis.json"
python3 "${SCRIPT_DIR}/posteval_gtlen.py" \
    --results "$RESULTS" --test_json "$TEST_JSON" \
    --out "$OUT_DIR/gtlen_analysis.json" --label "$CKPT_MODEL_ID" || echo "[warn] gtlen 분해 실패(무시)"

echo ""
echo "[완료] $OUT_DIR"
ls -la "$OUT_DIR"
