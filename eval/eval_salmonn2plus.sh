#!/bin/bash
# ============================================================
# SALMONN2+ Eval 런처 (chunked 추론 + 자동 resume)
#
# 동작
#   ${BASE_DIR}/data/test/${TESTSET}/chunk_*.json 들을 순회하며 chunk 마다
#   torchrun 추론 → master 에 append → eval_miou_summary.json 갱신.
#   chunk 0 의 LoRA merge 결과(generation_0)는 .merged_model/ 로 보존하여 재사용.
#
# 사전 준비 (한 번만)
#   python3 ${BASE_DIR}/eval/_chunk_helpers.py split \
#       --test_json ${BASE_DIR}/data/<원본>.json \
#       --chunks_dir ${BASE_DIR}/data/test/<TESTSET>/ \
#       --chunk_size 500
#
# 자동 resume
#   $OUT_DIR/.chunk_idx 또는 master test_results_rank0.json 길이로 다음
#   chunk index 추정 → 그 다음 chunk 부터 이어서 진행. (같은 OUT_DIR 에서
#   중단된 후 같은 명령 재실행하면 됨)
#
# 사용법 (KEY=VALUE)
#   bash eval_salmonn2plus.sh \
#       STAGE=sft CKPT_MODEL_ID=salmonn2p_7b_unav_baseline CKPT_STEP=1500 \
#       BASE_MODEL_ID=base/video_salmonn2_plus_7B_time_tokens \
#       TESTSET=unav100 GPUS=0
#
# 결과 경로 (EVAL_TAG = fps<round(1/BASE_INTERVAL)>_<format> ; format ∈ off/natural/tti)
#   $OUT_DIR = outputs/<stage>/<model>/<CKPT_FOLDER>/<EVAL_TAG>/<TESTSET>/  (LoRA)
#            = outputs/base/<base_model>/<EVAL_TAG>/<TESTSET>/                 (CKPT_STEP=base)
#   예) outputs/sft/salmonn2p_7b_unav_fps10_tti/checkpoint-100/fps10_tti/unav100/
#     test_results_rank0.json    : master, chunk 별 append
#     eval_miou_summary.json     : 매 chunk 후 갱신
#     eval_miou_progress.jsonl   : chunk 별 timestamp + metrics 누적
#     inference.log              : chunk 별 stdout 누적
#     .chunk_idx, .chunk_workdir/, .merged_model/ : 임시 (성공시 자동 정리)
# ============================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /workspace/setup.sh
conda activate salmonn2plus
source "$SCRIPT_DIR/../paths.env"

export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

# ---- 기본값 ----
STAGE=sft
CKPT_MODEL_ID=salmonn2p_7b_unav_baseline
CKPT_STEP=
BASE_MODEL_ID=video_salmonn2_plus_7B_time_tokens
TESTSET=unav100
GPUS=0
CONFIG=config.yaml
TTI_TIME_FORMAT_CLI=""   # 비어있으면 config.yaml 값 사용

# ---- KEY=VALUE 파싱 ----
for arg in "$@"; do
    case "$arg" in
        STAGE=*)            STAGE="${arg#*=}" ;;
        CKPT_MODEL_ID=*)    CKPT_MODEL_ID="${arg#*=}" ;;
        CKPT_STEP=*)        CKPT_STEP="${arg#*=}" ;;
        BASE_MODEL_ID=*)    BASE_MODEL_ID="${arg#*=}" ;;
        TESTSET=*)          TESTSET="${arg#*=}" ;;
        GPUS=*)             GPUS="${arg#*=}" ;;
        CONFIG=*)           CONFIG="${arg#*=}" ;;
        TTI_TIME_FORMAT=*)  TTI_TIME_FORMAT_CLI="${arg#*=}" ;;
        *)
            echo "[에러] 지원하지 않는 인자: $arg"
            echo "지원 인자: STAGE, CKPT_MODEL_ID, CKPT_STEP, BASE_MODEL_ID, TESTSET, GPUS, CONFIG, TTI_TIME_FORMAT"
            exit 1
            ;;
    esac
done

STAGE=$(echo "$STAGE" | tr '[:upper:]' '[:lower:]')
if [ "$STAGE" != "sft" ] && [ "$STAGE" != "gdpo" ]; then
    echo "[에러] STAGE는 sft|gdpo 여야 합니다 (받은 값: $STAGE)"
    exit 1
fi

# ---- config.yaml 로드 (flat KEY: VALUE) ----
CONFIG_DIR="${SCRIPT_DIR}/${CONFIG}"
if [ ! -f "$CONFIG_DIR" ]; then
    echo "[에러] config 파일 없음: $CONFIG_DIR"
    exit 1
fi
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

# ---- CLI override (YAML 보다 우선) ----
if [ -n "$TTI_TIME_FORMAT_CLI" ]; then
    TTI_TIME_FORMAT="$TTI_TIME_FORMAT_CLI"
fi

NUM_GPUS=$(echo "$GPUS" | awk -F',' '{print NF}')
export CUDA_VISIBLE_DEVICES=$GPUS
export ARNOLD_WORKER_GPU=$NUM_GPUS

# ---- EVAL_TAG = fps<N>_<format>  (config.yaml 기반 자동 생성) ----
FPS_INT=$(awk -v b="${BASE_INTERVAL:-0.2}" 'BEGIN { printf "%d", (1.0/b)+0.5 }')
case "${TTI_TIME_FORMAT:-off}" in
    off)            FORMAT_TAG=off ;;
    natural_text)   FORMAT_TAG=natural ;;
    special_token)  FORMAT_TAG=tti ;;
    *)              FORMAT_TAG="${TTI_TIME_FORMAT:-off}" ;;
esac
EVAL_TAG="fps${FPS_INT}_${FORMAT_TAG}"

# ---- 경로 ----
BASE_CODE="${BASE_DIR}/video_SALMONN2_plus"
MODEL_BASE="${CKPT_DIR}/${BASE_MODEL_ID}"
MODEL_DIR="${CKPT_DIR}/${STAGE}/${CKPT_MODEL_ID}"
EVAL_SCRIPT="${BASE_DIR}/eval/eval_miou_multiseg.py"
HELPER="${BASE_DIR}/eval/_chunk_helpers.py"
CHUNKS_DIR="${TEST_DIR}/${TESTSET}"

# ---- chunks 검증 ----
if [ ! -d "$CHUNKS_DIR" ]; then
    echo "[에러] testset 디렉토리 없음: $CHUNKS_DIR"
    echo "  먼저: python3 $HELPER split --test_json <원본.json> --chunks_dir $CHUNKS_DIR --chunk_size 500"
    exit 1
fi
mapfile -t CHUNK_FILES < <(ls "$CHUNKS_DIR"/chunk_*.json 2>/dev/null | sort)
N_CHUNKS=${#CHUNK_FILES[@]}
if [ "$N_CHUNKS" -le 0 ]; then
    echo "[에러] $CHUNKS_DIR 에 chunk_*.json 없음"
    exit 1
fi

# eval_miou 가 사용할 _full.json (없으면 1회 생성)
TEST_JSON="${CHUNKS_DIR}/_full.json"
N_TOTAL=$(python3 "$HELPER" build_full --chunks_dir "$CHUNKS_DIR" | awk '{print $1}')

# ---- CKPT_STEP 해석 ----
CKPT_STEP_LOWER=$(echo "$CKPT_STEP" | tr '[:upper:]' '[:lower:]')
IS_BASE_EVAL=false
if [ -z "$CKPT_STEP" ]; then
    LORA_CKPT=$(ls -d "$MODEL_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
    if [ -z "$LORA_CKPT" ]; then
        echo "[에러] $MODEL_DIR 에 checkpoint-* 없음"
        exit 1
    fi
    CKPT_FOLDER=$(basename "$LORA_CKPT")
elif [ "$CKPT_STEP_LOWER" = "base" ] || [ "$CKPT_STEP_LOWER" = "no" ]; then
    LORA_CKPT="No"
    CKPT_FOLDER="base"
    IS_BASE_EVAL=true
else
    if [[ "$CKPT_STEP" == checkpoint-* ]]; then
        CKPT_FOLDER="$CKPT_STEP"
    else
        CKPT_FOLDER="checkpoint-${CKPT_STEP}"
    fi
    LORA_CKPT="${MODEL_DIR}/${CKPT_FOLDER}"
    if [ ! -d "$LORA_CKPT" ]; then
        echo "[에러] 체크포인트 폴더 없음: $LORA_CKPT"
        exit 1
    fi
fi

if [ "$IS_BASE_EVAL" = "true" ]; then
    OUT_DIR="${EVAL_DIR}/base/${BASE_MODEL_ID}/${EVAL_TAG}/${TESTSET}"
else
    OUT_DIR="${EVAL_DIR}/${STAGE}/${CKPT_MODEL_ID}/${CKPT_FOLDER}/${EVAL_TAG}/${TESTSET}"
fi
mkdir -p "$OUT_DIR"

MASTER_RESULT="$OUT_DIR/test_results_rank0.json"
SUMMARY_FILE="$OUT_DIR/eval_miou_summary.json"
PROGRESS_LOG="$OUT_DIR/eval_miou_progress.jsonl"
INFER_LOG="$OUT_DIR/inference.log"
CHUNK_IDX_FILE="$OUT_DIR/.chunk_idx"
CHUNK_WORKDIR="$OUT_DIR/.chunk_workdir"
MERGED_MODEL_DIR="$OUT_DIR/.merged_model"

# ---- resume offset ----
RESUME_OUT=$(python3 "$HELPER" resume_offset \
    --master "$MASTER_RESULT" \
    --chunks_dir "$CHUNKS_DIR" \
    --chunk_idx_file "$CHUNK_IDX_FILE")
read START_CHUNK N_MASTER EXPECTED_N <<<"$RESUME_OUT"

# master 가 chunk 경계와 어긋나면 잘라냄
if [ "$N_MASTER" -gt "$EXPECTED_N" ]; then
    python3 "$HELPER" truncate_master --master "$MASTER_RESULT" --keep "$EXPECTED_N" >/dev/null
    echo "[CLEAN] truncated master $N_MASTER -> $EXPECTED_N (chunk-aligned)"
    N_MASTER=$EXPECTED_N
fi

cat <<EOF
==================================================
  STAGE             : $STAGE
  CKPT_MODEL_ID     : $CKPT_MODEL_ID
  CKPT_STEP         : ${CKPT_STEP:-<latest>}  ->  $CKPT_FOLDER
  LORA_CKPT         : $LORA_CKPT
  BASE_MODEL_ID     : $BASE_MODEL_ID
  EVAL_TAG          : $EVAL_TAG  (BASE_INTERVAL=$BASE_INTERVAL → fps=$FPS_INT, TTI=${TTI_TIME_FORMAT:-off})
  TESTSET           : $TESTSET  ($N_CHUNKS chunks, $N_TOTAL samples)
  GPUS              : $GPUS  (count=$NUM_GPUS)
  OUT_DIR           : $OUT_DIR
  START_CHUNK       : $START_CHUNK  (master n=$N_MASTER)
==================================================
EOF

# ---- 종료 시 정리 ----
SUCCESS=0
cleanup_all() {
    rm -rf "$CHUNK_WORKDIR" 2>/dev/null || true
    if [ "$SUCCESS" = "1" ]; then
        rm -rf "$MERGED_MODEL_DIR" 2>/dev/null || true
        rm -f "$CHUNK_IDX_FILE" 2>/dev/null || true
    fi
}
trap cleanup_all EXIT

# 모든 chunk 끝났으면 최종 평가만
if [ "$START_CHUNK" -ge "$N_CHUNKS" ]; then
    echo "[SKIP] all $N_CHUNKS chunks already done; running final eval only"
    python3 "$EVAL_SCRIPT" \
        --results "$MASTER_RESULT" --test_json "$TEST_JSON" \
        --max_time "$MAX_TIME" --out_dir "$OUT_DIR" --progress_log "$PROGRESS_LOG"
    SUCCESS=1
    echo "[완료] $OUT_DIR"
    exit 0
fi

cd "$BASE_CODE"
FIRST_GPU=$(echo "$GPUS" | awk -F',' '{print $1}')
MASTER_PORT=$((12900 + FIRST_GPU))

# ---- chunk loop ----
for ((i=START_CHUNK; i<N_CHUNKS; i++)); do
    chunk_json="${CHUNK_FILES[$i]}"
    chunk_id=$(basename "$chunk_json" .json | sed 's/^chunk_//')

    # merged_model 이 있으면 재사용, 없으면 LoRA merge
    if [ -d "$MERGED_MODEL_DIR" ] && [ "$IS_BASE_EVAL" != "true" ]; then
        _model_name="$MERGED_MODEL_DIR"
        _lora_ckpt="No"
    else
        _model_name="$MODEL_BASE"
        _lora_ckpt="$LORA_CKPT"
    fi

    rm -rf "$CHUNK_WORKDIR"
    mkdir -p "$CHUNK_WORKDIR"

    header="
=== [CHUNK $((i+1))/$N_CHUNKS] id=$chunk_id  start=$(date -Iseconds) ==="
    echo "$header"
    echo "$header" >> "$INFER_LOG"

    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        qwenvl/train/train_qwen.py \
        --model_base "$MODEL_BASE" \
        --run_test True \
        --pred_rank 0 \
        --deepspeed "$DEEPSPEED_CONFIG" \
        --model_name_or_path "$_model_name" \
        --dataset_use "$chunk_json" \
        --bf16 \
        --output_dir "$CHUNK_WORKDIR" \
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
        --lora_ckpt "$_lora_ckpt" \
        --no_audio "$NO_AUDIO" \
        --tti_time_format "${TTI_TIME_FORMAT:-off}" \
        2>&1 | tee -a "$INFER_LOG"

    chunk_result="$CHUNK_WORKDIR/test_results_rank0.json"
    if [ ! -f "$chunk_result" ]; then
        echo "[에러] chunk $chunk_id 결과 파일 없음: $chunk_result"
        exit 1
    fi

    # LoRA merge 결과를 처음 만든 chunk 라면 .merged_model 로 보존
    if [ "$IS_BASE_EVAL" != "true" ] && [ ! -d "$MERGED_MODEL_DIR" ] && [ -d "$CHUNK_WORKDIR/generation_0" ]; then
        mv "$CHUNK_WORKDIR/generation_0" "$MERGED_MODEL_DIR"
        echo "[MERGE] preserved -> $MERGED_MODEL_DIR"
    fi

    # master append + chunk_idx 업데이트
    AGG_OUT=$(python3 "$HELPER" append --master "$MASTER_RESULT" --chunk_results "$chunk_result")
    read N_AFTER N_ADDED <<<"$AGG_OUT"
    echo "$((i + 1))" > "$CHUNK_IDX_FILE.tmp" && mv "$CHUNK_IDX_FILE.tmp" "$CHUNK_IDX_FILE"

    # summary 갱신 + 한 줄 요약
    python3 "$EVAL_SCRIPT" \
        --results "$MASTER_RESULT" --test_json "$TEST_JSON" \
        --max_time "$MAX_TIME" --out_dir "$OUT_DIR" \
        --progress_log "$PROGRESS_LOG" --quiet
    echo "[CHUNK $((i+1))/$N_CHUNKS] done  master_n=$N_AFTER  added=$N_ADDED  $(date -Iseconds)"
    python3 "$HELPER" summary_oneline --summary "$SUMMARY_FILE"
done

# ---- 최종 평가 (loud) ----
echo ""
echo "=== FINAL EVAL  $(date -Iseconds) ==="
python3 "$EVAL_SCRIPT" \
    --results "$MASTER_RESULT" --test_json "$TEST_JSON" \
    --max_time "$MAX_TIME" --out_dir "$OUT_DIR" --progress_log "$PROGRESS_LOG"

SUCCESS=1
echo "[완료] $OUT_DIR"
