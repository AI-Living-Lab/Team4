#!/bin/bash
# ============================================================
# SALMONN2+ 베이스 모델 (time-token 학습 안 됨) Baseline Eval 런처
#   - eval_salmonn2plus.sh의 변형:
#     · BASE_MODEL_ID 기본값 = video_salmonn2_plus_7B (vs _time_tokens)
#     · LoRA 미적용 (CKPT_STEP=base 강제)
#     · 결과 파싱은 자연어 "From X to Y" 패턴 지원하는
#       eval_miou_multiseg_natural.py 사용
#     · 테스트 JSON / 프롬프트는 수정 없이 그대로 사용
#
# 사용법:
#   bash eval_baseline_salmonn2plus.sh \
#       BASE_MODEL_ID=video_salmonn2_plus_7B \
#       TESTSET_FILE=unav100_test_sub80.json \
#       GPUS=0 \
#       CONFIG=config.yaml
#
# 결과 저장 위치: outputs/base/${BASE_MODEL_ID}/${TESTSET_NAME}/
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /workspace/setup.sh
conda activate salmonn2plus

export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

# ---- 기본값 ----
BASE_MODEL_ID=video_salmonn2_plus_7B
TESTSET_FILE=unav100_test_sub80.json
GPUS=0
CONFIG=config.yaml

# ---- KEY=VALUE 인자 파싱 ----
for arg in "$@"; do
    case "$arg" in
        BASE_MODEL_ID=*)    BASE_MODEL_ID="${arg#*=}" ;;
        TESTSET_FILE=*)     TESTSET_FILE="${arg#*=}" ;;
        GPUS=*)             GPUS="${arg#*=}" ;;
        CONFIG=*)           CONFIG="${arg#*=}" ;;
        *)
            echo "[에러] 지원하지 않는 인자: $arg"
            echo "지원 인자: BASE_MODEL_ID, TESTSET_FILE, GPUS, CONFIG"
            exit 1
            ;;
    esac
done

# ---- config.yaml 로드 ----
CONFIG_DIR="${SCRIPT_DIR}/${CONFIG}"
if [ ! -f "$CONFIG_DIR" ]; then
    echo "[에러] config 파일을 찾을 수 없음: $CONFIG_DIR"
    exit 1
fi

while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    line="${line%%#*}"
    [[ -z "${line// }" ]] && continue
    key="${line%%:*}"
    val="${line#*:}"
    key="$(echo "$key" | awk '{$1=$1;print}')"
    val="$(echo "$val" | awk '{$1=$1;print}')"
    val="${val#\"}"; val="${val%\"}"
    val="${val#\'}"; val="${val%\'}"
    [[ -z "$key" ]] && continue
    eval "$key=\"\$val\""
done < "$CONFIG_DIR"

# ---- GPU ----
NUM_GPUS=$(echo "$GPUS" | awk -F',' '{print NF}')
export CUDA_VISIBLE_DEVICES=$GPUS
export ARNOLD_WORKER_GPU=$NUM_GPUS

# ---- 경로 조립 ----
BASE_CODE="${BASE_DIR}/video_SALMONN2_plus"
MODEL_BASE="${CKPT_DIR}/base/${BASE_MODEL_ID}"
TEST_JSON="${BASE_DIR}/data/${TESTSET_FILE}"
EVAL_SCRIPT="${BASE_DIR}/eval/eval_miou_multiseg_natural.py"

if [ ! -d "$MODEL_BASE" ]; then
    echo "[에러] 베이스 모델 폴더 없음: $MODEL_BASE"
    exit 1
fi
if [ ! -f "$TEST_JSON" ]; then
    echo "[에러] 테스트 json 없음: $TEST_JSON"
    exit 1
fi

TESTSET_NAME="${TESTSET_FILE%.json}"
OUT_DIR="${BASE_DIR}/outputs/base/${BASE_MODEL_ID}/${TESTSET_NAME}"

# LoRA 미적용 강제
LORA_CKPT="No"

echo "=================================================="
echo "  [BASELINE EVAL — natural-language parser]"
echo "  BASE_MODEL_ID   : $BASE_MODEL_ID"
echo "  MODEL_BASE      : $MODEL_BASE"
echo "  LORA            : (none)"
echo "  TESTSET         : $TEST_JSON"
echo "  GPUS            : $GPUS  (count=$NUM_GPUS)"
echo "  CONFIG          : $CONFIG_DIR"
echo "  OUT_DIR         : $OUT_DIR"
echo "  EVAL_SCRIPT     : $EVAL_SCRIPT"
echo "=================================================="

mkdir -p "$OUT_DIR"

# 종료 시 generation_* 임시 폴더 정리
cleanup_gen() { rm -rf "$OUT_DIR"/generation_* 2>/dev/null || true; }
trap cleanup_gen EXIT

cd "$BASE_CODE"

FIRST_GPU=$(echo "$GPUS" | awk -F',' '{print $1}')
MASTER_PORT=$((12900 + FIRST_GPU))

# ---- 추론 ----
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
    2>&1 | tee "$OUT_DIR/inference.log"

# ---- 자연어 파서 평가 ----
RESULT_FILE=$(find "$OUT_DIR" -name "test_results_rank0.json" 2>/dev/null | head -1)
if [ ! -f "$RESULT_FILE" ]; then
    echo "[에러] 추론 결과 파일을 찾을 수 없습니다: $OUT_DIR/test_results_rank0.json"
    exit 1
fi

# 디버깅 편의: 파싱 실패 샘플 처음 10개 별도 저장
python3 "$EVAL_SCRIPT" \
    --results "$RESULT_FILE" \
    --test_json "$TEST_JSON" \
    --max_time "$MAX_TIME" \
    --out_dir "$OUT_DIR" \
    --dump_parse_fail 10 \
    2>&1 | tee -a "$OUT_DIR/inference.log"

echo "[완료] $OUT_DIR 에 평가 결과 저장 (eval_miou_summary.json)"
