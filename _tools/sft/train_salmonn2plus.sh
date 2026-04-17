#!/bin/bash
# ============================================================
# SALMONN2+ UnAV-100 fine-tuning
#   - Data: UnAV-100 multi-segment QA
#   - Hyperparameters: see config.yaml (override with CONFIG=...)
#
# Usage (all args optional, KEY=VALUE style):
#   bash train_salmonn2plus.sh \
#       MODEL_ID=salmonn2p7b_unav100_baseline \
#       DATASET_NAME=unav100_train_multiseg_salmonn2plus.json \
#       BASE_MODEL_ID=video_salmonn2_plus_7B_time_tokens \
#       GPUS=0,1,2 \
#       CONFIG=config.yaml
#
# Defaults are applied when args are omitted.
# GPUS accepts comma-separated ids (e.g. "0" or "0,1,2,3"); count is auto-derived.
# CONFIG defaults to ./config.yaml (next to this script).
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /workspace/setup.sh
conda activate salmonn2plus

# ---- Defaults ----
MODEL_ID=salmonn2p7b_unav100_baseline
DATASET_NAME=unav100_train_multiseg_salmonn2plus.json
BASE_MODEL_ID=video_salmonn2_plus_7B_time_tokens
GPUS=0
CONFIG=config.yaml

# ---- Parse KEY=VALUE args ----
for arg in "$@"; do
    case "$arg" in
        MODEL_ID=*)         MODEL_ID="${arg#*=}" ;;
        DATASET_NAME=*)     DATASET_NAME="${arg#*=}" ;;
        BASE_MODEL_ID=*)  BASE_MODEL_ID="${arg#*=}" ;;
        GPUS=*)             GPUS="${arg#*=}" ;;
        CONFIG=*)           CONFIG="${arg#*=}" ;;
        *)
            echo "[ERROR] Unknown argument: $arg"
            echo "Supported: MODEL_ID, DATASET_NAME, BASE_MODEL_ID, GPUS, CONFIG"
            exit 1
            ;;
    esac
done

CONFIG_DIR="${SCRIPT_DIR}/${CONFIG}"

# ---- Load YAML config (flat KEY: VALUE only) ----
if [ ! -f "$CONFIG_DIR" ]; then
    echo "[ERROR] Config file not found: $CONFIG_DIR"
    exit 1
fi

while IFS= read -r line || [ -n "$line" ]; do
    # strip CR (Windows line endings) and inline comments
    line="${line%$'\r'}"
    line="${line%%#*}"
    # skip blank lines
    [[ -z "${line// }" ]] && continue
    # split on first ':'
    key="${line%%:*}"
    val="${line#*:}"
    # trim whitespace
    key="$(echo "$key" | awk '{$1=$1;print}')"
    val="$(echo "$val" | awk '{$1=$1;print}')"
    # strip surrounding quotes
    val="${val#\"}"; val="${val%\"}"
    val="${val#\'}"; val="${val%\'}"
    [[ -z "$key" ]] && continue
    eval "$key=\"\$val\""
done < "$CONFIG_DIR"

# ---- Derive GPU count ----
NUM_GPUS=$(echo "$GPUS" | awk -F',' '{print NF}')

export CUDA_VISIBLE_DEVICES=$GPUS
export ARNOLD_WORKER_GPU=$NUM_GPUS
export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

cd "${BASE_DIR}/video_SALMONN2_plus"

MODEL=${CKPT_DIR}/${BASE_MODEL_ID}
MODEL_BASE=${CKPT_DIR}/${BASE_MODEL_ID}
DATASET=${JSON_DIR}/${DATASET_NAME}
MODEL_DIR=${CKPT_DIR}/${MODEL_ID}

echo "=================================================="
echo "  CONFIG          : $CONFIG_DIR"
echo "  MODEL_ID        : $MODEL_ID"
echo "  DATASET_NAME    : $DATASET_NAME"
echo "  BASE_MODEL_ID : $BASE_MODEL_ID"
echo "  GPUS            : $GPUS  (count=$NUM_GPUS)"
echo "  MODEL           : $MODEL"
echo "  DATASET         : $DATASET"
echo "  MODEL_DIR       : $MODEL_DIR"
echo "=================================================="

mkdir -p "$MODEL_DIR"
cp "$CONFIG_DIR" "$MODEL_DIR/config.used.yaml"

torchrun --standalone --nproc_per_node=$ARNOLD_WORKER_GPU \
    qwenvl/train/train_qwen.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path "$MODEL" \
    --model_base "$MODEL_BASE" \
    --dataset_use "$DATASET" \
    --tune_mm_vision "$TUNE_MM_VISION" \
    --tune_mm_mlp "$TUNE_MM_MLP" \
    --tune_mm_llm "$TUNE_MM_LLM" \
    --tune_mm_audio "$TUNE_MM_AUDIO" \
    --tune_mm_qformer "$TUNE_MM_QFORMER" \
    --tune_lm_head "$TUNE_LM_HEAD" \
    --use_lora "$USE_LORA" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --bf16 \
    --output_dir "$MODEL_DIR" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --max_pixels "$MAX_PIXELS" \
    --min_pixels "$MIN_PIXELS" \
    --video_max_frame_pixels "$VIDEO_MAX_FRAME_PIXELS" \
    --video_min_frame_pixels "$VIDEO_MIN_FRAME_PIXELS" \
    --eval_strategy "$EVAL_STRATEGY" \
    --save_strategy "$SAVE_STRATEGY" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_ratio "$WARMUP_RATIO" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --logging_steps "$LOGGING_STEPS" \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --run_name "$MODEL_ID" \
    --report_to "$REPORT_TO" \
    --video_min_frames "$VIDEO_MIN_FRAMES" \
    --video_max_frames "$VIDEO_MAX_FRAMES" \
    --base_interval "$BASE_INTERVAL" \
    --train_type "$TRAIN_TYPE" \
    --no_audio "$NO_AUDIO" \
    2>&1 | tee -a "$MODEL_DIR/train.log"
