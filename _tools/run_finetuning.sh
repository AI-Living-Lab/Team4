#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd $(dirname $0); pwd)
if [ -f "$SCRIPT_DIR/../paths.env" ]; then
    source "$SCRIPT_DIR/../paths.env"
else
    echo "[WARNING] paths.env not found. Copy paths.env.example to paths.env and fill in the paths."
fi

export PYTHONPATH=${BASE_DIR}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

MASTER_PORT_TRAIN=29501
MASTER_PORT_TEST=29502

BASE=${BASE_DIR}
MODEL_BASE=${BASE_MODEL}
BASE_CKPT=${SALMONN2_CKPT}

TRAIN_JSON=$BASE/data/unav100_train_dense.json
TEST_JSON=$BASE/data/unav100_test_dense_5.json

OUTDIR=$BASE/checkpoints
TEST_OUT=$BASE/output/test_full_sft3
LOGDIR=$BASE/tensor_logs_full_sft3

mkdir -p "$OUTDIR" "$TEST_OUT" "$LOGDIR"

# =========================================================
# Helper
# =========================================================
find_latest_checkpoint() {
  local dir="$1"
  ls -d "$dir"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true
}

# =========================================================
# TRAIN
# =========================================================
TRAIN_OUTPUT_DIR="$OUTDIR"

echo "===================="
echo "[1/2] TRAIN"
echo "train_json  = $TRAIN_JSON"
echo "base_ckpt   = $BASE_CKPT"
echo "model_base  = $MODEL_BASE"
echo "output_dir  = $TRAIN_OUTPUT_DIR"
echo "===================="

# resume 인자 조건부 설정 (핵심 수정 부분)
RESUME_ARG=""
EXISTING_CKPT=$(find_latest_checkpoint "$TRAIN_OUTPUT_DIR")
if [[ -n "$EXISTING_CKPT" ]]; then
  echo "[TRAIN] Found existing checkpoint: $EXISTING_CKPT → will resume"
  RESUME_ARG="--resume_from_checkpoint $EXISTING_CKPT"
else
  echo "[TRAIN] No existing checkpoint found → starting from scratch"
fi

torchrun --nproc_per_node=1 --master_port=${MASTER_PORT_TRAIN} \
  $BASE/llava/train/train.py \
  --version qwen_1_5 \
  --audio_visual True \
  --whisper_path openai/whisper-large-v3 \
  --freeze_whisper True \
  --freeze_backbone True \
  --window_level_Qformer True \
  --second_per_window 0.5 \
  --second_stride 0.5 \
  --video_fps 1 \
  --max_time 60 \
  --data_path "$TRAIN_JSON" \
  --model_base "$MODEL_BASE" \
  --ckpt "$BASE_CKPT" \
  --output_dir "$TRAIN_OUTPUT_DIR" \
  --bf16 True \
  --lora_enable True \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --evaluation_strategy "no" \
  --use_timestamps_crop False \
  --mm_spatial_pool_stride 4 \
  --model_max_length 4096 \
  --modality_max_length "[64,256,4096]" \
  --add_time_token True \
  --num_train_epochs 20 \
  --save_strategy "epoch" \
  --save_total_limit 4 \
  --gradient_checkpointing True \
  --mm_patch_merge_type spatial \
  --logging_strategy steps \
  --logging_steps 5 \
  --report_to tensorboard \
  --logging_dir "$LOGDIR" \
  --disable_tqdm False \
  $RESUME_ARG
