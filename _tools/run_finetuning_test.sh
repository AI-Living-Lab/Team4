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

# 기존 폴더 말고 새 폴더 사용
OUTDIR=$BASE/checkpoints_open_aligner
TEST_OUT=$BASE/output/test_open_aligner
LOGDIR=$BASE/tensor_logs_open_aligner

mkdir -p "$OUTDIR" "$TEST_OUT" "$LOGDIR"

# =========================================================
# Helper
# =========================================================
find_latest_checkpoint() {
  local dir="$1"
  ls -d "$dir"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true
}

TRAIN_OUTPUT_DIR="$OUTDIR"

# =========================================================
# TEST
# =========================================================
echo "===================="
echo "[2/2] TEST (load_from_lora from latest checkpoint of this run)"
echo "===================="

LATEST_CKPT=$(find_latest_checkpoint "$TRAIN_OUTPUT_DIR")
if [[ -z "${LATEST_CKPT:-}" ]]; then
  echo "[ERROR] No checkpoint-* found in $TRAIN_OUTPUT_DIR"
  exit 1
fi

echo "[TEST][INFO] Latest checkpoint = $LATEST_CKPT"

echo "[TEST][INFO] Checking tokenizer files in $LATEST_CKPT"
ls -l \
  "$LATEST_CKPT"/tokenizer* \
  "$LATEST_CKPT"/special_tokens_map.json \
  "$LATEST_CKPT"/tokenizer_config.json \
  "$LATEST_CKPT"/added_tokens.json \
  2>/dev/null || true

if [[ ! -f "$LATEST_CKPT/tokenizer_config.json" ]]; then
  echo "[ERROR] tokenizer files not found in checkpoint: $LATEST_CKPT"
  echo "[ERROR] This likely means time tokens are not preserved in tokenizer."
  exit 1
fi

torchrun --nproc_per_node=1 --master_port=${MASTER_PORT_TEST} \
  $BASE/llava/train/train.py \
  --version qwen_1_5 \
  --audio_visual True \
  --whisper_path openai/whisper-large-v3 \
  --freeze_whisper True \
  --window_level_Qformer True \
  --second_per_window 0.5 \
  --second_stride 0.5 \
  --video_fps 1 \
  --max_time 60 \
  --test_data_path "$TEST_JSON" \
  --model_base "$MODEL_BASE" \
  --ckpt "$BASE_CKPT" \
  --load_from_lora True \
  --load_full False \
  --lora_enable True \
  --lora_path "$LATEST_CKPT" \
  --bf16 True \
  --do_test True \
  --test_output_dir "$TEST_OUT" \
  --output_dir "$TRAIN_OUTPUT_DIR" \
  --model_max_length 4096 \
  --modality_max_length "[64,256,4096]" \
  --add_time_token True \
  --use_timestamps_crop False \
  --mm_spatial_pool_stride 4 \
  --per_device_eval_batch_size 1 \
  --max_new_tokens 512 \
  --do_sample False \
  --mm_patch_merge_type spatial \
  --evaluation_strategy "no"

echo "[DONE] test_results.json should be in: $TEST_OUT/test_results.json"
echo "[DONE] train output_dir used in this run: $TRAIN_OUTPUT_DIR"
