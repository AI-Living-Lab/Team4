#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/home/aix23102/audiolm/vS2_eunji:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

MASTER_PORT_TRAIN=29501
MASTER_PORT_TEST=29502

BASE=/home/aix23102/audiolm/vS2_eunji
MODEL_BASE=/home/aix23102/audiolm/video-SALMONN-2/checkpoints/llava_onevision_qwen2_7b_ov
BASE_CKPT=/home/aix23102/audiolm/video-SALMONN-2/checkpoints/video_salmonn2_hf

TRAIN_JSON=$BASE/data/unav100_train_vs2.json
TEST_JSON=$BASE/data/unav100_test_vs2_5.json

OUTDIR=$BASE/checkpoints_subset_sft-600
TEST_OUT=$BASE/output/test_time_token_single_total
LOGDIR=$BASE/tensor_logs_time_token_single_total

mkdir -p "$OUTDIR" "$TEST_OUT" "$LOGDIR"

# =========================================================
# Helper
# =========================================================
find_latest_checkpoint() {
  local dir="$1"
  ls -d "$dir"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true
}

is_real_resume_checkpoint() {
  local ckpt="$1"
  [[ -f "$ckpt/trainer_state.json" && -f "$ckpt/optimizer.pt" && -f "$ckpt/scheduler.pt" ]]
}

# =========================================================
# Detect training mode
# =========================================================
LATEST_CKPT_TRAIN=$(find_latest_checkpoint "$OUTDIR")

REAL_RESUME=0
WARM_START=0
SCRATCH=0
TRAIN_OUTPUT_DIR="$OUTDIR"

if [[ -n "${LATEST_CKPT_TRAIN:-}" ]]; then
  echo "[TRAIN][FOUND] Latest checkpoint: $LATEST_CKPT_TRAIN"

  if is_real_resume_checkpoint "$LATEST_CKPT_TRAIN"; then
    REAL_RESUME=1
    TRAIN_OUTPUT_DIR="$OUTDIR"
    echo "[TRAIN][MODE] Real resume is available."
  else
    WARM_START=1
    TRAIN_OUTPUT_DIR="${OUTDIR}_warmstart_from_$(basename "$LATEST_CKPT_TRAIN")"
    mkdir -p "$TRAIN_OUTPUT_DIR"
    echo "[TRAIN][MODE] Real resume files missing."
    echo "[TRAIN][MODE] Warm-start from LoRA checkpoint: $LATEST_CKPT_TRAIN"
    echo "[TRAIN][MODE] New output_dir = $TRAIN_OUTPUT_DIR"
  fi
else
  SCRATCH=1
  TRAIN_OUTPUT_DIR="$OUTDIR"
  echo "[TRAIN][MODE] No checkpoint found. Training will start from scratch."
fi

# echo "===================="
# echo "[1/2] TRAIN"
# echo "train_json      = $TRAIN_JSON"
# echo "base_ckpt       = $BASE_CKPT"
# echo "model_base      = $MODEL_BASE"
# echo "output_dir      = $TRAIN_OUTPUT_DIR"
# echo "latest_ckpt     = ${LATEST_CKPT_TRAIN:-NONE}"
# echo "real_resume     = $REAL_RESUME"
# echo "warm_start      = $WARM_START"
# echo "scratch         = $SCRATCH"
# echo "===================="

# # =========================================================
# # TRAIN
# # =========================================================
# if [[ "$REAL_RESUME" -eq 1 ]]; then
#   echo "[TRAIN] Running REAL RESUME path..."

#   torchrun --nproc_per_node=1 --master_port=${MASTER_PORT_TRAIN} \
#     $BASE/llava/train/train.py \
#     --version qwen_1_5 \
#     --audio_visual True \
#     --whisper_path openai/whisper-large-v3 \
#     --freeze_whisper True \
#     --freeze_backbone True \
#     --window_level_Qformer True \
#     --second_per_window 0.5 \
#     --second_stride 0.5 \
#     --video_fps 1 \
#     --max_time 60 \
#     --data_path "$TRAIN_JSON" \
#     --model_base "$MODEL_BASE" \
#     --ckpt "$BASE_CKPT" \
#     --output_dir "$TRAIN_OUTPUT_DIR" \
#     --bf16 True \
#     --lora_enable True \
#     --lora_r 32 \
#     --lora_alpha 64 \
#     --lora_dropout 0.05 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 5e-5 \
#     --evaluation_strategy "no" \
#     --use_timestamps_crop False \
#     --mm_spatial_pool_stride 4 \
#     --model_max_length 2048 \
#     --modality_max_length "[64,256,2048]" \
#     --add_time_token True \
#     --num_train_epochs 20 \
#     --save_strategy "steps" \
#     --save_steps 200 \
#     --save_total_limit 4 \
#     --gradient_checkpointing True \
#     --mm_patch_merge_type spatial \
#     --logging_strategy steps \
#     --logging_steps 5 \
#     --report_to tensorboard \
#     --logging_dir "$LOGDIR" \
#     --disable_tqdm False

# elif [[ "$WARM_START" -eq 1 ]]; then
#   echo "[TRAIN] Running WARM-START path from LoRA checkpoint..."

#   torchrun --nproc_per_node=1 --master_port=${MASTER_PORT_TRAIN} \
#     $BASE/llava/train/train.py \
#     --version qwen_1_5 \
#     --audio_visual True \
#     --whisper_path openai/whisper-large-v3 \
#     --freeze_whisper True \
#     --freeze_backbone True \
#     --window_level_Qformer True \
#     --second_per_window 0.5 \
#     --second_stride 0.5 \
#     --video_fps 1 \
#     --max_time 60 \
#     --data_path "$TRAIN_JSON" \
#     --model_base "$MODEL_BASE" \
#     --ckpt "$BASE_CKPT" \
#     --load_from_lora True \
#     --load_full False \
#     --lora_enable True \
#     --lora_path "$LATEST_CKPT_TRAIN" \
#     --output_dir "$TRAIN_OUTPUT_DIR" \
#     --bf16 True \
#     --lora_r 32 \
#     --lora_alpha 64 \
#     --lora_dropout 0.05 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 5e-5 \
#     --evaluation_strategy "no" \
#     --use_timestamps_crop False \
#     --mm_spatial_pool_stride 4 \
#     --model_max_length 2048 \
#     --modality_max_length "[64,256,2048]" \
#     --add_time_token True \
#     --num_train_epochs 20 \
#     --save_strategy "steps" \
#     --save_steps 200 \
#     --save_total_limit 4 \
#     --gradient_checkpointing True \
#     --mm_patch_merge_type spatial \
#     --logging_strategy steps \
#     --logging_steps 5 \
#     --report_to tensorboard \
#     --logging_dir "$LOGDIR" \
#     --disable_tqdm False

# else
#   echo "[TRAIN] Running SCRATCH path..."

#   torchrun --nproc_per_node=1 --master_port=${MASTER_PORT_TRAIN} \
#     $BASE/llava/train/train.py \
#     --version qwen_1_5 \
#     --audio_visual True \
#     --whisper_path openai/whisper-large-v3 \
#     --freeze_whisper True \
#     --freeze_backbone True \
#     --window_level_Qformer True \
#     --second_per_window 0.5 \
#     --second_stride 0.5 \
#     --video_fps 1 \
#     --max_time 60 \
#     --data_path "$TRAIN_JSON" \
#     --model_base "$MODEL_BASE" \
#     --ckpt "$BASE_CKPT" \
#     --output_dir "$TRAIN_OUTPUT_DIR" \
#     --bf16 True \
#     --lora_enable True \
#     --lora_r 32 \
#     --lora_alpha 64 \
#     --lora_dropout 0.05 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 5e-5 \
#     --evaluation_strategy "no" \
#     --use_timestamps_crop False \
#     --mm_spatial_pool_stride 4 \
#     --model_max_length 2048 \
#     --modality_max_length "[64,256,2048]" \
#     --add_time_token True \
#     --num_train_epochs 20 \
#     --save_strategy "steps" \
#     --save_steps 200 \
#     --save_total_limit 4 \
#     --gradient_checkpointing True \
#     --mm_patch_merge_type spatial \
#     --logging_strategy steps \
#     --logging_steps 5 \
#     --report_to tensorboard \
#     --logging_dir "$LOGDIR" \
#     --disable_tqdm False
# fi

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
  --model_max_length 2048 \
  --modality_max_length "[64,256,2048]" \
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