#!/usr/bin/env bash
# ============================================================
# run_inference.sh
#   vS2 fine-tuned 모델로 UnAV-100 test set inference 수행
#   결과: $TEST_OUT/test_results.json
# ============================================================
set -euo pipefail

export PYTHONPATH=/home/aix23102/audiolm/vS2_eunji:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=5,6

CKPT=71379

BASE=/home/aix23102/audiolm/vS2_eunji
MODEL_BASE=/home/aix23102/audiolm/video-SALMONN-2/checkpoints/llava_onevision_qwen2_7b_ov
BASE_CKPT=/home/aix23102/audiolm/video-SALMONN-2/checkpoints/video_salmonn2_hf

# ↓ 평가할 checkpoint 폴더로 교체
LORA_PATH=$BASE/checkpoints_open_aligner/checkpoint-$CKPT

TEST_JSON=$BASE/data/unav100_test_dense.json
TEST_OUT=$BASE/eval/results/unav100_test_uf_$CKPT


mkdir -p "$TEST_OUT"
 
torchrun --nproc_per_node=2 --master_port=29521 \
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
  --mm_spatial_pool_stride 4 \
  --mm_patch_merge_type spatial \
  --model_max_length 4096 \
  --modality_max_length "[64,256,4096]" \
  --add_time_token True \
  --model_base "$MODEL_BASE" \
  --ckpt "$BASE_CKPT" \
  --lora_enable True \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --load_from_lora True \
  --lora_path "$LORA_PATH" \
  --do_test True \
  --test_data_path "$TEST_JSON" \
  --test_output_dir "$TEST_OUT" \
  --max_new_tokens 1024 \
  --output_dir "$TEST_OUT" \
  --bf16 True \
  --per_device_eval_batch_size 1 \
  --dataloader_num_workers 2 \
  --remove_unused_columns False \
  --use_timestamps_crop False \
  --evaluation_strategy "no"
 