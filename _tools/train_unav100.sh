#!/bin/bash
# train_unav100.sh
# UnAV-100 dense localization finetuning
# run_finetuning.sh의 문제(vision_tower 누락 등)를 수정한 버전

export CUDA_VISIBLE_DEVICES=6,7
export ARNOLD_WORKER_GPU=2

PROJECT_ROOT=$(cd $(dirname $0); pwd)
cd $PROJECT_ROOT

export SAVE_DIR=/data0/aix23102/checkpoints_open_aligner
export MODEL_MAX_LENGTH=10240
export POOLING_STRIDE=4
OUTPUT_DIR=${SAVE_DIR}/unav100_sft
RESUME_ARG=""
LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [[ -n "$LATEST_CKPT" ]]; then
    echo "[RESUME] Found checkpoint: $LATEST_CKPT"
    RESUME_ARG="--resume_from_checkpoint $LATEST_CKPT"
else
    echo "[RESUME] No checkpoint found → training from scratch"
fi

bash /home/aix23102/audiolm/vS2_eunji/scripts/run.sh \
    --training_data /home/aix23102/audiolm/vS2_eunji/data/unav100_train_dense.json \
    --model_base    /home/aix23102/audiolm/video-SALMONN-2/checkpoints/llava_onevision_qwen2_7b_ov \
    --model         /home/aix23102/audiolm/video-SALMONN-2/checkpoints/video_salmonn2_hf \
    --save_model_name unav100_sft \
    --epochs        5 \
    --save_strategy epoch \
    --lr            2e-5 \
    --max_time      60 \
    --fps           1 \
    --add_time_token \
    --use_lora \
    --lora_r        128 \
    --lora_alpha    256 \
    --lora_dropout  0.05 \
    --winqf_second  0.5 \
    --mm_pooling_position after \
    --deepspeed_type zero2 \
    --audio_visual \
    $RESUME_ARG
