#!/bin/bash
# train_pu_valor.sh
# PU-VALOR time-event alignment 학습 스크립트

export CUDA_VISIBLE_DEVICES=4,5,6,7
export ARNOLD_WORKER_GPU=4

PROJECT_ROOT=$(cd $(dirname $0); pwd)
cd $PROJECT_ROOT

export SAVE_DIR=/data0/aix23102/checkpoints_open_aligner
export MODEL_MAX_LENGTH=24576
export POOLING_STRIDE=4
OUTPUT_DIR=${SAVE_DIR}/pu_valor_time_event
RESUME_ARG=""
LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [[ -n "$LATEST_CKPT" ]]; then
    echo "[RESUME] Found checkpoint: $LATEST_CKPT"
    RESUME_ARG="--resume_from_checkpoint $LATEST_CKPT"
else
    echo "[RESUME] No checkpoint found → training from scratch"
fi

bash /home/aix23102/audiolm/vS2_eunji/scripts/run.sh \
    --training_data /home/aix23102/audiolm/vS2_eunji/data/pu_valor_train.json \
    --model_base    /home/aix23102/audiolm/video-SALMONN-2/checkpoints/llava_onevision_qwen2_7b_ov \
    --model         /home/aix23102/audiolm/video-SALMONN-2/checkpoints/video_salmonn2_hf \
    --load_from_lora \
    --lora_path /data0/aix23102/checkpoints_open_aligner/unav100_sft/checkpoint-16225 \
    --merge_and_new_lora \
    --save_model_name pu_valor_time_event \
    --epochs        3 \
    --save_strategy epoch \
    --lr            2e-5 \
    --max_time      400 \
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
