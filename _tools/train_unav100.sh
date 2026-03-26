#!/bin/bash
# train_unav100.sh
# UnAV-100 dense localization finetuning
# run_finetuning.sh의 문제(vision_tower 누락 등)를 수정한 버전

export CUDA_VISIBLE_DEVICES=4,5
export ARNOLD_WORKER_GPU=2

PROJECT_ROOT=$(cd $(dirname $0); pwd)
cd $PROJECT_ROOT


bash /home/aix23102/audiolm/vS2_eunji/scripts/run.sh \
    --training_data /home/aix23102/audiolm/vS2_eunji/data/unav100_train_dense.json \
    --model_base    /home/aix23102/audiolm/video-SALMONN-2/checkpoints/llava_onevision_qwen2_7b_ov \
    --model         /home/aix23102/audiolm/video-SALMONN-2/checkpoints/video_salmonn2_hf \
    --save_model_name unav100_sft \
    --epochs        5 \
    --save_steps    500 \
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
    --audio_visual
