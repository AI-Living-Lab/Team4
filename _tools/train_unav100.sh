#!/bin/bash
# train_unav100.sh
# UnAV-100 dense localization finetuning
# run_finetuning.sh의 문제(vision_tower 누락 등)를 수정한 버전

export CUDA_VISIBLE_DEVICES=4,5
export ARNOLD_WORKER_GPU=2

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR

if [ -f "$SCRIPT_DIR/../paths.env" ]; then
    source "$SCRIPT_DIR/../paths.env"
else
    echo "[WARNING] paths.env not found. Copy paths.env.example to paths.env and fill in the paths."
fi

bash ${BASE_DIR}/scripts/run.sh \
    --training_data ${BASE_DIR}/data/unav100_train_dense.json \
    --model_base    ${BASE_MODEL} \
    --model         ${SALMONN2_CKPT} \
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
