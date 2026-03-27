#!/bin/bash
# train_pu_valor.sh
# PU-VALOR time-event alignment 학습 스크립트

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR

if [ -f "$SCRIPT_DIR/../paths.env" ]; then
    source "$SCRIPT_DIR/../paths.env"
else
    echo "[WARNING] paths.env not found. Copy paths.env.example to paths.env and fill in the paths."
fi

bash ${BASE_DIR}/scripts/run.sh \
    --training_data ${BASE_DIR}/data/pu_valor_train.json \
    --model_base    ${BASE_MODEL} \
    --model         ${SFT_CKPT} \
    --load_from_lora \
    --save_model_name pu_valor_time_event \
    --epochs        3 \
    --save_steps    500 \
    --lr            2e-5 \
    --max_time      170 \
    --fps           1 \
    --add_time_token \
    --use_lora \
    --lora_r        128 \
    --lora_alpha    256 \
    --lora_dropout  0.05 \
    --winqf_second  0.5 \
    --mm_pooling_position after \
    --audio_visual
