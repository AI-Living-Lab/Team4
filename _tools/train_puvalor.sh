#!/bin/bash
# train_pu_valor.sh
# PU-VALOR time-event alignment 학습 스크립트

PROJECT_ROOT=$(cd $(dirname $0); pwd)
cd $PROJECT_ROOT

bash /home/aix23102/audiolm/vS2_eunji/scripts/run.sh \
    --training_data /home/aix23102/audiolm/vS2_eunji/data/pu_valor_train.json \
    --model_base    /path/to/OV-Qwen2-7B-AM9 \
    --model         /path/to/base_checkpoint.bin \
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
