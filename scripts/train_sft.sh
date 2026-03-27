#!/bin/bash

# Copyright (2025) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

PROJECT_ROOT=$(cd $(dirname $0); pwd)
cd $PROJECT_ROOT

# 경로 설정 로드
if [ -f ../paths.env ]; then
    source ../paths.env
else
    echo "[WARNING] paths.env not found. Copy paths.env.example to paths.env and fill in the paths."
fi

bash run.sh \
    --training_data ${DATA_DIR}/training_data.json \
    --model_base ${BASE_MODEL} \
    --model ${CHECKPOINTS_DIR}/sft_ckpt/checkpoint-12000.bin \
    --load_from_lora \
    --save_model_name debug \
    --epochs 5 --save_steps 20 --lr 2e-5 --max_time 110 --fps 1 --add_time_token \
    --use_lora --lora_r 128 --lora_alpha 256  --lora_dropout 0.05 --load_full \
    --pretrain_weight ${CHECKPOINTS_DIR}/pretrain_audio/checkpoint-30000.bin \
    --winqf_second 0.5 --audio_visual \
    --mm_pooling_position after
