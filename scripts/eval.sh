set -e

#!/bin/bash

# ---- Force all caches to user-writable paths ----
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="$HOME/.cache"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$HOME/.cache/torch"
export XET_CACHE_HOME="$HOME/.cache/xet"
# -------------------------------------------------

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

NODE_RANK=${NODE_RANK:-0}
PROJECT_ROOT=$(cd $(dirname $0); pwd)
cd $PROJECT_ROOT

if [ -f ../paths.env ]; then
    source ../paths.env
else
    echo "[WARNING] paths.env not found. Copy paths.env.example to paths.env and fill in the paths."
fi

# Note: 
# To evaluate a PyTorch model in ".bin" format, please add the "--load_from_lora" parameter.
# For Hugginface format models, this parameter is not required.
# To use this script for dpo data generation, you can add "--dpo_train".

#   --use_lora --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 --load_from_lora \

bash run.sh \
    --do_test \
    --test_data ${BASE_DIR}/data/unav100_test_dense.json \
    --test_id 3_unav_multi \
    --max_time 110 \
    --fps 1 \
    --model ${SALMONN2_CKPT} \
    --model_base ${BASE_MODEL} \
    --add_time_token --mm_pooling_position after \
    --audio_visual --winqf_second 0.5

