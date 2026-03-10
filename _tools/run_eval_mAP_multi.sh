set -e

#!/bin/bash

# ---- Force all caches to user-writable paths ----
export PYTHONNOUSERSITE=1
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

# Note: 
# To evaluate a PyTorch model in ".bin" format, please add the "--load_from_lora" parameter.
# For Hugginface format models, this parameter is not required.
# To use this script for dpo data generation, you can add "--dpo_train".

#   --use_lora --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 --load_from_lora \

export CUDA_VISIBLE_DEVICES=4,5

# 1. Inference (fine-tuned model)
bash /home/aix23102/audiolm/vS2_eunji/scripts/run.sh \
  --do_test \
  --test_data /home/aix23102/audiolm/vS2_eunji/data/unav100_test_vs2.json \
  --test_id 2_vs2_ft\
  --model /home/aix23102/audiolm/vS2_eunji/checkpoints/finetuning2/merged \
  --model_base /home/aix23102/audiolm/video-SALMONN-2/checkpoints/llava_onevision_qwen2_7b_ov \
  --audio_visual \
  --second_per_window 0.5 \
  --second_stride 0.5

# 2. Evaluation (CCNet-style mAP)
python /home/aix23102/audiolm/vS2_eunji/_tools/eval_mAP_multi.py \
  --unav_json /home/aix23102/audiolm/CCNet/data/unav_100/annotations/unav100_annotations.json \
  --vs2_results /home/aix23102/audiolm/vS2_eunji/output/test/0_vs2_original/test_results.json \
  --split test