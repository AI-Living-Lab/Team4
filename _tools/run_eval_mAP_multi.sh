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

SCRIPT_DIR=$(cd $(dirname $0); pwd)
if [ -f "$SCRIPT_DIR/../paths.env" ]; then
    source "$SCRIPT_DIR/../paths.env"
else
    echo "[WARNING] paths.env not found. Copy paths.env.example to paths.env and fill in the paths."
fi

NODE_RANK=${NODE_RANK:-0}
cd $SCRIPT_DIR

export CUDA_VISIBLE_DEVICES=4,5

# 1. Inference (fine-tuned model)
bash ${BASE_DIR}/scripts/run.sh \
  --do_test \
  --test_data ${BASE_DIR}/data/unav100_test_vs2.json \
  --test_id 2_vs2_ft \
  --model ${SFT_CKPT} \
  --model_base ${BASE_MODEL} \
  --audio_visual \
  --second_per_window 0.5 \
  --second_stride 0.5

# 2. Evaluation (CCNet-style mAP)
python ${BASE_DIR}/_tools/eval_mAP_multi.py \
  --unav_json ${UNAV_ANNO} \
  --vs2_results ${BASE_DIR}/output/test/0_vs2_original/test_results.json \
  --split test
