#!/bin/bash
# ============================================================
# train_gdpo.sh
#   GDPO 학습 실행 스크립트
#   SFT 체크포인트 → GDPO 강화학습
# ============================================================
set -e

PROJECT_ROOT=$(cd $(dirname $0); pwd)
cd $PROJECT_ROOT
cd ..

export PYTHONPATH="$(pwd):$PYTHONPATH"

# 경로 설정 로드
if [ -f ./paths.env ]; then
    source ./paths.env
else
    echo "[WARNING] paths.env not found. Copy paths.env.example to paths.env and fill in the paths."
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ============================================================
# 1. 데이터 변환 (SFT → GDPO 형식)
# ============================================================
GDPO_DATA=${BASE_DIR}/data/unav100_train_gdpo.json

if [ ! -f "$GDPO_DATA" ]; then
    echo "[GDPO] Converting training data to GDPO format..."
    python ${BASE_DIR}/_tools/GDPO/convert_to_gdpo.py \
        --input  ${BASE_DIR}/data/unav100_train_dense.json \
        --output ${GDPO_DATA}
fi

# ============================================================
# 2. GDPO 학습
# ============================================================
echo "[GDPO] Starting GDPO training..."

python ${BASE_DIR}/_tools/GDPO/train_gdpo.py \
    --model_path    ${SFT_CKPT} \
    --model_base    ${BASE_MODEL} \
    --dataset_path  ${GDPO_DATA} \
    --output_dir    ${BASE_DIR}/output/gdpo \
    --num_generations       4 \
    --max_completion_length 1024 \
    --num_train_epochs      1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --beta          0.04 \
    --reward_weights 0.1 0.3 0.6 \
    --lora_r        32 \
    --lora_alpha    64 \
    --lora_dropout  0.05 \
    --logging_steps 5 \
    --save_steps    500 \
    --seed          2024 \
    --bf16

echo "[GDPO] Training complete!"
