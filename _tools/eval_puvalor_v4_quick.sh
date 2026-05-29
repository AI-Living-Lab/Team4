#!/bin/bash
# ============================================================
# salmonn2plus_puvalor_v4 quick eval (200-sample subset, single GPU)
#   - Used for prompt-hint A/B (full / none / min) on already-merged ckpt
#   - GPU는 환경변수 EVAL_GPU 로 지정 (기본 5), no shard, merge step skip
# Usage: EVAL_GPU=5 eval_puvalor_v4_quick.sh <merged_model_dir> <test_json> <tag>
#   tag: full|none|min  -> output dir suffix
# ============================================================
set -eo pipefail

source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

export ARNOLD_WORKER_NUM=1 ARNOLD_ID=0 METIS_WORKER_0_HOST=localhost

BASE_CODE=/home/aix23102/audiolm/vS2_eunji/video_SALMONN2_plus
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
EVAL_SCRIPT=/home/aix23102/audiolm/vS2_eunji/eval/eval_miou_subgroup.py

MERGED_MODEL=${1:?Usage: $0 <merged_model_dir> <test_json> <tag>}
TEST_JSON=${2:?missing test_json}
TAG=${3:?missing tag}

CKPT_NAME=$(basename "$MERGED_MODEL" | sed 's/_merged$//')
RESULTS_BASE=/home/aix23102/audiolm/vS2_eunji/eval/results/puvalor_v4_quick_${CKPT_NAME}_${TAG}
mkdir -p "$RESULTS_BASE"

if [ ! -f "$MERGED_MODEL/config.json" ]; then
    echo "[ERROR] merged model missing: $MERGED_MODEL" >&2
    exit 1
fi

echo "========== QUICK EVAL =========="
echo "  ckpt   : $MERGED_MODEL"
echo "  test   : $TEST_JSON"
echo "  tag    : $TAG"
echo "  out    : $RESULTS_BASE"

cd "$BASE_CODE"

PORT=$((16200 + RANDOM % 200))
CUDA_VISIBLE_DEVICES=${EVAL_GPU:-5} torchrun --nproc_per_node=1 --master_port=$PORT \
    qwenvl/train/train_qwen.py \
    --model_base "$MODEL_BASE" --run_test True --pred_rank 0 \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "$MERGED_MODEL" \
    --dataset_use "$TEST_JSON" --bf16 --output_dir "$RESULTS_BASE" \
    --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
    --max_pixels 176400 --min_pixels 784 \
    --video_max_frame_pixels 28224 --video_min_frame_pixels 784 \
    --eval_strategy "no" --save_strategy "no" --learning_rate 1e-5 \
    --model_max_length 100000 --gradient_checkpointing True --dataloader_num_workers 2 \
    --run_name "quick_${CKPT_NAME}_${TAG}" --report_to none \
    --video_min_frames 64 --video_max_frames 512 --base_interval 0.2 \
    --lora_ckpt "No" --no_audio False 2>&1 | tee "$RESULTS_BASE/inference.log"

# train_qwen.py writes test_results_rank0.json into a nested dir matching --run_name.
# Find it and copy as merged_results.json so the eval script can read it.
RESULT_FILE=$(find "$RESULTS_BASE" -name "test_results_rank0.json" | head -1)
if [ -z "$RESULT_FILE" ]; then
    echo "[ERROR] test_results_rank0.json not found under $RESULTS_BASE" >&2
    exit 1
fi
cp "$RESULT_FILE" "$RESULTS_BASE/merged_results.json"
echo "Saved: $RESULTS_BASE/merged_results.json"

python3 "$EVAL_SCRIPT" \
    --results "$RESULTS_BASE/merged_results.json" \
    --test_json "$TEST_JSON" \
    --max_time 9999.0 \
    --out_dir "$RESULTS_BASE"

echo "========== DONE: $RESULTS_BASE =========="
