#!/usr/bin/env bash
# ============================================================
# SALMONN2+ UnAV-100 evaluation (single checkpoint × single test set)
#
# Usage:
#   bash run_eval_salmonn2p.sh <checkpoint_path> [test_json] [gpu_id]
#
# Examples:
#   bash run_eval_salmonn2p.sh checkpoint-500
#   bash run_eval_salmonn2p.sh checkpoint-500 multiseg_sub80
#   bash run_eval_salmonn2p.sh checkpoint-500 dense 2
#
# checkpoint_path:
#   - Full path: /workspace/checkpoints/.../checkpoint-500
#   - Short name: checkpoint-500  (resolved under salmonn2plus_unav100_multiseg/)
#
# test_json (default: multiseg_sub80):
#   - Full path: /workspace/rl/Team4/data/unav100_test_dense.json
#   - Short name: multiseg_sub80 | dense | dense_5 | single | single_5
#
# Output name is auto-generated:
#   salmonn2p_unav_{test_short}_{ckpt_name}
#   e.g. salmonn2p_unav_multiseg_sub80_ckpt500
# ============================================================
set -eo pipefail

source /workspace/setup.sh
conda activate salmonn2plus

export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

# ── args ──────────────────────────────────────────────────
CKPT_INPUT="${1:?Usage: $0 <checkpoint_path> [test_json] [gpu_id]}"
TEST_INPUT="${2:-multiseg_sub80}"
GPU="${3:-0}"

# ── resolve checkpoint path ──────────────────────────────
if [[ -d "$CKPT_INPUT" ]]; then
    LORA_CKPT="$CKPT_INPUT"
else
    LORA_CKPT="${CHECKPOINTS_DIR}/salmonn2plus_unav100_multiseg/${CKPT_INPUT}"
fi

if [[ ! -d "$LORA_CKPT" ]]; then
    echo "ERROR: checkpoint not found: $LORA_CKPT" >&2
    exit 1
fi

# ── resolve test json ────────────────────────────────────
if [[ -f "$TEST_INPUT" ]]; then
    TEST_JSON="$TEST_INPUT"
else
    TEST_JSON="${BASE_DIR}/data/unav100_test_${TEST_INPUT}.json"
fi

if [[ ! -f "$TEST_JSON" ]]; then
    echo "ERROR: test json not found: $TEST_JSON" >&2
    echo "Available:" >&2
    ls "${BASE_DIR}"/data/unav100_test_*.json 2>/dev/null | sed 's/.*unav100_test_/  /;s/\.json//' >&2
    exit 1
fi

# ── auto-generate name ───────────────────────────────────
CKPT_NAME=$(basename "$LORA_CKPT" | sed 's/checkpoint-/ckpt/')
TEST_SHORT=$(basename "$TEST_JSON" .json | sed 's/unav100_test_//')
NAME="salmonn2p_unav_${TEST_SHORT}_${CKPT_NAME}"

# ── paths ─────────────────────────────────────────────────
BASE_CODE=${BASE_DIR}/video_SALMONN2_plus
MODEL_BASE=${CHECKPOINTS_DIR}/video_salmonn2_plus_7B_time_tokens
EVAL_SCRIPT=${BASE_DIR}/eval/eval_miou_multiseg.py
OUT_DIR=${BASE_DIR}/eval/results/salmonn2plus_sweep

cd "$BASE_CODE"
mkdir -p "$OUT_DIR"

PORT=$((12900 + GPU))

echo "================================================"
echo "  Eval: $NAME"
echo "  LoRA: $LORA_CKPT"
echo "  Test: $TEST_JSON"
echo "  GPU:  $GPU"
echo "================================================"

CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$PORT \
    qwenvl/train/train_qwen.py \
    --model_base "$MODEL_BASE" \
    --run_test True \
    --pred_rank 0 \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "$MODEL_BASE" \
    --dataset_use "$TEST_JSON" \
    --bf16 \
    --output_dir "$OUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_pixels 176400 \
    --min_pixels 784 \
    --video_max_frame_pixels 25088 \
    --video_min_frame_pixels 3136 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-5 \
    --model_max_length 6000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --run_name "$NAME" \
    --report_to none \
    --video_min_frames 4 \
    --video_max_frames 128 \
    --base_interval 2 \
    --lora_ckpt "$LORA_CKPT" \
    --no_audio False 2>&1 | tee "$OUT_DIR/${NAME}_inference.log"

# Clean up merged model to free disk space
rm -rf "$OUT_DIR/generation_0"

# ── eval ──────────────────────────────────────────────────
RESULT_FILE="$OUT_DIR/$NAME/test_results_rank0.json"
if [[ -f "$RESULT_FILE" ]]; then
    python3 "$EVAL_SCRIPT" \
        --results "$RESULT_FILE" \
        --test_json "$TEST_JSON" \
        --max_time 60.0 \
        --out_dir "$OUT_DIR/$NAME"
    echo "✓ $NAME evaluation complete"
else
    echo "✗ $NAME: test_results_rank0.json not found at $RESULT_FILE" >&2
    exit 1
fi
