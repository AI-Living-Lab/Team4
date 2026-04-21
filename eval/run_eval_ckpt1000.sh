#!/usr/bin/env bash
set -eo pipefail

source /workspace/setup.sh
conda activate salmonn2plus

export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

BASE_CODE=${BASE_DIR}/video_SALMONN2_plus
MODEL_BASE=${CHECKPOINTS_DIR}/video_salmonn2_plus_7B_time_tokens
TEST_JSON=${BASE_DIR}/data/unav100_test_multiseg_sub80.json
EVAL_SCRIPT=${BASE_DIR}/eval/eval_miou_multiseg.py

LORA_CKPT=${CHECKPOINTS_DIR}/salmonn2plus_unav100_multiseg/checkpoint-1500
NAME="salmonn2p_unav_multiseg_sft_ckpt1500"
OUT_DIR=${BASE_DIR}/eval/results/salmonn2plus_sweep

cd "$BASE_CODE"
mkdir -p "$OUT_DIR"

echo "================================================"
echo "  Eval: $NAME"
echo "  LoRA: $LORA_CKPT"
echo "  GPU: 0"
echo "================================================"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12900 \
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
    --video_max_frame_pixels 28224 \
    --video_min_frame_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-5 \
    --model_max_length 100000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --run_name "$NAME" \
    --report_to none \
    --video_min_frames 64 \
    --video_max_frames 128 \
    --base_interval 0.2 \
    --lora_ckpt "$LORA_CKPT" \
    --no_audio False 2>&1 | tee "$OUT_DIR/${NAME}_inference.log"

# Clean up merged model to free disk space
rm -rf "$OUT_DIR/generation_0"

# eval
RESULT_FILE="$OUT_DIR/$NAME/test_results_rank0.json"
if [ -f "$RESULT_FILE" ]; then
    python3 "$EVAL_SCRIPT" \
        --results "$RESULT_FILE" \
        --test_json "$TEST_JSON" \
        --max_time 60.0 \
        --out_dir "$OUT_DIR/$NAME"
    echo "✓ $NAME evaluation complete"
else
    echo "✗ $NAME: test_results_rank0.json not found"
fi
