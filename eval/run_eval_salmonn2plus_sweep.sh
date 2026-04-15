#!/usr/bin/env bash
# ============================================================
# SALMONN2+ 6개 모델 × UnAV-100 multi-segment subset(80v) 평가
# GPU 4,5,6,7 — 4개 모델 병렬 → 2개 모델 병렬
# ============================================================
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
RESULTS_BASE=${BASE_DIR}/eval/results/salmonn2plus_sweep

PUVALOR_DIR=${CHECKPOINTS_DIR}/salmonn2plus_puvalor_0.3ep
UNAV_DIR=${CHECKPOINTS_DIR}/salmonn2plus_unav100_multiseg

cd "$BASE_CODE"

run_eval() {
    local GPU=$1
    local NAME=$2
    local LORA_CKPT=$3
    local OUT_DIR=$RESULTS_BASE/$NAME
    local PORT=$((12900 + GPU))

    mkdir -p "$OUT_DIR"

    echo "[GPU $GPU] Starting $NAME (LoRA: $LORA_CKPT)"

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
        --no_audio False 2>&1 | tee "$OUT_DIR/inference.log"

    # eval
    RESULT_FILE=$(find "$OUT_DIR" -name "test_results_rank0.json" 2>/dev/null | head -1)
    if [ -f "$RESULT_FILE" ]; then
        python3 "$EVAL_SCRIPT" \
            --results "$RESULT_FILE" \
            --test_json "$TEST_JSON" \
            --max_time 60.0 \
            --out_dir "$OUT_DIR"
        echo "[GPU $GPU] ✓ $NAME done"
    else
        echo "[GPU $GPU] ✗ $NAME: results file not found"
    fi
}

echo "================================================"
echo "  SALMONN2+ Sweep: 6 models × subset 80v"
echo "  GPUs: 4,5,6,7 (4 parallel)"
echo "================================================"

rm -rf "$RESULTS_BASE" 2>/dev/null
mkdir -p "$RESULTS_BASE"

# Batch 1: 4 models in parallel (GPU 4,5,6,7)
run_eval 4 "base_no_training"        "No" &
run_eval 5 "puvalor_0.2ep"           "$PUVALOR_DIR/checkpoint-5056" &
run_eval 6 "puvalor_0.3ep"           "$PUVALOR_DIR/checkpoint-7581" &
run_eval 7 "puvalor0.3_unav_0.39ep"  "$UNAV_DIR/checkpoint-1000" &
wait
echo "[Batch 1 done]"

# Batch 2: 2 models in parallel (GPU 4,5)
run_eval 4 "puvalor0.3_unav_0.77ep"  "$UNAV_DIR/checkpoint-2000" &
run_eval 5 "puvalor0.3_unav_1.0ep"   "$UNAV_DIR/checkpoint-2589" &
wait
echo "[Batch 2 done]"

# Summary
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "════════════════════════════════════════════════════════════════════"
printf "%-32s %8s %8s %8s %8s %8s %8s\n" "Model" "mIoU" "R@0.1" "R@0.3" "R@0.5" "R@0.7" "Parse%"
echo "────────────────────────────────────────────────────────────────────"
for DIR in $RESULTS_BASE/*/; do
    NAME=$(basename $DIR)
    SUMMARY=$DIR/eval_miou_summary.json
    if [ -f "$SUMMARY" ]; then
        python3 -c "
import json
d = json.load(open('$SUMMARY'))
n = d['n_samples']
pok = d['parse_ok']
print(f'$NAME {d[\"mIoU_%\"]:.2f} {d[\"R@1\"][\"0.1\"]:.2f} {d[\"R@1\"][\"0.3\"]:.2f} {d[\"R@1\"][\"0.5\"]:.2f} {d[\"R@1\"][\"0.7\"]:.2f} {pok*100/max(n,1):.0f}%')
" | while read line; do printf "%-32s %8s %8s %8s %8s %8s %8s\n" $line; done
    fi
done
echo "════════════════════════════════════════════════════════════════════"
