#!/bin/bash
# ============================================================
# Ablation A/B 체크포인트 eval (4 models total)
#   - A: q/k/v LoRA, B: q/k/v/o + MLP LoRA
#   - UnAV-100 subset 80v, high-res inference
#   - GPU 4,5 (2 parallel)
# ============================================================
set -eo pipefail

source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

export ARNOLD_WORKER_NUM=1 ARNOLD_ID=0 METIS_WORKER_0_HOST=localhost

BASE_CODE=/home/aix23102/audiolm/video-SALMONN-2/video_SALMONN2_plus
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
TEST_JSON=/home/aix23102/audiolm/vS2_eunji/data/unav100_test_multiseg_sub80.json
EVAL_SCRIPT=/home/aix23102/audiolm/vS2_eunji/eval/eval_miou_multiseg.py
RESULTS_BASE=/home/aix23102/audiolm/vS2_eunji/eval/results/ablation_A_vs_B

cd "$BASE_CODE"
mkdir -p "$RESULTS_BASE"

run_eval() {
    local GPU=$1 NAME=$2 LORA_CKPT=$3
    local OUT_DIR=$RESULTS_BASE/$NAME
    local PORT=$((13800 + GPU))
    mkdir -p "$OUT_DIR"
    echo "[GPU $GPU] Starting $NAME"

    CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$PORT \
        qwenvl/train/train_qwen.py \
        --model_base "$MODEL_BASE" --run_test True --pred_rank 0 \
        --deepspeed scripts/zero2.json \
        --model_name_or_path "$MODEL_BASE" \
        --dataset_use "$TEST_JSON" --bf16 --output_dir "$OUT_DIR" \
        --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
        --max_pixels 176400 --min_pixels 784 \
        --video_max_frame_pixels 28224 --video_min_frame_pixels 784 \
        --eval_strategy "no" --save_strategy "no" --learning_rate 1e-5 \
        --model_max_length 100000 \
        --gradient_checkpointing True --dataloader_num_workers 2 \
        --run_name "$NAME" --report_to none \
        --video_min_frames 64 --video_max_frames 128 --base_interval 0.2 \
        --lora_ckpt "$LORA_CKPT" --no_audio False 2>&1 | tee "$OUT_DIR/inference.log"

    # Clean generation_0 to save disk
    rm -rf "$OUT_DIR/generation_0" 2>/dev/null

    RESULT_FILE="$OUT_DIR/$NAME/test_results_rank0.json"
    if [ -f "$RESULT_FILE" ]; then
        python3 "$EVAL_SCRIPT" --results "$RESULT_FILE" --test_json "$TEST_JSON" --max_time 60.0 --out_dir "$OUT_DIR"
        echo "[GPU $GPU] ✓ $NAME done"
    else
        echo "[GPU $GPU] ✗ $NAME: results not found"
    fi
}

A_DIR=/data0/aix23102/checkpoints_open_aligner/ablation_A_qkv_highres
B_DIR=/data0/aix23102/checkpoints_open_aligner/ablation_B_qkvo_mlp_highres

NAMES=("A_qkv_0.1ep" "A_qkv_0.2ep" "B_qkvo_mlp_0.1ep" "B_qkvo_mlp_0.2ep")
CKPTS=("$A_DIR/checkpoint-259" "$A_DIR/checkpoint-518" "$B_DIR/checkpoint-259" "$B_DIR/checkpoint-518")

echo "════════════════════════════════════════════"
echo "  Ablation A vs B eval (4 models)"
echo "════════════════════════════════════════════"

# Run in pairs (GPU 4, 5)
i=0
while [ $i -lt ${#NAMES[@]} ]; do
    run_eval 4 "${NAMES[$i]}" "${CKPTS[$i]}" &
    PID1=$!
    j=$((i+1))
    if [ $j -lt ${#NAMES[@]} ]; then
        run_eval 5 "${NAMES[$j]}" "${CKPTS[$j]}" &
        PID2=$!
        wait $PID1 $PID2
        i=$((i+2))
    else
        wait $PID1
        i=$((i+1))
    fi
    echo "[Batch done: $i/${#NAMES[@]}]"
done

# Summary
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  SUMMARY: A (q/k/v) vs B (q/k/v/o + MLP)"
echo "════════════════════════════════════════════════════════════════"
printf "%-22s %8s %8s %8s %8s %8s\n" "Model" "mIoU" "R@0.3" "R@0.5" "R@0.7" "Parse%"
echo "────────────────────────────────────────────────────────────────"
for name in "${NAMES[@]}"; do
    SUMMARY="$RESULTS_BASE/$name/eval_miou_summary.json"
    if [ -f "$SUMMARY" ]; then
        python3 -c "
import json; d=json.load(open('$SUMMARY'))
n=d['n_samples']; pok=d['parse_ok']
print(f'  {d[\"mIoU_%\"]:.2f}  {d[\"R@1\"][\"0.3\"]:.2f}  {d[\"R@1\"][\"0.5\"]:.2f}  {d[\"R@1\"][\"0.7\"]:.2f}  {pok*100/max(n,1):.0f}%')
" | while read line; do printf "%-22s %s\n" "$name" "$line"; done
    else
        printf "%-22s %s\n" "$name" "FAILED"
    fi
done
echo "════════════════════════════════════════════════════════════════"
