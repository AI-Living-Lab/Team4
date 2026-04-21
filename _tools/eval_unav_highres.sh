#!/bin/bash
# ============================================================
# UnAV-100 checkpoint eval (high-res, more frames)
#   - video_max_frame_pixels=28224, video_min_frame_pixels=784
#   - video_min_frames=64, base_interval=0.2
#   - model_max_length=100000
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
RESULTS_BASE=/home/aix23102/audiolm/vS2_eunji/eval/results/salmonn2plus_lora_timetoken_highres
UNAV_DIR=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_unav100_multiseg_lora_timetoken

cd "$BASE_CODE"
mkdir -p "$RESULTS_BASE"

run_eval() {
    local GPU=$1 NAME=$2 LORA_CKPT=$3
    local OUT_DIR=$RESULTS_BASE/$NAME
    local PORT=$((13700 + GPU))
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
        --video_max_frame_pixels 28224 \
        --video_min_frame_pixels 784 \
        --eval_strategy "no" --save_strategy "no" --learning_rate 1e-5 \
        --model_max_length 100000 \
        --gradient_checkpointing True --dataloader_num_workers 2 \
        --run_name "$NAME" --report_to none \
        --video_min_frames 64 \
        --video_max_frames 128 \
        --base_interval 0.2 \
        --lora_ckpt "$LORA_CKPT" --no_audio False 2>&1 | tee "$OUT_DIR/inference.log"

    # Clean up generation_0 to save disk
    rm -rf "$OUT_DIR/generation_0" 2>/dev/null

    RESULT_FILE="$OUT_DIR/$NAME/test_results_rank0.json"
    if [ -f "$RESULT_FILE" ]; then
        python3 "$EVAL_SCRIPT" --results "$RESULT_FILE" --test_json "$TEST_JSON" --max_time 60.0 --out_dir "$OUT_DIR"
        echo "[GPU $GPU] ✓ $NAME done"
    else
        echo "[GPU $GPU] ✗ $NAME: results not found at $RESULT_FILE"
    fi
}

echo "════════════════════════════════════════════════════════"
echo "  High-res eval: PU-VALOR 0.5ep + UnAV 0.1~0.5ep"
echo "  Settings: frame_pixels=28224/784, min_frames=64, interval=0.2"
echo "════════════════════════════════════════════════════════"

UNAV_STEPS_PER_01EP=259
CKPTS=($(ls -d "$UNAV_DIR"/checkpoint-* 2>/dev/null | sort -V | head -5))

ALL_NAMES=()
ALL_CKPTS=()
for ckpt in "${CKPTS[@]}"; do
    step=$(basename "$ckpt" | sed 's/checkpoint-//')
    ep_x10=$(( (step + UNAV_STEPS_PER_01EP/2) / UNAV_STEPS_PER_01EP ))
    ALL_NAMES+=("unav_0.${ep_x10}ep")
    ALL_CKPTS+=("$ckpt")
done

echo "[INFO] Models to evaluate: ${#ALL_NAMES[@]}"
for i in "${!ALL_NAMES[@]}"; do
    echo "  [$i] ${ALL_NAMES[$i]} → ${ALL_CKPTS[$i]}"
done

# Run in pairs (GPU 4, 5)
i=0
while [ $i -lt ${#ALL_NAMES[@]} ]; do
    run_eval 4 "${ALL_NAMES[$i]}" "${ALL_CKPTS[$i]}" &
    PID1=$!
    j=$((i+1))
    if [ $j -lt ${#ALL_NAMES[@]} ]; then
        run_eval 5 "${ALL_NAMES[$j]}" "${ALL_CKPTS[$j]}" &
        PID2=$!
        wait $PID1 $PID2
        i=$((i+2))
    else
        wait $PID1
        i=$((i+1))
    fi
    echo "[Batch done: $i/${#ALL_NAMES[@]}]"
done

# Summary
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  SUMMARY (high-res: 28224/784, 64 frames, interval=0.2)"
echo "════════════════════════════════════════════════════════════════"
printf "%-25s %8s %8s %8s %8s %8s\n" "Model" "mIoU" "R@0.3" "R@0.5" "R@0.7" "Parse%"
echo "────────────────────────────────────────────────────────────────"
for name in "${ALL_NAMES[@]}"; do
    SUMMARY="$RESULTS_BASE/$name/eval_miou_summary.json"
    if [ -f "$SUMMARY" ]; then
        python3 -c "
import json; d=json.load(open('$SUMMARY'))
n=d['n_samples']; pok=d['parse_ok']
print(f'  {d[\"mIoU_%\"]:.2f}  {d[\"R@1\"][\"0.3\"]:.2f}  {d[\"R@1\"][\"0.5\"]:.2f}  {d[\"R@1\"][\"0.7\"]:.2f}  {pok*100/max(n,1):.0f}%')
" | while read line; do printf "%-25s %s\n" "$name" "$line"; done
    else
        printf "%-25s %8s\n" "$name" "FAILED"
    fi
done
echo "════════════════════════════════════════════════════════════════"
