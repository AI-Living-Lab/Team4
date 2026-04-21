#!/bin/bash
# ============================================================
# PU-VALOR + UnAV-100 모든 체크포인트 inference + eval
#   - GPU 4,5 사용 (2개 모델 병렬)
#   - UnAV-100 test subset(80v)으로 mIoU + R@1 평가
# ============================================================
set -eo pipefail

source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

export ARNOLD_WORKER_NUM=1 ARNOLD_ID=0 METIS_WORKER_0_HOST=localhost

BASE_CODE=/home/aix23102/audiolm/video-SALMONN-2/video_SALMONN2_plus
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
TEST_JSON=/home/aix23102/audiolm/vS2_eunji/data/unav100_test_multiseg_sub80.json
EVAL_SCRIPT=/home/aix23102/audiolm/vS2_eunji/eval/eval_miou_multiseg.py
RESULTS_BASE=/home/aix23102/audiolm/vS2_eunji/eval/results/salmonn2plus_lora_timetoken

PUVALOR_DIR=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_0.3ep_lora_timetoken
UNAV_DIR=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_unav100_multiseg_lora_timetoken

cd "$BASE_CODE"
mkdir -p "$RESULTS_BASE"

run_eval() {
    local GPU=$1 NAME=$2 LORA_CKPT=$3
    local OUT_DIR=$RESULTS_BASE/$NAME
    local PORT=$((13600 + GPU))
    mkdir -p "$OUT_DIR"
    echo "[GPU $GPU] Starting $NAME (LoRA: $LORA_CKPT)"

    CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$PORT \
        qwenvl/train/train_qwen.py \
        --model_base "$MODEL_BASE" --run_test True --pred_rank 0 \
        --deepspeed scripts/zero2.json \
        --model_name_or_path "$MODEL_BASE" \
        --dataset_use "$TEST_JSON" --bf16 --output_dir "$OUT_DIR" \
        --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
        --max_pixels 176400 --min_pixels 784 --video_max_frame_pixels 25088 --video_min_frame_pixels 3136 \
        --eval_strategy "no" --save_strategy "no" --learning_rate 1e-5 \
        --model_max_length 6000 --gradient_checkpointing True --dataloader_num_workers 2 \
        --run_name "$NAME" --report_to none \
        --video_min_frames 4 --video_max_frames 128 --base_interval 2 \
        --lora_ckpt "$LORA_CKPT" --no_audio False 2>&1 | tee "$OUT_DIR/inference.log"

    RESULT_FILE="$OUT_DIR/$NAME/test_results_rank0.json"
    if [ -f "$RESULT_FILE" ]; then
        python3 "$EVAL_SCRIPT" --results "$RESULT_FILE" --test_json "$TEST_JSON" --max_time 60.0 --out_dir "$OUT_DIR"
        echo "[GPU $GPU] ✓ $NAME done"
    else
        echo "[GPU $GPU] ✗ $NAME: results not found at $RESULT_FILE"
    fi
}

echo "════════════════════════════════════════════════════════"
echo "  Eval all checkpoints: PU-VALOR 0.1~0.5ep + UnAV-100 0.1~1.0ep"
echo "  GPUs: 4, 5 (2 parallel)"
echo "════════════════════════════════════════════════════════"

# --- PU-VALOR checkpoints (0.1~0.5ep) ---
PUVALOR_CKPTS=($(ls -d "$PUVALOR_DIR"/checkpoint-* 2>/dev/null | sort -V))
echo "[INFO] Found ${#PUVALOR_CKPTS[@]} PU-VALOR checkpoints"

# --- UnAV-100 checkpoints (0.1~1.0ep) ---
UNAV_CKPTS=($(ls -d "$UNAV_DIR"/checkpoint-* 2>/dev/null | sort -V))
echo "[INFO] Found ${#UNAV_CKPTS[@]} UnAV-100 checkpoints"

# Combine all checkpoints with names
ALL_NAMES=()
ALL_CKPTS=()

PUVALOR_STEPS_PER_01EP=2527
for ckpt in "${PUVALOR_CKPTS[@]}"; do
    step=$(basename "$ckpt" | sed 's/checkpoint-//')
    ep_x10=$(( (step + PUVALOR_STEPS_PER_01EP/2) / PUVALOR_STEPS_PER_01EP ))
    name="puvalor_0.${ep_x10}ep"
    ALL_NAMES+=("$name")
    ALL_CKPTS+=("$ckpt")
done

UNAV_STEPS_PER_01EP=259
for ckpt in "${UNAV_CKPTS[@]}"; do
    step=$(basename "$ckpt" | sed 's/checkpoint-//')
    ep_x10=$(( (step + UNAV_STEPS_PER_01EP/2) / UNAV_STEPS_PER_01EP ))
    if [ "$ep_x10" -eq 10 ]; then
        name="puvalor0.5_unav_1.0ep"
    else
        name="puvalor0.5_unav_0.${ep_x10}ep"
    fi
    ALL_NAMES+=("$name")
    ALL_CKPTS+=("$ckpt")
done

echo "[INFO] Total models to evaluate: ${#ALL_NAMES[@]}"
for i in "${!ALL_NAMES[@]}"; do
    echo "  [$i] ${ALL_NAMES[$i]} → ${ALL_CKPTS[$i]}"
done

# Run in pairs (GPU 4, 5)
i=0
while [ $i -lt ${#ALL_NAMES[@]} ]; do
    # GPU 4
    run_eval 4 "${ALL_NAMES[$i]}" "${ALL_CKPTS[$i]}" &
    PID1=$!

    # GPU 5 (if available)
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
echo "  SUMMARY (LoRA + time token, aligner freeze)"
echo "════════════════════════════════════════════════════════════════"
printf "%-30s %8s %8s %8s %8s %8s\n" "Model" "mIoU" "R@0.3" "R@0.5" "R@0.7" "Parse%"
echo "────────────────────────────────────────────────────────────────"
for name in "${ALL_NAMES[@]}"; do
    SUMMARY="$RESULTS_BASE/$name/eval_miou_summary.json"
    if [ -f "$SUMMARY" ]; then
        python3 -c "
import json; d=json.load(open('$SUMMARY'))
n=d['n_samples']; pok=d['parse_ok']
print(f'  {d[\"mIoU_%\"]:.2f}  {d[\"R@1\"][\"0.3\"]:.2f}  {d[\"R@1\"][\"0.5\"]:.2f}  {d[\"R@1\"][\"0.7\"]:.2f}  {pok*100/max(n,1):.0f}%')
" | while read line; do printf "%-30s %s\n" "$name" "$line"; done
    else
        printf "%-30s %8s\n" "$name" "FAILED"
    fi
done
echo "════════════════════════════════════════════════════════════════"
