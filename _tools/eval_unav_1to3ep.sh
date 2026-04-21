#!/bin/bash
# 1.0→3.0ep 10개 신규 ckpt (steps 3108~7770) eval
set -eo pipefail

source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

export ARNOLD_WORKER_NUM=1 ARNOLD_ID=0 METIS_WORKER_0_HOST=localhost

UNAV_DIR=/data0/aix23102/checkpoints_open_aligner/unav_from_puvalor_lowres_0.2ep
RESULTS_BASE=/home/aix23102/audiolm/vS2_eunji/eval/results/unav_from_puvalor_lowres_1to3ep

# Wait until all 10 new checkpoints exist (steps 3108..7770 at 518 interval)
echo "[WAITER] $(date) Waiting for 10 new ckpts (steps 3108~7770)..."
while :; do
    count=0
    for s in 3108 3626 4144 4662 5180 5698 6216 6734 7252 7770; do
        [ -d "$UNAV_DIR/checkpoint-$s" ] && count=$((count+1))
    done
    if [ "$count" -eq 10 ]; then break; fi
    sleep 180
done
# Safety: training must be done
while pgrep -f "train_qwen.*unav_from_puvalor_lowres" > /dev/null 2>&1; do
    sleep 60
done
echo "[WAITER] $(date) Training finished. Starting eval..."
sleep 30

BASE_CODE=/home/aix23102/audiolm/video-SALMONN-2/video_SALMONN2_plus
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
TEST_JSON=/home/aix23102/audiolm/vS2_eunji/data/unav100_test_multiseg_sub80.json
EVAL_SCRIPT=/home/aix23102/audiolm/vS2_eunji/eval/eval_miou_multiseg.py

cd "$BASE_CODE"
mkdir -p "$RESULTS_BASE"

run_eval() {
    local GPU=$1 NAME=$2 LORA_CKPT=$3
    local OUT_DIR=$RESULTS_BASE/$NAME
    local PORT=$((14200 + GPU))
    mkdir -p "$OUT_DIR"
    echo "[GPU $GPU] $NAME"

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
        --model_max_length 100000 --gradient_checkpointing True --dataloader_num_workers 2 \
        --run_name "$NAME" --report_to none \
        --video_min_frames 64 --video_max_frames 128 --base_interval 0.2 \
        --lora_ckpt "$LORA_CKPT" --no_audio False 2>&1 | tee "$OUT_DIR/inference.log"
    rm -rf "$OUT_DIR/generation_0" 2>/dev/null

    local RF="$OUT_DIR/$NAME/test_results_rank0.json"
    if [ -f "$RF" ]; then
        python3 "$EVAL_SCRIPT" --results "$RF" --test_json "$TEST_JSON" --max_time 60.0 --out_dir "$OUT_DIR"
    fi
}

# 10 new checkpoints: 1.2, 1.4, ..., 3.0 ep
STEPS=(3108 3626 4144 4662 5180 5698 6216 6734 7252 7770)
NAMES=()
CKPTS=()
for i in "${!STEPS[@]}"; do
    ep_x10=$(( 12 + i*2 ))   # 12, 14, ..., 30
    if [ "$ep_x10" -ge 20 ] && [ "$((ep_x10 % 10))" -eq 0 ]; then
        NAMES+=("unav_$((ep_x10/10)).0ep")
    else
        NAMES+=("unav_$((ep_x10/10)).$((ep_x10%10))ep")
    fi
    CKPTS+=("$UNAV_DIR/checkpoint-${STEPS[$i]}")
done

# Run 2 parallel on GPU 4,5
i=0
while [ $i -lt ${#NAMES[@]} ]; do
    PIDS=()
    for gpu in 4 5; do
        if [ $i -lt ${#NAMES[@]} ]; then
            run_eval $gpu "${NAMES[$i]}" "${CKPTS[$i]}" &
            PIDS+=($!)
            i=$((i+1))
        fi
    done
    wait "${PIDS[@]}"
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  UnAV from PU-VALOR 0.2ep low-res — epochs 1.2 to 3.0"
echo "════════════════════════════════════════════════════════════════"
printf "%-20s %8s %8s %8s %8s\n" "Model" "mIoU" "R@0.3" "R@0.5" "R@0.7"
for name in "${NAMES[@]}"; do
    SUMMARY="$RESULTS_BASE/$name/eval_miou_summary.json"
    if [ -f "$SUMMARY" ]; then
        python3 -c "
import json; d=json.load(open('$SUMMARY'))
print(f'  {d[\"mIoU_%\"]:.2f}  {d[\"R@1\"][\"0.3\"]:.2f}  {d[\"R@1\"][\"0.5\"]:.2f}  {d[\"R@1\"][\"0.7\"]:.2f}')
" | while read line; do printf "%-20s %s\n" "$name" "$line"; done
    fi
done
echo "════════════════════════════════════════════════════════════════"
