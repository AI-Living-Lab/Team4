#!/bin/bash
# ============================================================
# Full pipeline (자동):
#   1. PU-VALOR 0.5ep high-res 학습 완료 대기
#   2. 5개 PU-VALOR checkpoint eval (0.1~0.5ep)
#   3. Best ckpt 선택
#   4. UnAV-100 1ep high-res 학습 (best PU-VALOR ckpt 기반, save every 0.1ep)
#   5. 10개 UnAV checkpoint eval (0.1~1.0ep)
#   6. 최종 summary
# ============================================================
set -eo pipefail

source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

export ARNOLD_WORKER_NUM=1 ARNOLD_ID=0 METIS_WORKER_0_HOST=localhost

BASE_CODE=/home/aix23102/audiolm/video-SALMONN-2/video_SALMONN2_plus
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
TEST_JSON=/home/aix23102/audiolm/vS2_eunji/data/unav100_test_multiseg_sub80.json
EVAL_SCRIPT=/home/aix23102/audiolm/vS2_eunji/eval/eval_miou_multiseg.py

PUVALOR_DIR=/data0/aix23102/checkpoints_open_aligner/puvalor_highres_0.5ep
UNAV_DIR=/data0/aix23102/checkpoints_open_aligner/unav100_highres_1.0ep
PUVALOR_RESULTS=/home/aix23102/audiolm/vS2_eunji/eval/results/puvalor_highres_eval
UNAV_RESULTS=/home/aix23102/audiolm/vS2_eunji/eval/results/unav_highres_eval

mkdir -p "$PUVALOR_RESULTS" "$UNAV_RESULTS"

# ============================================================
# [1] Wait for PU-VALOR training to finish
# ============================================================
echo "[PIPELINE] $(date) Waiting for PU-VALOR training to produce final checkpoint..."
# Wait until 5 checkpoints exist (means training finished all planned saves)
while [ "$(ls -d "$PUVALOR_DIR"/checkpoint-* 2>/dev/null | wc -l)" -lt 5 ]; do
    sleep 300
done
# Extra safety: wait until no train_qwen process for puvalor_highres is running
while pgrep -f "train_qwen.*puvalor_highres_0.5ep" > /dev/null 2>&1; do
    sleep 60
done
echo "[PIPELINE] $(date) PU-VALOR training finished!"
sleep 30

# ============================================================
# [2] Eval 5 PU-VALOR checkpoints
# ============================================================
cd "$BASE_CODE"

run_eval() {
    local GPU=$1 NAME=$2 LORA_CKPT=$3 OUT_BASE=$4
    local OUT_DIR=$OUT_BASE/$NAME
    local PORT=$((14000 + GPU))
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
        --video_min_frames 64 --video_max_frames 256 --base_interval 0.2 \
        --lora_ckpt "$LORA_CKPT" --no_audio False 2>&1 | tee "$OUT_DIR/inference.log"
    rm -rf "$OUT_DIR/generation_0" 2>/dev/null

    local RF="$OUT_DIR/$NAME/test_results_rank0.json"
    if [ -f "$RF" ]; then
        python3 "$EVAL_SCRIPT" --results "$RF" --test_json "$TEST_JSON" --max_time 60.0 --out_dir "$OUT_DIR"
    fi
}

echo "[PIPELINE] $(date) Starting PU-VALOR checkpoint eval..."

PUVALOR_CKPTS=($(ls -d "$PUVALOR_DIR"/checkpoint-* 2>/dev/null | sort -V))
STEPS_PER_01EP=3369
PUVALOR_NAMES=()
for ckpt in "${PUVALOR_CKPTS[@]}"; do
    step=$(basename "$ckpt" | sed 's/checkpoint-//')
    ep_x10=$(( (step + STEPS_PER_01EP/2) / STEPS_PER_01EP ))
    PUVALOR_NAMES+=("puvalor_0.${ep_x10}ep")
done

# Run up to 3 parallel (GPU 4,5,6)
i=0
while [ $i -lt ${#PUVALOR_NAMES[@]} ]; do
    PIDS=()
    for gpu in 4 5 6; do
        if [ $i -lt ${#PUVALOR_NAMES[@]} ]; then
            run_eval $gpu "${PUVALOR_NAMES[$i]}" "${PUVALOR_CKPTS[$i]}" "$PUVALOR_RESULTS" &
            PIDS+=($!)
            i=$((i+1))
        fi
    done
    wait "${PIDS[@]}"
done

# ============================================================
# [3] Find best PU-VALOR checkpoint by mIoU
# ============================================================
echo "[PIPELINE] $(date) Finding best PU-VALOR checkpoint..."
BEST_NAME=""
BEST_MIOU=-1
BEST_IDX=-1
for idx in "${!PUVALOR_NAMES[@]}"; do
    name="${PUVALOR_NAMES[$idx]}"
    SUMMARY="$PUVALOR_RESULTS/$name/eval_miou_summary.json"
    if [ -f "$SUMMARY" ]; then
        miou=$(python3 -c "import json; print(json.load(open('$SUMMARY'))['mIoU_%'])")
        echo "  $name: mIoU = $miou"
        is_better=$(python3 -c "print(float('$miou') > float('$BEST_MIOU'))")
        if [ "$is_better" = "True" ]; then
            BEST_MIOU=$miou
            BEST_NAME=$name
            BEST_IDX=$idx
        fi
    fi
done
BEST_CKPT="${PUVALOR_CKPTS[$BEST_IDX]}"
echo "[PIPELINE] BEST PU-VALOR: $BEST_NAME (mIoU=$BEST_MIOU) → $BEST_CKPT"

# ============================================================
# [4] UnAV-100 1ep training from best PU-VALOR
# ============================================================
echo "[PIPELINE] $(date) Starting UnAV-100 1ep training..."

# UnAV-100: 10358 samples, effective_batch=3 (bs=1, accum=1, gpu=3)
# steps_per_ep = 10358/3 ≈ 3453, 1ep = 3453 steps
# save every 0.1ep = 345 steps

export CUDA_VISIBLE_DEVICES=4,5,6
export ARNOLD_WORKER_GPU=3

mkdir -p "$UNAV_DIR"
torchrun --standalone --nproc_per_node=3 \
    qwenvl/train/train_qwen.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "$MODEL_BASE" \
    --model_base "$MODEL_BASE" \
    --dataset_use /home/aix23102/audiolm/vS2_eunji/data/unav100_train_multiseg_salmonn2plus.json \
    --tune_mm_vision False --tune_mm_mlp False --tune_mm_llm False --tune_mm_audio False --tune_mm_qformer False \
    --use_lora True --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 \
    --lora_ckpt "$BEST_CKPT" \
    --bf16 \
    --output_dir "$UNAV_DIR" \
    --num_train_epochs 1 --max_steps 3453 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
    --max_pixels 176400 --min_pixels 784 \
    --video_max_frame_pixels 28224 --video_min_frame_pixels 784 \
    --eval_strategy "no" --save_strategy "steps" --save_steps 345 --save_total_limit 11 \
    --learning_rate 5e-5 --weight_decay 0 --warmup_ratio 0.03 --max_grad_norm 1 --lr_scheduler_type cosine \
    --logging_steps 1 --model_max_length 100000 --gradient_checkpointing True --dataloader_num_workers 4 \
    --run_name unav_highres_from_best --report_to wandb \
    --video_min_frames 64 --video_max_frames 256 --base_interval 0.2 \
    --train_type sft --no_audio False \
    2>&1 | tee "$UNAV_DIR/train.log"

echo "[PIPELINE] $(date) UnAV-100 training finished!"
sleep 15

# ============================================================
# [5] Eval 10 UnAV checkpoints
# ============================================================
echo "[PIPELINE] $(date) Starting UnAV checkpoint eval..."

UNAV_CKPTS=($(ls -d "$UNAV_DIR"/checkpoint-* 2>/dev/null | sort -V))
UNAV_STEPS_PER_01EP=345
UNAV_NAMES=()
for ckpt in "${UNAV_CKPTS[@]}"; do
    step=$(basename "$ckpt" | sed 's/checkpoint-//')
    ep_x10=$(( (step + UNAV_STEPS_PER_01EP/2) / UNAV_STEPS_PER_01EP ))
    if [ "$ep_x10" -eq 10 ]; then
        UNAV_NAMES+=("unav_1.0ep")
    else
        UNAV_NAMES+=("unav_0.${ep_x10}ep")
    fi
done

i=0
while [ $i -lt ${#UNAV_NAMES[@]} ]; do
    PIDS=()
    for gpu in 4 5 6; do
        if [ $i -lt ${#UNAV_NAMES[@]} ]; then
            run_eval $gpu "${UNAV_NAMES[$i]}" "${UNAV_CKPTS[$i]}" "$UNAV_RESULTS" &
            PIDS+=($!)
            i=$((i+1))
        fi
    done
    wait "${PIDS[@]}"
done

# ============================================================
# [6] Final Summary
# ============================================================
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  FINAL SUMMARY: PU-VALOR 0.5ep high-res → UnAV-100 1ep high-res"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "--- PU-VALOR checkpoints (base for UnAV) ---"
printf "%-25s %8s %8s %8s %8s\n" "Model" "mIoU" "R@0.3" "R@0.5" "R@0.7"
for name in "${PUVALOR_NAMES[@]}"; do
    SUMMARY="$PUVALOR_RESULTS/$name/eval_miou_summary.json"
    if [ -f "$SUMMARY" ]; then
        python3 -c "
import json; d=json.load(open('$SUMMARY'))
print(f'  {d[\"mIoU_%\"]:.2f}  {d[\"R@1\"][\"0.3\"]:.2f}  {d[\"R@1\"][\"0.5\"]:.2f}  {d[\"R@1\"][\"0.7\"]:.2f}')
" | while read line; do printf "%-25s %s\n" "$name" "$line"; done
    fi
done

echo ""
echo "--- UnAV checkpoints (from best PU-VALOR: $BEST_NAME) ---"
printf "%-25s %8s %8s %8s %8s\n" "Model" "mIoU" "R@0.3" "R@0.5" "R@0.7"
for name in "${UNAV_NAMES[@]}"; do
    SUMMARY="$UNAV_RESULTS/$name/eval_miou_summary.json"
    if [ -f "$SUMMARY" ]; then
        python3 -c "
import json; d=json.load(open('$SUMMARY'))
print(f'  {d[\"mIoU_%\"]:.2f}  {d[\"R@1\"][\"0.3\"]:.2f}  {d[\"R@1\"][\"0.5\"]:.2f}  {d[\"R@1\"][\"0.7\"]:.2f}')
" | while read line; do printf "%-25s %s\n" "$name" "$line"; done
    fi
done
echo "════════════════════════════════════════════════════════════════════════"
echo "[PIPELINE] $(date) All done!"
