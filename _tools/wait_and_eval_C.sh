#!/bin/bash
# C 학습 완료 대기 후 eval 실행
set -eo pipefail

echo "[INFO] $(date) Waiting for C training..."
while pgrep -f "ablation_C_no_puvalor" > /dev/null; do
    sleep 60
done
echo "[INFO] $(date) C training finished!"
sleep 10

# Run eval on C checkpoints
source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus
export ARNOLD_WORKER_NUM=1 ARNOLD_ID=0 METIS_WORKER_0_HOST=localhost

BASE_CODE=/home/aix23102/audiolm/video-SALMONN-2/video_SALMONN2_plus
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
TEST_JSON=/home/aix23102/audiolm/vS2_eunji/data/unav100_test_multiseg_sub80.json
EVAL_SCRIPT=/home/aix23102/audiolm/vS2_eunji/eval/eval_miou_multiseg.py
RESULTS_BASE=/home/aix23102/audiolm/vS2_eunji/eval/results/ablation_A_vs_B
C_DIR=/data0/aix23102/checkpoints_open_aligner/ablation_C_no_puvalor_highres

cd "$BASE_CODE"

run_eval() {
    local GPU=$1 NAME=$2 LORA_CKPT=$3
    local OUT_DIR=$RESULTS_BASE/$NAME
    local PORT=$((13900 + GPU))
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
        --model_max_length 100000 --gradient_checkpointing True --dataloader_num_workers 2 \
        --run_name "$NAME" --report_to none \
        --video_min_frames 64 --video_max_frames 128 --base_interval 0.2 \
        --lora_ckpt "$LORA_CKPT" --no_audio False 2>&1 | tee "$OUT_DIR/inference.log"
    rm -rf "$OUT_DIR/generation_0" 2>/dev/null
    local RF="$OUT_DIR/$NAME/test_results_rank0.json"
    if [ -f "$RF" ]; then
        python3 "$EVAL_SCRIPT" --results "$RF" --test_json "$TEST_JSON" --max_time 60.0 --out_dir "$OUT_DIR"
        echo "[GPU $GPU] ✓ $NAME done"
    fi
}

run_eval 4 "C_no_puvalor_0.1ep" "$C_DIR/checkpoint-259" &
run_eval 5 "C_no_puvalor_0.2ep" "$C_DIR/checkpoint-518" &
wait

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  FINAL: PU-VALOR Pretraining Effect Analysis"
echo "════════════════════════════════════════════════════════════════"
printf "%-30s %8s %8s %8s %8s\n" "Model" "mIoU" "R@0.3" "R@0.5" "R@0.7"
echo "────────────────────────────────────────────────────────────────"
for n in C_no_puvalor_0.1ep C_no_puvalor_0.2ep A_qkv_0.1ep A_qkv_0.2ep; do
    SUMMARY="$RESULTS_BASE/$n/eval_miou_summary.json"
    if [ -f "$SUMMARY" ]; then
        python3 -c "
import json; d=json.load(open('$SUMMARY'))
print(f'  {d[\"mIoU_%\"]:.2f}  {d[\"R@1\"][\"0.3\"]:.2f}  {d[\"R@1\"][\"0.5\"]:.2f}  {d[\"R@1\"][\"0.7\"]:.2f}')
" | while read line; do printf "%-30s %s\n" "$n" "$line"; done
    fi
done
echo "════════════════════════════════════════════════════════════════"
