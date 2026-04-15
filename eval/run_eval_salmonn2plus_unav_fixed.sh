#!/usr/bin/env bash
# ============================================================
# SALMONN2+ UnAV 모델 평가 (PU-VALOR merge된 base 사용)
# GPU 4,5,6,7 — 3개 UnAV ckpt 병렬 + PU-VALOR 0.3ep 재확인
# ============================================================
set -eo pipefail

source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

BASE_CODE=/home/aix23102/audiolm/video-SALMONN-2/video_SALMONN2_plus
# PU-VALOR가 merge된 base model
MERGED_BASE=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_unav100_multiseg/base
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
UNAV_DIR=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_unav100_multiseg
TEST_JSON=/home/aix23102/audiolm/vS2_eunji/data/unav100_test_multiseg_sub80.json
EVAL_SCRIPT=/home/aix23102/audiolm/vS2_eunji/eval/eval_miou_multiseg.py
RESULTS_BASE=/home/aix23102/audiolm/vS2_eunji/eval/results/salmonn2plus_unav_fixed

cd "$BASE_CODE"

run_eval() {
    local GPU=$1
    local NAME=$2
    local MODEL_PATH=$3
    local LORA_CKPT=$4
    local OUT_DIR=$RESULTS_BASE/$NAME
    local PORT=$((12950 + GPU))

    mkdir -p "$OUT_DIR"
    echo "[GPU $GPU] Starting $NAME"

    CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$PORT \
        qwenvl/train/train_qwen.py \
        --model_base "$MODEL_BASE" \
        --run_test True \
        --pred_rank 0 \
        --deepspeed scripts/zero2.json \
        --model_name_or_path "$MODEL_PATH" \
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

    RESULT_FILE=$(find "$OUT_DIR" -name "test_results_rank0.json" 2>/dev/null | head -1)
    if [ -f "$RESULT_FILE" ]; then
        python3 "$EVAL_SCRIPT" --results "$RESULT_FILE" --test_json "$TEST_JSON" --max_time 60.0 --out_dir "$OUT_DIR"
        echo "[GPU $GPU] ✓ $NAME done"
    else
        echo "[GPU $GPU] ✗ $NAME: results not found"
    fi
}

echo "================================================"
echo "  SALMONN2+ UnAV eval (PU-VALOR merged base)"
echo "  GPUs: 4,5,6,7"
echo "================================================"

rm -rf "$RESULTS_BASE" 2>/dev/null
mkdir -p "$RESULTS_BASE"

# 4개 동시: UnAV 3개 (merged base + UnAV LoRA) + PU-VALOR 0.3ep (base + PU-VALOR LoRA)
run_eval 4 "unav_0.39ep"  "$MERGED_BASE" "$UNAV_DIR/checkpoint-1000" &
run_eval 5 "unav_0.77ep"  "$MERGED_BASE" "$UNAV_DIR/checkpoint-2000" &
run_eval 6 "unav_1.0ep"   "$MERGED_BASE" "$UNAV_DIR/checkpoint-2589" &
run_eval 7 "puvalor_0.3ep_recheck" "$MODEL_BASE" "/data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_0.3ep/checkpoint-7581" &
wait

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "════════════════════════════════════════════════════════════════════"
for DIR in $RESULTS_BASE/*/; do
    NAME=$(basename $DIR)
    SUMMARY=$DIR/eval_miou_summary.json
    if [ -f "$SUMMARY" ]; then
        python3 -c "
import json; d=json.load(open('$SUMMARY'))
n=d['n_samples']; pok=d['parse_ok']
r=d['R@1']
print(f'  $NAME: mIoU={d[\"mIoU_%\"]:.2f}% R@0.3={r[\"0.3\"]:.2f}% R@0.5={r[\"0.5\"]:.2f}% R@0.7={r[\"0.7\"]:.2f}% Parse={pok*100//n}%')
"
    fi
done
echo "════════════════════════════════════════════════════════════════════"
