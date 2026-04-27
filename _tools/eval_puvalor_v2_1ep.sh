#!/bin/bash
# ============================================================
# salmonn2plus_puvalor_v2_1ep (best=checkpoint-15160) 평가
#   - GPU 4,5 에서 2-shard 병렬 inference
#   - PU-VALOR GOUT (single+multi 통합) + UnAV-100
#   - eval_miou_subgroup.py 로 overall/single/multi mIoU 리포트
# ============================================================
set -eo pipefail

source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

export ARNOLD_WORKER_NUM=1 ARNOLD_ID=0 METIS_WORKER_0_HOST=localhost

BASE_CODE=/home/aix23102/audiolm/vS2_eunji/video_SALMONN2_plus
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
SHARD_DIR=/home/aix23102/audiolm/vS2_eunji/data/eval_shards
RESULTS_BASE=/home/aix23102/audiolm/vS2_eunji/eval/results/puvalor_v2_1ep_best
MERGED_MODEL=$RESULTS_BASE/merged_model
EVAL_SCRIPT=/home/aix23102/audiolm/vS2_eunji/eval/eval_miou_subgroup.py

cd "$BASE_CODE"
mkdir -p "$RESULTS_BASE"

# GPU list (positional → shard rank)
GPUS=(4 5)

run_shard() {
    local DATASET_NAME=$1   # e.g. puvalor_gout or unav100
    local RANK=$2
    local GPU=${GPUS[$RANK]}
    local TEST_JSON=$SHARD_DIR/${DATASET_NAME}_shard${RANK}.json
    local OUT_DIR=$RESULTS_BASE/${DATASET_NAME}
    local RUN_NAME=shard${RANK}
    local PORT=$((15000 + GPU))

    mkdir -p "$OUT_DIR"
    echo "[GPU $GPU / RANK $RANK] $DATASET_NAME"

    CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$PORT \
        qwenvl/train/train_qwen.py \
        --model_base "$MODEL_BASE" --run_test True --pred_rank $RANK \
        --deepspeed scripts/zero2.json \
        --model_name_or_path "$MERGED_MODEL" \
        --dataset_use "$TEST_JSON" --bf16 --output_dir "$OUT_DIR" \
        --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
        --max_pixels 176400 --min_pixels 784 \
        --video_max_frame_pixels 28224 --video_min_frame_pixels 784 \
        --eval_strategy "no" --save_strategy "no" --learning_rate 1e-5 \
        --model_max_length 100000 --gradient_checkpointing True --dataloader_num_workers 2 \
        --run_name "$RUN_NAME" --report_to none \
        --video_min_frames 64 --video_max_frames 128 --base_interval 0.2 \
        --lora_ckpt "No" --no_audio False 2>&1 | tee "$OUT_DIR/inference_rank${RANK}.log"
}

merge_and_eval() {
    local DATASET_NAME=$1
    local TEST_JSON_FULL=$2
    local OUT_DIR=$RESULTS_BASE/${DATASET_NAME}

    echo "[MERGE] $DATASET_NAME"
    python3 - <<PYEOF
import json, os
OUT = "$OUT_DIR"
DN = "$DATASET_NAME"
N_SHARDS = 2
full_path = "$TEST_JSON_FULL"
full = json.load(open(full_path))
n = len(full)
shards = []
for r in range(N_SHARDS):
    p = os.path.join(OUT, f"shard{r}", f"test_results_rank{r}.json")
    if not os.path.exists(p):
        print(f"MISSING: {p}"); continue
    shards.append(json.load(open(p)))
reassembled = [None] * n
for i in range(n):
    r = i % N_SHARDS
    pos = i // N_SHARDS
    if pos >= len(shards[r]):
        print(f"WARN: shard{r} pos {pos} OOR (len={len(shards[r])})")
        continue
    reassembled[i] = shards[r][pos]
reassembled = [x for x in reassembled if x is not None]
print(f"Reassembled: {len(reassembled)}")
out_merged = os.path.join(OUT, "merged_results.json")
with open(out_merged, "w") as f:
    json.dump(reassembled, f, ensure_ascii=False)
print(f"Saved: {out_merged}")
PYEOF

    echo "[EVAL] $DATASET_NAME"
    python3 "$EVAL_SCRIPT" \
        --results "$OUT_DIR/merged_results.json" \
        --test_json "$TEST_JSON_FULL" \
        --max_time 9999.0 \
        --out_dir "$OUT_DIR"
}

# ---- Dataset 1: PU-VALOR GOUT ----
echo "========== PU-VALOR GOUT =========="
for r in 0 1; do
    run_shard puvalor_gout $r &
done
wait
merge_and_eval puvalor_gout /home/aix23102/audiolm/vS2_eunji/data/puvalor_test_gout.json

# ---- Dataset 2: UnAV-100 ----
echo "========== UnAV-100 =========="
for r in 0 1; do
    run_shard unav100 $r &
done
wait
merge_and_eval unav100 /home/aix23102/audiolm/vS2_eunji/data/unav100_test_multiseg_salmonn2plus.json

echo "========== ALL DONE =========="
echo "Results: $RESULTS_BASE"
