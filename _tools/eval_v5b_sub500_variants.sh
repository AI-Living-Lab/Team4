#!/bin/bash
# ============================================================
# v5_b sub500 prompt-variant comparison
#   3 variants: baseline / V1 / V3
#   Runs sequentially on GPU 4,5,6,7 (4-shard parallel each)
# ============================================================
set -eo pipefail
source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

V5B_CKPT=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_v5_b/checkpoint-11371
LOG_DIR=/home/aix23102/audiolm/vS2_eunji/_tools/eval_v5b_variants_logs
mkdir -p "$LOG_DIR"
echo "[start] $(date)"

# We will inline the variant test_json via a modified eval_v5_sub500.sh approach.
# Simpler: copy eval_v5_sub500.sh logic, but parametrize TEST_JSON path.

BASE_CODE=/home/aix23102/audiolm/vS2_eunji/video_SALMONN2_plus
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
DATA_DIR=/home/aix23102/audiolm/vS2_eunji/data
EVAL_PY=/home/aix23102/audiolm/vS2_eunji/eval/eval_miou_v5.py
TEST_JSON_RAW=$DATA_DIR/puvalor_test_v5_t1t2_sub500.json  # gt_segments
MERGED=${V5B_CKPT}_merged

if [ ! -f "$MERGED/config.json" ]; then
    echo "[ERROR] merged model not found: $MERGED"
    exit 1
fi

# 4-GPU shards
export ARNOLD_WORKER_NUM=1 ARNOLD_ID=0 METIS_WORKER_0_HOST=localhost
GPUS=(4 5 6 7)
N_SHARDS=4

run_variant () {
    local NAME=$1
    local TEST_JSON=$2
    echo "========== variant=$NAME =========="
    echo "[$NAME] test_json: $TEST_JSON"
    local RESULTS_BASE=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_v5_b/eval_sub500_variants/${NAME}
    local SHARD_DIR=$RESULTS_BASE/shards
    mkdir -p "$RESULTS_BASE" "$SHARD_DIR"

    # shard JSON
    N_SHARDS=$N_SHARDS TEST_JSON=$TEST_JSON SHARD_DIR=$SHARD_DIR python3 - <<PYEOF
import json, os
N = int(os.environ["N_SHARDS"])
data = json.load(open(os.environ["TEST_JSON"]))
for r in range(N):
    shard = [data[i] for i in range(len(data)) if i % N == r]
    json.dump(shard, open(os.path.join(os.environ["SHARD_DIR"], f"shard{r}.json"), "w"), ensure_ascii=False)
    print(f"  shard{r}: {len(shard)}")
PYEOF

    # parallel inference
    cd "$BASE_CODE"
    pids=()
    for r in $(seq 0 $((N_SHARDS - 1))); do
        local GPU=${GPUS[$r]}
        local OUT_DIR=$RESULTS_BASE/shard${r}
        local PORT=$((19500 + GPU))
        mkdir -p "$OUT_DIR"
        echo "[$NAME] GPU $GPU rank $r"
        CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$PORT \
            qwenvl/train/train_qwen.py \
            --model_base "$MODEL_BASE" --run_test True --pred_rank $r \
            --deepspeed scripts/zero2.json \
            --model_name_or_path "$MERGED" \
            --dataset_use "$SHARD_DIR/shard${r}.json" --bf16 --output_dir "$OUT_DIR" \
            --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
            --max_pixels 176400 --min_pixels 784 \
            --video_max_frame_pixels 28224 --video_min_frame_pixels 784 \
            --eval_strategy "no" --save_strategy "no" --learning_rate 1e-5 \
            --model_max_length 100000 --gradient_checkpointing True --dataloader_num_workers 2 \
            --run_name "${NAME}_s${r}" --report_to none \
            --video_min_frames 64 --video_max_frames 128 --base_interval 0.2 \
            --lora_ckpt "No" --no_audio False > "$OUT_DIR/inference.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p; done

    # reassemble
    N_SHARDS=$N_SHARDS TEST_JSON=$TEST_JSON TEST_JSON_RAW=$TEST_JSON_RAW \
        RESULTS_BASE=$RESULTS_BASE python3 - <<PYEOF
import json, os
N = int(os.environ["N_SHARDS"])
full = json.load(open(os.environ["TEST_JSON"]))
n = len(full)
shards = []
for r in range(N):
    p = os.path.join(os.environ["RESULTS_BASE"], f"shard{r}", f"shard{r}", f"test_results_rank{r}.json")
    if not os.path.exists(p):
        print(f"MISSING: {p}"); shards.append([]); continue
    shards.append(json.load(open(p)))
out = [None]*n
for i in range(n):
    r = i % N; pos = i // N
    if pos < len(shards[r]):
        out[i] = shards[r][pos]
keep_idx = [i for i, x in enumerate(out) if x is not None]
ordered_results = [out[i] for i in keep_idx]
raw_full = json.load(open(os.environ["TEST_JSON_RAW"]))
ordered_test = [raw_full[i] for i in keep_idx]
json.dump(ordered_results, open(os.path.join(os.environ["RESULTS_BASE"], "results_ordered.json"), "w"), ensure_ascii=False)
json.dump(ordered_test, open(os.path.join(os.environ["RESULTS_BASE"], "test_ordered.json"), "w"), ensure_ascii=False)
print(f"Reassembled: {len(ordered_results)}")
PYEOF

    python3 "$EVAL_PY" \
        --results "$RESULTS_BASE/results_ordered.json" \
        --test_json "$RESULTS_BASE/test_ordered.json" \
        --max_time 9999.0 \
        --out_dir "$RESULTS_BASE"
    echo "[$NAME] DONE → $RESULTS_BASE"
    echo
}

run_variant baseline $DATA_DIR/puvalor_test_v5_t1t2_sub500_tok3_1.json    2>&1 | tee "$LOG_DIR/baseline.log"
run_variant v1       $DATA_DIR/puvalor_test_v5_t1t2_sub500_tok3_1_v1.json 2>&1 | tee "$LOG_DIR/v1.log"
run_variant v3       $DATA_DIR/puvalor_test_v5_t1t2_sub500_tok3_1_v3.json 2>&1 | tee "$LOG_DIR/v3.log"

echo "[done] $(date)"
echo
echo "=== Summary ==="
for n in baseline v1 v3; do
    f=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_v5_b/eval_sub500_variants/${n}/eval_miou_summary.json
    if [ -f "$f" ]; then
        echo "--- $n ---"
        cat "$f"
        echo
    fi
done
