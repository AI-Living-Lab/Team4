#!/bin/bash
# ============================================================
# v4 + v5_b on UnAV-100 test (3455 T2 TSG samples)
#   - v4 (4+1):  ckpt-10612_merged + unav_test_t2_tok4_1.json
#   - v5_b (3+1): ckpt-11371_merged + unav_test_t2_tok3_1.json
#   - GPU 4,5,6,7 (4-shard parallel each model, sequential)
#
# Waits for v6_a training (train_v6_a.sh) to finish before starting.
# ============================================================
set -o pipefail
source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

LOG=/home/aix23102/audiolm/vS2_eunji/_tools/eval_unav_v4_v5b.log
echo "[start] $(date)" | tee "$LOG"

# ---- Wait for v6_a script to exit (poll PID 2111004) ----
V6A_PID=2111004
echo "[wait] polling v6_a PID=$V6A_PID every 60s..." | tee -a "$LOG"
while kill -0 $V6A_PID 2>/dev/null; do
    sleep 60
done
echo "[wait] v6_a finished at $(date)" | tee -a "$LOG"

# Safety: wait 30s for GPU cleanup
sleep 30

export ARNOLD_WORKER_NUM=1 ARNOLD_ID=0 METIS_WORKER_0_HOST=localhost
BASE_CODE=/home/aix23102/audiolm/vS2_eunji/video_SALMONN2_plus
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
EVAL_PY=/home/aix23102/audiolm/vS2_eunji/eval/eval_miou_v5.py
DATA_DIR=/home/aix23102/audiolm/vS2_eunji/data
TEST_JSON_RAW=$DATA_DIR/unav_test_t2.json
GPUS=(4 5 6 7)
N_SHARDS=4

run_unav () {
    local TAG=$1
    local MERGED=$2
    local TOK_TEST=$3
    local OUT_PARENT=$4
    echo "==================================================================" | tee -a "$LOG"
    echo "[$TAG] $(date)" | tee -a "$LOG"
    echo "[$TAG] merged: $MERGED" | tee -a "$LOG"
    echo "[$TAG] test  : $TOK_TEST" | tee -a "$LOG"

    local RESULTS_BASE=$OUT_PARENT/eval_unav_t2_$TAG
    local SHARD_DIR=$RESULTS_BASE/shards
    mkdir -p "$RESULTS_BASE" "$SHARD_DIR"

    # shard
    echo "[$TAG] sharding $TOK_TEST into $N_SHARDS..." | tee -a "$LOG"
    N_SHARDS=$N_SHARDS TEST=$TOK_TEST SHARD_DIR=$SHARD_DIR python3 - <<PYEOF 2>&1 | tee -a "$LOG"
import json, os
N = int(os.environ["N_SHARDS"])
data = json.load(open(os.environ["TEST"]))
for r in range(N):
    shard = [data[i] for i in range(len(data)) if i % N == r]
    json.dump(shard, open(os.path.join(os.environ["SHARD_DIR"], f"shard{r}.json"), "w"), ensure_ascii=False)
    print(f"  shard{r}: {len(shard)}")
PYEOF

    # parallel inference
    cd "$BASE_CODE"
    pids=()
    for r in 0 1 2 3; do
        GPU=${GPUS[$r]}
        OUT_DIR=$RESULTS_BASE/shard${r}
        PORT=$((21500 + GPU))
        mkdir -p "$OUT_DIR"
        echo "[$TAG] GPU $GPU rank $r" | tee -a "$LOG"
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
            --run_name "shard${r}" --report_to none \
            --video_min_frames 64 --video_max_frames 128 --base_interval 0.2 \
            --lora_ckpt "No" --no_audio False > "$OUT_DIR/inference.log" 2>&1 &
        pids+=($!)
    done
    echo "[$TAG] waiting ${#pids[@]} shards..." | tee -a "$LOG"
    for p in "${pids[@]}"; do wait $p; done

    # reassemble
    echo "[$TAG] reassembling..." | tee -a "$LOG"
    N_SHARDS=$N_SHARDS TEST=$TOK_TEST RAW=$TEST_JSON_RAW RB=$RESULTS_BASE python3 - <<PYEOF 2>&1 | tee -a "$LOG"
import json, os
N = int(os.environ["N_SHARDS"])
full = json.load(open(os.environ["TEST"]))
shards = []
for r in range(N):
    p = os.path.join(os.environ["RB"], f"shard{r}", f"shard{r}", f"test_results_rank{r}.json")
    if not os.path.exists(p):
        print(f"MISSING: {p}"); shards.append([]); continue
    shards.append(json.load(open(p)))
n = len(full)
out = [None]*n
for i in range(n):
    r = i % N; pos = i // N
    if pos < len(shards[r]): out[i] = shards[r][pos]
keep_idx = [i for i, x in enumerate(out) if x is not None]
ordered_results = [out[i] for i in keep_idx]
raw_full = json.load(open(os.environ["RAW"]))
ordered_test = [raw_full[i] for i in keep_idx]
json.dump(ordered_results, open(os.path.join(os.environ["RB"], "results_ordered.json"), "w"), ensure_ascii=False)
json.dump(ordered_test, open(os.path.join(os.environ["RB"], "test_ordered.json"), "w"), ensure_ascii=False)
print(f"Reassembled: {len(ordered_results)}")
PYEOF

    # mIoU
    python3 "$EVAL_PY" \
        --results "$RESULTS_BASE/results_ordered.json" \
        --test_json "$RESULTS_BASE/test_ordered.json" \
        --max_time 9999.0 \
        --out_dir "$RESULTS_BASE" 2>&1 | tee -a "$LOG"
    echo "[$TAG] DONE → $RESULTS_BASE" | tee -a "$LOG"
}

# v4
run_unav "v4_ckpt10612" \
    "/data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_v4/checkpoint-10612_merged" \
    "$DATA_DIR/unav_test_t2_tok4_1.json" \
    "/data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_v4"

# v5_b
run_unav "v5b_ckpt11371" \
    "/data0/aix23102/checkpoints_open_aligner/salmonn2plus_v5_b/checkpoint-11371_merged" \
    "$DATA_DIR/unav_test_t2_tok3_1.json" \
    "/data0/aix23102/checkpoints_open_aligner/salmonn2plus_v5_b"

echo | tee -a "$LOG"
echo "============== UnAV SUMMARY ==============" | tee -a "$LOG"
for d in \
    /data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_v4/eval_unav_t2_v4_ckpt10612 \
    /data0/aix23102/checkpoints_open_aligner/salmonn2plus_v5_b/eval_unav_t2_v5b_ckpt11371; do
    if [ -f "$d/eval_miou_summary.json" ]; then
        echo "--- $(basename $d) ---" | tee -a "$LOG"
        cat "$d/eval_miou_summary.json" | tee -a "$LOG"
        echo | tee -a "$LOG"
    fi
done
echo "[done] $(date)" | tee -a "$LOG"
