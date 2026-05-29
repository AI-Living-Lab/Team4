#!/bin/bash
# ============================================================
# v4 ckpt-10612 full PU-VALOR t1t2 (3172) inference + mIoU
#   baseline prompt (no hint), 4+1 tokenization
#   GPU 4,5,6,7 (4-shard parallel)
# ============================================================
set -eo pipefail
source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

export ARNOLD_WORKER_NUM=1 ARNOLD_ID=0 METIS_WORKER_0_HOST=localhost

BASE_CODE=/home/aix23102/audiolm/vS2_eunji/video_SALMONN2_plus
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
DATA_DIR=/home/aix23102/audiolm/vS2_eunji/data
EVAL_PY=/home/aix23102/audiolm/vS2_eunji/eval/eval_miou_v5.py

CKPT=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_v4/checkpoint-10612
MERGED=${CKPT}_merged
TEST_JSON=$DATA_DIR/puvalor_test_v5_t1t2_tok4_1.json
TEST_JSON_RAW=$DATA_DIR/puvalor_test_v5_t1t2.json

RESULTS_BASE=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_v4/eval_full_t1t2_checkpoint-10612_baseline
SHARD_DIR=$RESULTS_BASE/shards
mkdir -p "$RESULTS_BASE" "$SHARD_DIR"

GPUS=(4 5 6 7)
N_SHARDS=4

echo "[start] $(date)"
echo "[v4 full] CKPT=$CKPT"
echo "[v4 full] TEST=$TEST_JSON ($(python3 -c "import json; print(len(json.load(open('$TEST_JSON'))))") samples)"
echo "[v4 full] RESULTS=$RESULTS_BASE"
echo "[v4 full] merged exists? $([ -f $MERGED/config.json ] && echo yes || echo NO)"

# ---- (1) Shard test JSON ----
echo "========== SHARD =========="
N_SHARDS=$N_SHARDS python3 - <<PYEOF
import json, os
N = int(os.environ["N_SHARDS"])
data = json.load(open("$TEST_JSON"))
for r in range(N):
    shard = [data[i] for i in range(len(data)) if i % N == r]
    out = os.path.join("$SHARD_DIR", f"shard{r}.json")
    json.dump(shard, open(out, "w"), ensure_ascii=False)
    print(f"  shard{r}: {len(shard)} -> {out}")
PYEOF

# ---- (2) Parallel inference ----
echo "========== INFERENCE =========="
cd "$BASE_CODE"
pids=()
for r in 0 1 2 3; do
    GPU=${GPUS[$r]}
    OUT_DIR=$RESULTS_BASE/shard${r}
    PORT=$((20500 + GPU))
    mkdir -p "$OUT_DIR"
    echo "  [GPU $GPU rank $r]"
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
echo "  waiting on ${#pids[@]} shards..."
for p in "${pids[@]}"; do wait $p; done

# ---- (3) Reassemble ----
echo "========== REASSEMBLE =========="
N_SHARDS=$N_SHARDS python3 - <<PYEOF
import json, os
N = int(os.environ["N_SHARDS"])
full = json.load(open("$TEST_JSON"))
n = len(full)
shards = []
for r in range(N):
    p = os.path.join("$RESULTS_BASE", f"shard{r}", f"shard{r}", f"test_results_rank{r}.json")
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
raw_full = json.load(open("$TEST_JSON_RAW"))
ordered_test = [raw_full[i] for i in keep_idx]
json.dump(ordered_results, open(os.path.join("$RESULTS_BASE", "results_ordered.json"), "w"), ensure_ascii=False)
json.dump(ordered_test, open(os.path.join("$RESULTS_BASE", "test_ordered.json"), "w"), ensure_ascii=False)
print(f"Reassembled: {len(ordered_results)}")
PYEOF

# ---- (4) mIoU ----
echo "========== mIoU =========="
python3 "$EVAL_PY" \
    --results "$RESULTS_BASE/results_ordered.json" \
    --test_json "$RESULTS_BASE/test_ordered.json" \
    --max_time 9999.0 \
    --out_dir "$RESULTS_BASE"

echo "[done] $(date)"
echo
echo "=== Final ==="
cat "$RESULTS_BASE/eval_miou_summary.json"
