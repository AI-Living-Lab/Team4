#!/bin/bash
# ============================================================
# Wait for orphaned V1 inference to finish, then reassemble V1.
# Then run V3 inference (4 shards parallel on GPU 4,5,6,7), reassemble V3.
# Finally print baseline / V1 / V3 mIoU.
# ============================================================
set -o pipefail
source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

DATA=/home/aix23102/audiolm/vS2_eunji/data
RB=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_v5_b/eval_sub500_variants
LOG=/home/aix23102/audiolm/vS2_eunji/_tools/finish_v5b_variants.log

echo "[start] $(date)" | tee -a "$LOG"

# Wait for V1 to finish (orphaned inference)
echo "[wait] polling V1 shards every 30s..." | tee -a "$LOG"
while true; do
    n_done=0
    for r in 0 1 2 3; do
        f=$RB/v1/shard${r}/v1_s${r}/test_results_rank${r}.json
        [ -f "$f" ] && n_done=$((n_done + 1))
    done
    echo "  V1 shards done: $n_done/4 @ $(date +%H:%M:%S)" | tee -a "$LOG"
    [ "$n_done" -eq 4 ] && break
    sleep 30
done
echo "[V1 done] reassembling..." | tee -a "$LOG"
python3 /home/aix23102/audiolm/vS2_eunji/_tools/reassemble_and_eval_variant.py \
    --variant v1 \
    --test_json $DATA/puvalor_test_v5_t1t2_sub500_tok3_1_v1.json 2>&1 | tee -a "$LOG"

# Launch V3 inference (using same 4 GPUs)
BASE_CODE=/home/aix23102/audiolm/vS2_eunji/video_SALMONN2_plus
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
MERGED=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_v5_b/checkpoint-11371_merged
GPUS=(4 5 6 7)
N_SHARDS=4
V3_TEST=$DATA/puvalor_test_v5_t1t2_sub500_tok3_1_v3.json

export ARNOLD_WORKER_NUM=1 ARNOLD_ID=0 METIS_WORKER_0_HOST=localhost

V3_BASE=$RB/v3
SHARD_DIR=$V3_BASE/shards
mkdir -p "$V3_BASE" "$SHARD_DIR"
echo "[V3 launch] sharding..." | tee -a "$LOG"
python3 - <<PYEOF 2>&1 | tee -a "$LOG"
import json, os
data = json.load(open("$V3_TEST"))
for r in range(4):
    shard = [data[i] for i in range(len(data)) if i % 4 == r]
    json.dump(shard, open(os.path.join("$SHARD_DIR", f"shard{r}.json"), "w"), ensure_ascii=False)
    print(f"  shard{r}: {len(shard)}")
PYEOF

cd "$BASE_CODE"
pids=()
for r in 0 1 2 3; do
    GPU=${GPUS[$r]}
    OUT_DIR=$V3_BASE/shard${r}
    PORT=$((19500 + GPU))
    mkdir -p "$OUT_DIR"
    echo "  [V3] GPU $GPU rank $r" | tee -a "$LOG"
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
        --run_name "v3_s${r}" --report_to none \
        --video_min_frames 64 --video_max_frames 128 --base_interval 0.2 \
        --lora_ckpt "No" --no_audio False > "$OUT_DIR/inference.log" 2>&1 &
    pids+=($!)
done
echo "[V3] waiting for ${#pids[@]} shards..." | tee -a "$LOG"
for p in "${pids[@]}"; do wait $p; done
echo "[V3 done] reassembling..." | tee -a "$LOG"

python3 /home/aix23102/audiolm/vS2_eunji/_tools/reassemble_and_eval_variant.py \
    --variant v3 \
    --test_json $DATA/puvalor_test_v5_t1t2_sub500_tok3_1_v3.json 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "============== SUMMARY ==============" | tee -a "$LOG"
for n in baseline v1 v3; do
    f=$RB/$n/eval_miou_summary.json
    if [ -f "$f" ]; then
        echo "--- $n ---" | tee -a "$LOG"
        cat "$f" | tee -a "$LOG"
        echo | tee -a "$LOG"
    fi
done
echo "[done] $(date)" | tee -a "$LOG"
