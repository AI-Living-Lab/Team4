#!/bin/bash
# ============================================================
# v6_a on UnAV-100 test (3455 T2 TSG samples)
#   - Polls v6_a training (PID 2111004) every 60s; runs after training finishes
#   - Merges latest v6_a checkpoint (load_best_model_at_end=True → latest = best)
#   - GPU 4,5,6,7 (4-shard parallel)
#   - 3+1 tokenization → unav_test_t2_tok3_1.json (same as v5_b)
# ============================================================
set -o pipefail
source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

LOG=/home/aix23102/audiolm/vS2_eunji/_tools/eval_unav_v6a_after.log
echo "[start] $(date)" | tee "$LOG"

V6A_PID=2111004
echo "[wait] polling v6_a PID=$V6A_PID every 60s..." | tee -a "$LOG"
while kill -0 $V6A_PID 2>/dev/null; do
    sleep 60
done
echo "[wait] v6_a finished at $(date)" | tee -a "$LOG"
sleep 30  # GPU cleanup margin

export ARNOLD_WORKER_NUM=1 ARNOLD_ID=0 METIS_WORKER_0_HOST=localhost
BASE_CODE=/home/aix23102/audiolm/vS2_eunji/video_SALMONN2_plus
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
EVAL_PY=/home/aix23102/audiolm/vS2_eunji/eval/eval_miou_v5.py
DATA_DIR=/home/aix23102/audiolm/vS2_eunji/data
TEST_JSON_RAW=$DATA_DIR/unav_test_t2.json
TOK_TEST=$DATA_DIR/unav_test_t2_tok3_1.json
OUT_PARENT=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_v6_a
GPUS=(4 5 6 7)
N_SHARDS=4

# ---- Detect latest v6_a checkpoint ----
LATEST_CKPT=$(ls -d $OUT_PARENT/checkpoint-* 2>/dev/null | grep -v "_merged" | sort -V | tail -1)
if [ -z "$LATEST_CKPT" ]; then
    echo "[ERROR] no checkpoint-* found in $OUT_PARENT" | tee -a "$LOG"; exit 1
fi
MERGED=${LATEST_CKPT}_merged
echo "[v6_a] LATEST_CKPT=$LATEST_CKPT" | tee -a "$LOG"
echo "[v6_a] MERGED=$MERGED" | tee -a "$LOG"

# ---- Merge LoRA + sidecar → base ----
if [ ! -f "$MERGED/config.json" ]; then
    echo "========== MERGE LoRA → BASE ==========" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="" BASE_CODE_PY=$BASE_CODE MODEL_BASE_PY=$MODEL_BASE \
    LORA_PY=$LATEST_CKPT MERGED_PY=$MERGED python3 - <<PYEOF 2>&1 | tee -a "$LOG"
import os, sys, torch, shutil
sys.path.insert(0, os.environ['BASE_CODE_PY'])
from peft import PeftModel
from safetensors.torch import load_file
from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus

BASE = os.environ['MODEL_BASE_PY']
LORA = os.environ['LORA_PY']
OUT  = os.environ['MERGED_PY']
os.makedirs(OUT, exist_ok=True)
print(f"[1/4] Loading base {BASE}")
model = video_SALMONN2_plus.from_pretrained(
    BASE, attn_implementation="eager",
    torch_dtype=torch.bfloat16, device_map="cpu",
)
print(f"[2/4] Loading PEFT adapter {LORA}")
audio_layers = model.audio.layers
del model.audio.layers
model = PeftModel.from_pretrained(model, LORA)
model.model.audio.layers = audio_layers
print("[3/4] Merging")
model = model.merge_and_unload()

qt_sidecar = os.path.join(LORA, "audio_q_tokens.safetensors")
if os.path.exists(qt_sidecar):
    qt = load_file(qt_sidecar)
    if "audio.q_tokens" in qt:
        with torch.no_grad():
            model.audio.q_tokens.data.copy_(qt["audio.q_tokens"].to(model.audio.q_tokens.dtype))
        print(f"  q_tokens loaded, norm={float(torch.linalg.norm(model.audio.q_tokens.float())):.6f}")

print(f"[4/4] Saving merged → {OUT}")
model.save_pretrained(OUT, safe_serialization=True)
for f in ["added_tokens.json","merges.txt","special_tokens_map.json",
          "tokenizer_config.json","vocab.json","preprocessor_config.json"]:
    src = os.path.join(BASE, f)
    if os.path.exists(src):
        shutil.copy(src, OUT)
print("[merge DONE]")
PYEOF
else
    echo "[v6_a] merged model already exists, skipping merge" | tee -a "$LOG"
fi

# ---- UnAV inference ----
TAG="v6a_$(basename $LATEST_CKPT)"
RESULTS_BASE=$OUT_PARENT/eval_unav_t2_$TAG
SHARD_DIR=$RESULTS_BASE/shards
mkdir -p "$RESULTS_BASE" "$SHARD_DIR"
echo "==================================================================" | tee -a "$LOG"
echo "[$TAG] $(date)" | tee -a "$LOG"
echo "[$TAG] merged: $MERGED" | tee -a "$LOG"
echo "[$TAG] test  : $TOK_TEST" | tee -a "$LOG"
echo "[$TAG] sharding..." | tee -a "$LOG"
N_SHARDS=$N_SHARDS TEST=$TOK_TEST SHARD_DIR=$SHARD_DIR python3 - <<PYEOF 2>&1 | tee -a "$LOG"
import json, os
N = int(os.environ["N_SHARDS"])
data = json.load(open(os.environ["TEST"]))
for r in range(N):
    shard = [data[i] for i in range(len(data)) if i % N == r]
    json.dump(shard, open(os.path.join(os.environ["SHARD_DIR"], f"shard{r}.json"), "w"), ensure_ascii=False)
    print(f"  shard{r}: {len(shard)}")
PYEOF

cd "$BASE_CODE"
pids=()
for r in 0 1 2 3; do
    GPU=${GPUS[$r]}
    OUT_DIR=$RESULTS_BASE/shard${r}
    PORT=$((23500 + GPU))
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

python3 "$EVAL_PY" \
    --results "$RESULTS_BASE/results_ordered.json" \
    --test_json "$RESULTS_BASE/test_ordered.json" \
    --max_time 9999.0 \
    --out_dir "$RESULTS_BASE" 2>&1 | tee -a "$LOG"
echo "[$TAG] DONE → $RESULTS_BASE" | tee -a "$LOG"

echo | tee -a "$LOG"
echo "============== UnAV (v6_a) SUMMARY ==============" | tee -a "$LOG"
cat "$RESULTS_BASE/eval_miou_summary.json" | tee -a "$LOG"
echo "[done] $(date)" | tee -a "$LOG"
