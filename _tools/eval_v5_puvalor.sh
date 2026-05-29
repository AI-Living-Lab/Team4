#!/bin/bash
# ============================================================
# v5 학습 후 PU-VALOR test mIoU 평가 (T1+T2 만)
#   Usage:  eval_v5_puvalor.sh <OUTPUT_DIR> <TOK> <NAME>
#     OUTPUT_DIR: e.g. /data0/.../salmonn2plus_v5_a
#     TOK:        "4_1" or "3_2"  (training 의 tokenization 과 동일)
#     NAME:       "v5_a" / "v5_c" / "v5_b"  (run name)
#   ENV (optional):
#     EVAL_GPUS:  comma-separated GPU ids (default "4,5,6,7"). N shards = len(EVAL_GPUS).
#
# 흐름:
#   1) latest checkpoint 자동 탐지
#   2) LoRA + sidecar (q_tokens) merge → ${LATEST}_merged
#   3) PU-VALOR test T1+T2 (3,172 samples) 를 N-shard 로 분할 → 각 GPU 병렬 inference
#   4) shard 결과 재조립 → eval_miou_v5.py 로 mIoU/R@k 계산 (max_time=9999s, PU-VALOR max 538s)
# ============================================================
set -eo pipefail
source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

export ARNOLD_WORKER_NUM=1 ARNOLD_ID=0 METIS_WORKER_0_HOST=localhost

OUTPUT_DIR=${1:?Usage: $0 <OUTPUT_DIR> <TOK> <NAME>}
TOK=${2:?need tokenization (4_1 or 3_2)}
NAME=${3:?need run name (v5_a/c/b)}

# Parse GPU list (default: 4,5,6,7). Shard count = number of GPUs.
EVAL_GPUS_STR="${EVAL_GPUS:-4,5,6,7}"
IFS=',' read -ra GPUS <<< "$EVAL_GPUS_STR"
N_SHARDS=${#GPUS[@]}
echo "[eval_v5] GPUs=${EVAL_GPUS_STR} (N_SHARDS=$N_SHARDS)"

BASE_CODE=/home/aix23102/audiolm/vS2_eunji/video_SALMONN2_plus
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
DATA_DIR=/home/aix23102/audiolm/vS2_eunji/data
EVAL_PY=/home/aix23102/audiolm/vS2_eunji/eval/eval_miou_v5.py
TEST_JSON=$DATA_DIR/puvalor_test_v5_t1t2_tok${TOK}.json
TEST_JSON_RAW=$DATA_DIR/puvalor_test_v5_t1t2.json   # gt_segments 가 필요한 placeholder 버전

LATEST_CKPT=$(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -z "$LATEST_CKPT" ]; then
    echo "[ERROR] no checkpoint-* found in $OUTPUT_DIR"; exit 1
fi
MERGED_MODEL=${LATEST_CKPT}_merged
RESULTS_BASE=$OUTPUT_DIR/eval_puvalor_test_t1t2_$(basename $LATEST_CKPT)
SHARD_DIR=$RESULTS_BASE/shards
mkdir -p "$RESULTS_BASE" "$SHARD_DIR"

echo "[eval_v5] OUTPUT_DIR=$OUTPUT_DIR"
echo "[eval_v5] LATEST_CKPT=$LATEST_CKPT"
echo "[eval_v5] TOK=$TOK, TEST_JSON=$TEST_JSON"
echo "[eval_v5] RESULTS_BASE=$RESULTS_BASE"

# ---- (1) Merge LoRA → base ----
if [ ! -f "$MERGED_MODEL/config.json" ]; then
    echo "========== MERGE LoRA → BASE =========="
    CUDA_VISIBLE_DEVICES="" python3 - <<PYEOF
import os, sys, torch, shutil
sys.path.insert(0, '$BASE_CODE')
from peft import PeftModel
from safetensors.torch import load_file
from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus

BASE = "$MODEL_BASE"
LORA = "$LATEST_CKPT"
OUT  = "$MERGED_MODEL"
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

# audio.q_tokens sidecar
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
    echo "[eval_v5] merged model already exists, skipping merge"
fi

# ---- (2) N-shard 분할 (rank-stride) ----
echo "========== SHARD TEST JSON (N=$N_SHARDS) =========="
N_SHARDS=$N_SHARDS python3 - <<PYEOF
import json, os
N = int(os.environ["N_SHARDS"])
TEST = "$TEST_JSON"
DIR = "$SHARD_DIR"
data = json.load(open(TEST))
for r in range(N):
    shard = [data[i] for i in range(len(data)) if i % N == r]
    out = os.path.join(DIR, f"shard{r}.json")
    json.dump(shard, open(out, "w"), ensure_ascii=False)
    print(f"  shard{r}: {len(shard)} -> {out}")
print(f"total: {len(data)}")
PYEOF

# ---- (3) Parallel inference ----
echo "========== INFERENCE ($N_SHARDS shards parallel on GPU $EVAL_GPUS_STR) =========="
cd "$BASE_CODE"

run_shard() {
    local RANK=$1
    local GPU=${GPUS[$RANK]}
    local SHARD_JSON=$SHARD_DIR/shard${RANK}.json
    local OUT_DIR=$RESULTS_BASE/shard${RANK}
    local PORT=$((17500 + GPU))
    mkdir -p "$OUT_DIR"
    echo "[GPU $GPU / RANK $RANK] inference"

    CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$PORT \
        qwenvl/train/train_qwen.py \
        --model_base "$MODEL_BASE" --run_test True --pred_rank $RANK \
        --deepspeed scripts/zero2.json \
        --model_name_or_path "$MERGED_MODEL" \
        --dataset_use "$SHARD_JSON" --bf16 --output_dir "$OUT_DIR" \
        --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
        --max_pixels 176400 --min_pixels 784 \
        --video_max_frame_pixels 28224 --video_min_frame_pixels 784 \
        --eval_strategy "no" --save_strategy "no" --learning_rate 1e-5 \
        --model_max_length 100000 --gradient_checkpointing True --dataloader_num_workers 2 \
        --run_name "shard${RANK}" --report_to none \
        --video_min_frames 64 --video_max_frames 128 --base_interval 0.2 \
        --lora_ckpt "No" --no_audio False 2>&1 | tee "$OUT_DIR/inference.log"
}

for r in $(seq 0 $((N_SHARDS - 1))); do run_shard $r & done
wait

# ---- (4) Reassemble + eval ----
echo "========== MERGE & EVAL =========="
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
n_ok = sum(1 for x in out if x is not None)
print(f"Reassembled: {n_ok}/{n}")
# null 제거 + 같은 순서의 test_data subset 생성
keep_idx = [i for i, x in enumerate(out) if x is not None]
ordered_results = [out[i] for i in keep_idx]
ordered_test = [full[i] for i in keep_idx]
json.dump(ordered_results, open(os.path.join("$RESULTS_BASE", "results_ordered.json"), "w"), ensure_ascii=False)
json.dump(ordered_test, open(os.path.join("$RESULTS_BASE", "test_ordered.json"), "w"), ensure_ascii=False)
print(f"Saved ordered pair: {len(ordered_results)}")
PYEOF

# PU-VALOR 비디오는 더 길음 (max 538s) — max_time 충분히 크게
python3 "$EVAL_PY" \
    --results "$RESULTS_BASE/results_ordered.json" \
    --test_json "$RESULTS_BASE/test_ordered.json" \
    --max_time 9999.0 \
    --out_dir "$RESULTS_BASE"

echo "========== DONE =========="
echo "Results dir: $RESULTS_BASE"
echo "Summary: $RESULTS_BASE/eval_miou_summary.json"
