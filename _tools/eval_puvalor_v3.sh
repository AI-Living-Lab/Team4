#!/bin/bash
# ============================================================
# salmonn2plus_puvalor_v3 평가 (PU-VALOR test only)
#   - GPU 4,5,6 으로 3-shard 병렬 inference
#   - PU-VALOR GOUT (puvalor_test_gout.json, 2528) 만 평가
#   - eval_miou_subgroup.py 로 overall/single/multi mIoU 리포트
#   - 학습과 동일한 setting: video_max_frames=512, base_interval=0.2
# ============================================================
set -eo pipefail

source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

export ARNOLD_WORKER_NUM=1 ARNOLD_ID=0 METIS_WORKER_0_HOST=localhost

BASE_CODE=/home/aix23102/audiolm/vS2_eunji/video_SALMONN2_plus
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
TEST_JSON=/home/aix23102/audiolm/vS2_eunji/data/puvalor_test_gout.json
EVAL_SCRIPT=/home/aix23102/audiolm/vS2_eunji/eval/eval_miou_subgroup.py

# CKPT은 인자로 받음 (예: checkpoint-15162). MERGED 모델 디렉터리도 인자로.
CKPT=${1:?Usage: $0 <ckpt_dir> [merged_model_dir]}
MERGED_MODEL=${2:-${CKPT}_merged}

RESULTS_BASE=/home/aix23102/audiolm/vS2_eunji/eval/results/puvalor_v3_$(basename $CKPT)
SHARD_DIR=$RESULTS_BASE/shards
mkdir -p "$RESULTS_BASE" "$SHARD_DIR"

# ---- (1) Merge LoRA → base if MERGED_MODEL doesn't exist ----
if [ ! -f "$MERGED_MODEL/config.json" ]; then
    echo "========== MERGE LoRA → BASE =========="
    python3 - <<PYEOF
import os, sys, torch, shutil
sys.path.insert(0, '$BASE_CODE')
from peft import PeftModel
from safetensors.torch import load_file
from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus

BASE = "$MODEL_BASE"
LORA = "$CKPT"
OUT  = "$MERGED_MODEL"

os.makedirs(OUT, exist_ok=True)
print(f"[1/4] Loading base from {BASE}")
model = video_SALMONN2_plus.from_pretrained(
    BASE, attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16, device_map="cpu",
)
print(f"[2/4] Loading PEFT adapter from {LORA}")
audio_layers = model.audio.layers
del model.audio.layers
model = PeftModel.from_pretrained(model, LORA)
model.model.audio.layers = audio_layers

print("[3/4] Merging LoRA + modules_to_save")
model = model.merge_and_unload()

adapter_path = os.path.join(LORA, "adapter_model.safetensors")
if os.path.exists(adapter_path):
    aw = load_file(adapter_path)
    non_lora = {k.replace("base_model.model.", "", 1): v
                for k, v in aw.items() if "lora" not in k and "modules_to_save" not in k}
    if non_lora:
        missing, unexpected = model.load_state_dict(non_lora, strict=False)
        print(f"[INFO] Loaded {len(non_lora)} non-LoRA (missing={len(missing)}, unexpected={len(unexpected)})")

print(f"[4/4] Saving merged → {OUT}")
model.save_pretrained(OUT, safe_serialization=True)
for f in ["added_tokens.json","merges.txt","special_tokens_map.json",
          "tokenizer_config.json","vocab.json","preprocessor_config.json"]:
    src = os.path.join(BASE, f)
    if os.path.exists(src):
        shutil.copy(src, OUT)
print("[DONE]")
PYEOF
fi

# ---- (2) Build 3 shards from puvalor_test_gout.json ----
python3 - <<PYEOF
import json, os
data = json.load(open("$TEST_JSON"))
N = 3
for r in range(N):
    shard = [data[i] for i in range(len(data)) if i % N == r]
    out = os.path.join("$SHARD_DIR", f"shard{r}.json")
    with open(out, "w") as f:
        json.dump(shard, f, ensure_ascii=False)
    print(f"  shard{r}: {len(shard)} -> {out}")
PYEOF

# ---- (3) Inference on 3 shards (GPU 4,5,6) ----
GPUS=(4 5 6)
cd "$BASE_CODE"

run_shard() {
    local RANK=$1
    local GPU=${GPUS[$RANK]}
    local TEST_JSON_S=$SHARD_DIR/shard${RANK}.json
    local OUT_DIR=$RESULTS_BASE/shard${RANK}
    local PORT=$((16000 + GPU))
    mkdir -p "$OUT_DIR"
    echo "[GPU $GPU / RANK $RANK] inference"

    CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$PORT \
        qwenvl/train/train_qwen.py \
        --model_base "$MODEL_BASE" --run_test True --pred_rank $RANK \
        --deepspeed scripts/zero2.json \
        --model_name_or_path "$MERGED_MODEL" \
        --dataset_use "$TEST_JSON_S" --bf16 --output_dir "$OUT_DIR" \
        --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
        --max_pixels 176400 --min_pixels 784 \
        --video_max_frame_pixels 28224 --video_min_frame_pixels 784 \
        --eval_strategy "no" --save_strategy "no" --learning_rate 1e-5 \
        --model_max_length 100000 --gradient_checkpointing True --dataloader_num_workers 2 \
        --run_name "shard${RANK}" --report_to none \
        --video_min_frames 64 --video_max_frames 512 --base_interval 0.2 \
        --lora_ckpt "No" --no_audio False 2>&1 | tee "$OUT_DIR/inference.log"
}

echo "========== INFERENCE (3 shards parallel) =========="
for r in 0 1 2; do run_shard $r & done
wait

# ---- (4) Reassemble + evaluate ----
echo "========== MERGE & EVAL =========="
python3 - <<PYEOF
import json, os
N = 3
full = json.load(open("$TEST_JSON"))
n = len(full)
shards = []
for r in range(N):
    p = os.path.join("$RESULTS_BASE", f"shard{r}", f"shard{r}", f"test_results_rank{r}.json")
    if not os.path.exists(p):
        print(f"MISSING: {p}"); continue
    shards.append(json.load(open(p)))
out = [None]*n
for i in range(n):
    r = i % N; pos = i // N
    if pos < len(shards[r]):
        out[i] = shards[r][pos]
out = [x for x in out if x is not None]
print(f"Reassembled: {len(out)}")
mp = os.path.join("$RESULTS_BASE", "merged_results.json")
json.dump(out, open(mp,"w"), ensure_ascii=False)
print(f"Saved: {mp}")
PYEOF

python3 "$EVAL_SCRIPT" \
    --results "$RESULTS_BASE/merged_results.json" \
    --test_json "$TEST_JSON" \
    --max_time 9999.0 \
    --out_dir "$RESULTS_BASE"

echo "========== DONE =========="
echo "Results: $RESULTS_BASE"
