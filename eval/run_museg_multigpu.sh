#!/usr/bin/env bash
# MUSEG UnAV-100 멀티세그 추론 — 6-GPU(2~7) 분산 러너.
# 각 shard 가 서로소 청크를 맡고 results/chunk_XXXX.json 을 개별 저장(resume 안전).
# 모델 로드가 디스크(98% full) 경합이라 shard 간 stagger.
WS=/home/aix23102/audiolm/workspace
SRC=$WS/github/MUSEG/src/infer_unav_chunks.py
CKPT=$WS/checkpoints/base/MUSEG-7B
IND=$WS/outputs/base/MUSEG/unav100_multiseg/inputs
RESD=$WS/outputs/base/MUSEG/unav100_multiseg/results
LOGD=$WS/outputs/base/MUSEG/unav100_multiseg
NUM_SHARDS=6
GPU_BASE=2   # GPU 2..7

source ~/anaconda3/etc/profile.d/conda.sh && conda activate museg
mkdir -p "$RESD"

for sid in $(seq 0 $((NUM_SHARDS-1))); do
  gpu=$((GPU_BASE + sid))
  log=$LOGD/run_shard${sid}_gpu${gpu}.log
  echo "launch shard $sid on GPU $gpu → $log"
  CUDA_VISIBLE_DEVICES=$gpu nohup python "$SRC" \
    --ckpt_path "$CKPT" --inputs_dir "$IND" --results_dir "$RESD" \
    --batch_num 8 --max_model_len 32768 \
    --num_shards $NUM_SHARDS --shard_id $sid \
    > "$log" 2>&1 &
  disown
  sleep 25   # 모델 로드 디스크 경합 완화
done
echo "all $NUM_SHARDS shards launched."
