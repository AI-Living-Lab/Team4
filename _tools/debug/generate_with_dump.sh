#!/bin/bash
# ============================================================
# TTI debug interleave + model.generate — 인터리빙 덤프와 실제 생성 출력을
# 한 번에 얻기 위한 런처. (nproc=1, rank=0 강제)
#
# 일반적인 debug 는 _tools/debug/sweep_dump.sh (모델 로드 없음, 빠름) 로 충분.
# 이 스크립트는 덤프 결과와 모델 출력을 함께 비교하고 싶을 때 사용.
#
# 사용법:
#   bash _tools/debug/generate_with_dump.sh \
#     [BASE_MODEL=...] \
#     [LORA_CKPT=.../checkpoint-1000 | No] \
#     [DATASET=data/debug_interleave_samples.json] \
#     [OUT_BASE=_debug_out/generate] \
#     [TAG=interval0.2_maxf128] \
#     [BASE_INTERVAL=0.2] [VIDEO_MAX_FRAMES=128] [VIDEO_MIN_FRAMES=64] \
#     [MAX_PIXELS=176400] [MIN_PIXELS=784] \
#     [SAMPLE_LIMIT=-1] [GPU=0]
#
# 결과:
#   <OUT_BASE>/<TAG>/                         — dump JSON/TXT 파일
#   <OUT_BASE>/<TAG>/./test_results_rank0.json — generate 결과 (실제 출력)
#   <OUT_BASE>/<TAG>/inference.log
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_CODE="$BASE_DIR/video_SALMONN2_plus"

source /workspace/setup.sh
conda activate salmonn2plus

export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost
export ARNOLD_WORKER_GPU=1

BASE_MODEL="${BASE_MODEL:-/workspace/checkpoints/base/video_salmonn2_plus_7B_time_tokens}"
LORA_CKPT="${LORA_CKPT:-No}"
DATASET="${DATASET:-data/debug_interleave_samples.json}"
OUT_BASE="${OUT_BASE:-_debug_out/generate}"
BASE_INTERVAL="${BASE_INTERVAL:-0.2}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-128}"
VIDEO_MIN_FRAMES="${VIDEO_MIN_FRAMES:-64}"
MAX_PIXELS="${MAX_PIXELS:-176400}"
MIN_PIXELS="${MIN_PIXELS:-784}"
VIDEO_MAX_FRAME_PIXELS="${VIDEO_MAX_FRAME_PIXELS:-28224}"
VIDEO_MIN_FRAME_PIXELS="${VIDEO_MIN_FRAME_PIXELS:-784}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:--1}"
GPU="${GPU:-0}"
TAG="${TAG:-interval${BASE_INTERVAL}_maxf${VIDEO_MAX_FRAMES}}"
OUT_DIR="$BASE_DIR/$OUT_BASE/$TAG"

# 절대경로 처리
case "$DATASET" in
    /*) ABS_DATASET="$DATASET" ;;
    *)  ABS_DATASET="$BASE_DIR/$DATASET" ;;
esac

mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES=$GPU
MASTER_PORT=$((12900 + GPU))

echo "=================================================="
echo "  BASE_MODEL     : $BASE_MODEL"
echo "  LORA_CKPT      : $LORA_CKPT"
echo "  DATASET        : $ABS_DATASET"
echo "  OUT_DIR        : $OUT_DIR"
echo "  TAG            : $TAG"
echo "  BASE_INTERVAL  : $BASE_INTERVAL"
echo "  VIDEO_FRAMES   : [$VIDEO_MIN_FRAMES, $VIDEO_MAX_FRAMES]"
echo "  PIXELS         : [$MIN_PIXELS, $MAX_PIXELS]"
echo "  SAMPLE_LIMIT   : $SAMPLE_LIMIT"
echo "  GPU            : $GPU"
echo "=================================================="

cd "$BASE_CODE"

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
    qwenvl/train/train_qwen.py \
    --model_base "$BASE_MODEL" \
    --run_test True \
    --pred_rank 0 \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "$BASE_MODEL" \
    --dataset_use "$ABS_DATASET" \
    --bf16 \
    --output_dir "$OUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_pixels "$MAX_PIXELS" \
    --min_pixels "$MIN_PIXELS" \
    --video_max_frame_pixels "$VIDEO_MAX_FRAME_PIXELS" \
    --video_min_frame_pixels "$VIDEO_MIN_FRAME_PIXELS" \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-5 \
    --model_max_length 10000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --run_name "." \
    --report_to none \
    --video_min_frames "$VIDEO_MIN_FRAMES" \
    --video_max_frames "$VIDEO_MAX_FRAMES" \
    --base_interval "$BASE_INTERVAL" \
    --lora_ckpt "$LORA_CKPT" \
    --no_audio False \
    --debug_interleave_dir "$OUT_DIR" \
    --debug_interleave_generate True \
    --debug_interleave_sample_limit "$SAMPLE_LIMIT" \
    2>&1 | tee "$OUT_DIR/inference.log"

echo ""
echo "[완료] $OUT_DIR"
ls "$OUT_DIR"/*.json 2>/dev/null | head -20
