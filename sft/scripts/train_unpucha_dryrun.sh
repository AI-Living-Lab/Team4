#!/bin/bash
# ============================================================
# v8 SFT — unpucha_sft (balanced 3k, full time-token range)
#   train_v8.sh 와 데이터만 교체, 나머지 하이퍼파라미터 전부 동일.
#   목적: unav100(GT<=60s) 단일학습으로 백/십자리 time-token 이 미학습 →
#         ActivityNet 예측이 ~50s 에 갇힘. 3k 균형셋(charades1k/puvalor1k/
#         U-multi500/U-single500, GT end 최대 214s)으로 전 범위 지도.
#   변경점 (train_v8.sh 대비):
#     - DATASET = unpucha_sft.json (3,000)
#     - GPU 4→2 (0,1), per_device_bs 1, grad_accum 2→4  => eff batch 8 유지
#     - 경로 전부 로컬(/home/team404/workspace)로 리맵
#   1 epoch = eff batch 8 기준 375 step. eval/save 250 step.
# ============================================================
set -eo pipefail
source /home/team404/miniconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

export CUDA_VISIBLE_DEVICES=0
export ARNOLD_WORKER_GPU=1
export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/sft/video_SALMONN2_plus"

MODEL=/home/team404/workspace/checkpoints/base/video_salmonn2_plus_7B_time_tokens
MODEL_BASE=$MODEL
DATASET=/home/team404/workspace/data/train/unpucha_sft.json
EVAL_DATASET=/home/team404/workspace/data/test/unav100_v2_500.json
OUTPUT_DIR=/home/team404/workspace/checkpoints/sft/salmonn2plus_v8_unpucha_dryrun
RUN_NAME=salmonn2plus_v8_unpucha_dryrun
mkdir -p "$OUTPUT_DIR"

torchrun --standalone --nproc_per_node=$ARNOLD_WORKER_GPU \
    qwenvl/train/train_qwen.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "$MODEL" --model_base "$MODEL_BASE" \
    --dataset_use "$DATASET" --eval_dataset_use "$EVAL_DATASET" \
    --tune_mm_vision False --tune_mm_mlp True --tune_mm_llm False \
    --tune_mm_audio False --tune_mm_qformer True \
    --use_lora True --lora_r 16 --lora_alpha 16 --lora_dropout 0.1 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --bf16 --output_dir "$OUTPUT_DIR" \
    --max_steps 8 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_pixels 176400 --min_pixels 784 \
    --video_max_frame_pixels 28224 --video_min_frame_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 10 \
    --learning_rate 1e-4 --weight_decay 0 \
    --lora_lr 1e-4            --lora_wd 0.0 \
    --embed_lr 1e-4           --embed_wd 0.0 \
    --lm_head_lr 1e-4         --lm_head_wd 0.0 \
    --visual_merger_lr 1e-5   --visual_merger_wd 0.01 \
    --audio_qformer_lr 2e-5   --audio_qformer_wd 0.01 \
    --audio_proj_lr 2e-5      --audio_proj_wd 0.01 \
    --audio_q_tokens_lr 5e-5  --audio_q_tokens_wd 0.0 \
    --ordinal_enabled True \
    --time_ndig_int 3 --time_ndig_dec 1 \
    --lambda_ord 0.1 \
    --ordinal_unav_weight 1.0 \
    --tti_time_format off \
    --warmup_ratio 0.03 --max_grad_norm 1 \
    --lr_scheduler_type "cosine" --logging_steps 1 \
    --model_max_length 32768 --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name "$RUN_NAME" --report_to none \
    --video_min_frames 64 --video_max_frames 256 --base_interval 0.2 \
    --train_type sft --no_audio False \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"

echo "[v8-unpucha-sft] training done."
