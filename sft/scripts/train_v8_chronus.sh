#!/bin/bash
# ============================================================
# v8_chronus: natural(chronus) 포맷 250-step SFT  (natural RL ablation 용 warm-start)
#   train_v8.sh 의 chronus 변형 — RL 이 second{start}-second{end} 포맷을 내도록 SFT 로 먼저 데움.
#
#   ★ train_v8.sh(special_token SFT) 대비 바뀐 점 (그 외 동일):
#     1) DATASET   : unav100_v2.json → unav100_chronus.json  (출력 chronus, 멀티힌트 포함)
#     2) 백본: video_salmonn2_plus_7B_full (순수 백본, 타임토큰 미추가).
#        ⚠️ _time_tokens 는 타임토큰 추가된 별개 가중치 → chronus 엔 부적합(잘못된 백본).
#     3) embed/lm_head 전체 vocab row 학습: --tune_embed_lm_head True (row-mask 없음).
#        chronus 는 평문토큰(second/{/}/digit)을 쓰고 _full 엔 타임토큰이 없으므로 전체 학습.
#     4) ordinal loss OFF: --ordinal_enabled False  (타임토큰 없음 → 무의미)
#     4) GPU 4→1: nproc=1, grad_accum 2→8 로 eff batch 8 유지 (원본 4×2 = 1×8)
#     5) max_steps 250 (250-step 지점 = 원본 special_token SFT 와 동일 샘플수 2000)
#     6) eval OFF (chronus eval 셋 불요; ckpt 사후선택). tti_time_format off 는 동일.
#   OUTPUT: salmonn2plus_v8_chronus  → natural RL 의 model_base 로 사용.
#   GPU: 1장 (기본 3번; 0,1 은 타 학습 점유중이라 회피).
# ============================================================
set -eo pipefail
source /home/team404/miniconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

export CUDA_VISIBLE_DEVICES=3          # ⚠️ 빈 GPU 로 조정 (0,1 점유중이면 2 또는 3)
export ARNOLD_WORKER_GPU=1
export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/sft/video_SALMONN2_plus"

MODEL=/home/team404/workspace/checkpoints/base/video_salmonn2_plus_7B_full
MODEL_BASE=$MODEL
DATASET=/home/team404/workspace/data/train/unav100_chronus.json
OUTPUT_DIR=/home/team404/workspace/checkpoints/sft/salmonn2plus_v8_chronus
RUN_NAME=salmonn2plus_v8_chronus
mkdir -p "$OUTPUT_DIR"

torchrun --standalone --nproc_per_node=$ARNOLD_WORKER_GPU \
    qwenvl/train/train_qwen.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "$MODEL" --model_base "$MODEL_BASE" \
    --dataset_use "$DATASET" \
    --tune_mm_vision False --tune_mm_mlp True --tune_mm_llm False \
    --tune_mm_audio False --tune_mm_qformer True \
    --use_lora True --lora_r 16 --lora_alpha 16 --lora_dropout 0.1 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --tune_embed_lm_head True \
    --bf16 --output_dir "$OUTPUT_DIR" \
    --max_steps 250 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_pixels 176400 --min_pixels 784 \
    --video_max_frame_pixels 28224 --video_min_frame_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" --save_steps 250 \
    --save_total_limit 2 \
    --learning_rate 1e-4 --weight_decay 0 \
    --lora_lr 1e-4            --lora_wd 0.0 \
    --embed_lr 1e-4           --embed_wd 0.0 \
    --lm_head_lr 1e-4         --lm_head_wd 0.0 \
    --visual_merger_lr 1e-5   --visual_merger_wd 0.01 \
    --audio_qformer_lr 2e-5   --audio_qformer_wd 0.01 \
    --audio_proj_lr 2e-5      --audio_proj_wd 0.01 \
    --audio_q_tokens_lr 5e-5  --audio_q_tokens_wd 0.0 \
    --ordinal_enabled False \
    --tti_time_format off \
    --warmup_ratio 0.03 --max_grad_norm 1 \
    --lr_scheduler_type "cosine" --logging_steps 1 \
    --model_max_length 32768 --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name "$RUN_NAME" --report_to wandb \
    --video_min_frames 64 --video_max_frames 256 --base_interval 0.2 \
    --train_type sft --no_audio False \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"

echo "[v8_chronus] SFT 250-step done → $OUTPUT_DIR"
