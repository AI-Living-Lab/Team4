#!/bin/bash
# ============================================================
# v8: SFT-as-format-teacher  (IF 보존 + time token 형식만 학습)
#   패러다임 전환 (v7 검증 결과 반영):
#     - v7는 PU-VALOR로 grounding + format 동시 학습 → IF 0% (1500 step만에 사라짐)
#     - v8는 grounding은 GDPO에 위임, SFT는 time-token 형식 + multimodal alignment 만
#
#   v7 대비 핵심 변경:
#     1) 데이터: PU-VALOR 91k → UnAV-100 v2.json 10k
#        - RL(=GDPO) 단계와 분포 일치
#        - "format hint 있고 wrapper 없는" prompt (시간토큰 형식만 학습)
#     2) 시작 모델: 원본 base (v7 ckpt 사용 X, IF 살아있는 깨끗한 시작점)
#     3) LoRA capacity 대폭 축소:
#        - r:        128 → 16
#        - alpha:    128 → 16  (scaling 1.0 유지)
#        - dropout:  0.05 → 0.1
#        → q/k/v/o attention 분포 변형 최소 → IF 보존
#     4) LR: 모두 1e-4 유지 (v7와 동일).
#        - LoRA: r=16 capacity 8x ↓ 가 이미 drift 8배 줄임 → LR 추가로 낮출 필요 없음
#          오히려 학습 step 1295 (v7의 1/9)라 LR 유지해야 형식 학습에 충분
#        - embed/lm_head: row-mask 가 wrapper row 강제 freeze → time-token row만 안전 학습
#     4-bis) Best ckpt 선택: 학습 중 자동 X (eval_loss 가 IF 보존을 반영 못 함).
#        save_total_limit=10 으로 모든 ckpt 저장, 사후 50-sample IF test 로 best ckpt 선택.
#     5) TTI: special_token → off
#        - base 모델은 TTI 없이 학습됨, 분포 일치 우선
#     6) 학습 길이: epoch=1 (~1295 step), eval/save 매 250 step
#        - 250-step 마다 IF retention 정량 측정 (50-sample wrapper rate)
#        - wrapper rate < 50% 떨어지면 조기 종료 후보
#     7) multimodal modules (visual.merger, audio.qformer/q_tokens/proj):
#        v7와 동일 LR, trainable 유지 — LLM IF 영향 없으면서 audio/video→LLM 정렬 학습
#
#   GPU: 0,1,2,3 (4 GPUs, grad_accum=2, effective batch=8)
# ============================================================
set -eo pipefail
source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

export CUDA_VISIBLE_DEVICES=0,1,2,3
export ARNOLD_WORKER_GPU=4
export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/sft/video_SALMONN2_plus"

MODEL=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
MODEL_BASE=$MODEL
DATASET=/home/aix23102/audiolm/Team404/data/train/unav100_v2.json
EVAL_DATASET=/home/aix23102/audiolm/Team404/data/test/unav100_v2_500.json
OUTPUT_DIR=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_v8
RUN_NAME=salmonn2plus_v8
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
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_pixels 176400 --min_pixels 784 \
    --video_max_frame_pixels 28224 --video_min_frame_pixels 784 \
    --eval_strategy "steps" --eval_steps 250 \
    --save_strategy "steps" --save_steps 250 \
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
    --run_name "$RUN_NAME" --report_to wandb \
    --video_min_frames 64 --video_max_frames 256 --base_interval 0.2 \
    --train_type sft --no_audio False \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"

echo "[v8] training done."
