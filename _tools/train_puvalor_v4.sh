#!/bin/bash
# ============================================================
# SALMONN2+ PU-VALOR v4 학습
#   - GPU: 4,5,6 (3 GPUs)
#   - Train: data/puvalor_train.json (90,972)
#   - Val:   data/puvalor_val_sub500.json (500)
#   - LoRA target: q,k,v,o_proj  (gate/up/down 제외)
#   - modules_to_save (자동):
#       embed_tokens + lm_head      (time tokens; gradient-masked to time rows only)
#       visual.merger
#       audio.qformer + audio.audio_proj
#       audio.q_tokens (manually unfrozen since PEFT can't wrap nn.Parameter)
#   - video_max_frames=512, base_interval=0.2
#   - effective_batch = 1 × 2 × 3 = 6
#
# Per-module LR / weight_decay:
#   LoRA           1e-4 / 0.0
#   embed_tokens   8e-4 / 0.0   (new token rows only via gradient mask)
#   lm_head        8e-4 / 0.0   (new token rows only via gradient mask)
#   visual.merger  1e-5 / 0.01
#   audio.qformer  2e-5 / 0.01
#   audio.audio_proj 2e-5 / 0.01
#   audio.q_tokens 5e-5 / 0.0
# ============================================================
set -eo pipefail

source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

export CUDA_VISIBLE_DEVICES=4,5,6
export ARNOLD_WORKER_GPU=3
export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

cd /home/aix23102/audiolm/vS2_eunji/video_SALMONN2_plus

MODEL=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
DATASET=/home/aix23102/audiolm/vS2_eunji/data/puvalor_train.json
EVAL_DATASET=/home/aix23102/audiolm/vS2_eunji/data/puvalor_val_sub500.json
OUTPUT_DIR=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_v4
RUN_NAME=salmonn2plus_puvalor_v4

mkdir -p "$OUTPUT_DIR"

torchrun --standalone --nproc_per_node=$ARNOLD_WORKER_GPU \
    qwenvl/train/train_qwen.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "$MODEL" \
    --model_base "$MODEL_BASE" \
    --dataset_use "$DATASET" \
    --eval_dataset_use "$EVAL_DATASET" \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm False \
    --tune_mm_audio False \
    --tune_mm_qformer True \
    --use_lora True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --bf16 \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_pixels 176400 \
    --min_pixels 784 \
    --video_max_frame_pixels 28224 \
    --video_min_frame_pixels 784 \
    --eval_strategy "steps" \
    --eval_steps 1516 \
    --save_strategy "steps" \
    --save_steps 1516 \
    --save_total_limit 6 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    --early_stopping_patience 2 \
    --learning_rate 1e-4 \
    --weight_decay 0 \
    --lora_lr 1e-4            --lora_wd 0.0 \
    --embed_lr 8e-4           --embed_wd 0.0 \
    --lm_head_lr 8e-4         --lm_head_wd 0.0 \
    --visual_merger_lr 1e-5   --visual_merger_wd 0.01 \
    --audio_qformer_lr 2e-5   --audio_qformer_wd 0.01 \
    --audio_proj_lr 2e-5      --audio_proj_wd 0.01 \
    --audio_q_tokens_lr 5e-5  --audio_q_tokens_wd 0.0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 100000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name "$RUN_NAME" \
    --report_to wandb \
    --video_min_frames 64 \
    --video_max_frames 512 \
    --base_interval 0.2 \
    --train_type sft \
    --no_audio False \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"
