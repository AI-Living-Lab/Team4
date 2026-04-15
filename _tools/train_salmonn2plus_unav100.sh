#!/bin/bash
# ============================================================
# SALMONN2+ UnAV-100 multi-segment fine-tuning
#   - Data: UnAV-100 multi-segment QA
#   - LoRA: r=128, alpha=256 (from scratch)
#   - GPU 1
# ============================================================

source /workspace/setup.sh
conda activate salmonn2plus

export CUDA_VISIBLE_DEVICES=0
export ARNOLD_WORKER_GPU=1
export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

cd "${BASE_DIR}/video_SALMONN2_plus"

MODEL=${CHECKPOINTS_DIR}/video_salmonn2_plus_7B_time_tokens
MODEL_BASE=${CHECKPOINTS_DIR}/video_salmonn2_plus_7B_time_tokens
DATASET=${BASE_DIR}/data/unav100_train_multiseg_salmonn2plus.json
OUTPUT_DIR=${CHECKPOINTS_DIR}/salmonn2plus_unav100_multiseg
RUN_NAME=salmonn2plus_unav100

# UnAV-100: 10358 samples, effective_batch=2 (bs=1, accum=2, gpu=1)
# steps_per_epoch = 10358 / 2 = 5179
# 1 epoch, save every 500 steps

mkdir -p "$OUTPUT_DIR"

torchrun --standalone --nproc_per_node=$ARNOLD_WORKER_GPU \
    qwenvl/train/train_qwen.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "$MODEL" \
    --model_base "$MODEL_BASE" \
    --dataset_use "$DATASET" \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm False \
    --tune_mm_audio False \
    --tune_mm_qformer True \
    --use_lora True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --bf16 \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_pixels 176400 \
    --min_pixels 784 \
    --video_max_frame_pixels 28224 \
    --video_min_frame_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate 5e-5 \
    --weight_decay 0 \
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
    --video_max_frames 128 \
    --base_interval 0.2 \
    --train_type sft \
    --no_audio False \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"
