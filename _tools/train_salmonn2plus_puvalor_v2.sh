#!/bin/bash
# ============================================================
# SALMONN2+ PU-VALOR 학습 v2 (new splits + val + early stopping)
#   - GPU: 4,5,6 (3 GPUs)
#   - Train: /data0/aix23102/PU-VALOR/splits/train.json (90,972)
#   - Val:   /data0/aix23102/PU-VALOR/splits/val_sub500.json (500)
#   - 1 epoch 최대 (effective batch 6 → 15,162 steps)
#   - save + eval every 0.1 ep (1,516 steps) → 최대 10회
#   - early_stopping_patience=2 → overfit 감지 시 자동 중단
#   - load_best_model_at_end → 최종 모델 = best eval_loss ckpt
# ============================================================

source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

export CUDA_VISIBLE_DEVICES=4,5,6
export ARNOLD_WORKER_GPU=3
export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

cd /home/aix23102/audiolm/video-SALMONN-2/video_SALMONN2_plus

MODEL=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
DATASET=/data0/aix23102/PU-VALOR/splits/train.json
EVAL_DATASET=/data0/aix23102/PU-VALOR/splits/val_sub500.json
OUTPUT_DIR=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_v2_1ep
RUN_NAME=salmonn2plus_puvalor_v2_1ep_earlystop

mkdir -p "$OUTPUT_DIR"

torchrun --standalone --nproc_per_node=$ARNOLD_WORKER_GPU \
    qwenvl/train/train_qwen.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "$MODEL" \
    --model_base "$MODEL_BASE" \
    --dataset_use "$DATASET" \
    --eval_dataset_use "$EVAL_DATASET" \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --tune_mm_audio False \
    --tune_mm_qformer False \
    --use_lora True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
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
