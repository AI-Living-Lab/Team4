#!/bin/bash
# ============================================================
# SALMONN2+ PU-VALOR 학습 (0.3 epoch)
#   - Backbone: video-SALMONN2+ 7B (with time tokens)
#   - Data: PU-VALOR (SALMONN2+ format)
#   - LoRA: r=128, alpha=256
#   - GPU 5,7
# ============================================================

source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

export CUDA_VISIBLE_DEVICES=4,5
export ARNOLD_WORKER_GPU=2
export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

cd /home/aix23102/audiolm/video-SALMONN-2/video_SALMONN2_plus

MODEL=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
MODEL_BASE=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
DATASET=/home/aix23102/audiolm/vS2_eunji/data/pu_valor_train_salmonn2plus.json
OUTPUT_DIR=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_0.3ep
RUN_NAME=salmonn2plus_puvalor

# PU-VALOR: 101080 samples, effective_batch=4 (bs=1, accum=2, gpu=2)
# steps_per_epoch = 101080 / 4 = 25270
# 0.3 epoch = 7581 steps
# save every 1264 steps (same as vS2 Stage3)

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
    --max_steps 7581 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_pixels 176400 \
    --min_pixels 784 \
    --video_max_frame_pixels 25088 \
    --video_min_frame_pixels 3136 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1264 \
    --save_total_limit 5 \
    --learning_rate 1e-4 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 6000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name "$RUN_NAME" \
    --report_to wandb \
    --video_min_frames 4 \
    --video_max_frames 128 \
    --base_interval 2 \
    --train_type sft \
    --no_audio False \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"
