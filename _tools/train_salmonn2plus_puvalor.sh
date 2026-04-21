#!/bin/bash
# ============================================================
# SALMONN2+ PU-VALOR 학습 (0.5 epoch)
#   - Backbone: video-SALMONN2+ 7B (with time tokens)
#   - Data: PU-VALOR (SALMONN2+ format)
#   - LoRA: r=128, alpha=256 (q/k/v_proj)
#   - time token embed_tokens + lm_head 학습 (modules_to_save)
#   - Aligner freeze (tune_mm_mlp=False, tune_mm_qformer=False)
#   - High-res: 28224/784, 64 frames, interval=0.2, max_len=100000
#   - GPU 4,5
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
OUTPUT_DIR=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_0.5ep_highres
RUN_NAME=salmonn2plus_puvalor_highres

# PU-VALOR: 101080 samples, effective_batch=4 (bs=1, accum=2, gpu=2)
# steps_per_epoch = 101080 / 4 = 25270
# 0.5 epoch = 12635 steps
# save every 0.1 epoch = 2527 steps

mkdir -p "$OUTPUT_DIR"

torchrun --standalone --nproc_per_node=$ARNOLD_WORKER_GPU \
    qwenvl/train/train_qwen.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "$MODEL" \
    --model_base "$MODEL_BASE" \
    --dataset_use "$DATASET" \
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
    --max_steps 12635 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_pixels 176400 \
    --min_pixels 784 \
    --video_max_frame_pixels 28224 \
    --video_min_frame_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2527 \
    --save_total_limit 6 \
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
