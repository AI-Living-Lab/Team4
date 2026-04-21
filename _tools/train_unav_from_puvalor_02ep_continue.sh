#!/bin/bash
# ============================================================
# UnAV-100 학습 이어서: 1.0ep → 3.0ep (high-res)
#   - Resume from checkpoint-2590 (1.0ep)
#   - save every 0.2ep (518 steps) → 10 new checkpoints
#   - GPU 4,5
# ============================================================
set -eo pipefail

source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

export CUDA_VISIBLE_DEVICES=4,5
export ARNOLD_WORKER_GPU=2
export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

cd /home/aix23102/audiolm/video-SALMONN-2/video_SALMONN2_plus

MODEL=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
DATASET=/home/aix23102/audiolm/vS2_eunji/data/unav100_train_multiseg_salmonn2plus.json
OUTPUT_DIR=/data0/aix23102/checkpoints_open_aligner/unav_from_puvalor_lowres_0.2ep
RUN_NAME=unav_from_puvalor_lowres_0.2ep_cont
RESUME_CKPT=/data0/aix23102/checkpoints_open_aligner/unav_from_puvalor_lowres_0.2ep/checkpoint-2590

# Base: low-res PU-VALOR 0.2ep (same as before)
LORA_CKPT=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_0.3ep_lora_timetoken/checkpoint-5054
echo "Using PU-VALOR LoRA: $LORA_CKPT"
echo "Resuming from: $RESUME_CKPT"

# 3.0ep = 7770 steps (from scratch)
# save every 0.2ep = 518 steps → new ckpts at 3108, 3626, ..., 7770
# save_total_limit 25 to keep all 20 checkpoints (10 old + 10 new)

torchrun --standalone --nproc_per_node=$ARNOLD_WORKER_GPU \
    qwenvl/train/train_qwen.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "$MODEL" \
    --model_base "$MODEL" \
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
    --lora_ckpt "$LORA_CKPT" \
    --bf16 \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --max_steps 7770 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_pixels 176400 \
    --min_pixels 784 \
    --video_max_frame_pixels 28224 \
    --video_min_frame_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 518 \
    --save_total_limit 25 \
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
    --resume_from_checkpoint "$RESUME_CKPT" \
    2>&1 | tee -a "$OUTPUT_DIR/train_continue.log"
