#!/bin/bash
# ============================================================
# v8 DRY-RUN: 2 step 만 실행해서 loss / 세팅 검증.
#   - train_v8.sh 동일 hyperparam, 다만:
#     * max_steps=2 (num_train_epochs 무시)
#     * eval/save 끔
#     * wandb 끔 (report_to none)
#     * OUTPUT_DIR 분리 (_dryrun)
#     * GPU 1대 (0)
# ============================================================
set -eo pipefail
source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

export CUDA_VISIBLE_DEVICES=0
export ARNOLD_WORKER_GPU=1
export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/aix23102/audiolm/workspace/master/Team4/v8_sft/video_SALMONN2_plus

MODEL=/data0/aix23102/checkpoints_open_aligner/video_salmonn2_plus_7B_time_tokens
MODEL_BASE=$MODEL
DATASET=/home/aix23102/audiolm/Team404/data/train/unav100_v2.json
EVAL_DATASET=/home/aix23102/audiolm/Team404/data/test/unav100_v2_500.json
OUTPUT_DIR=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_v8_dryrun
RUN_NAME=salmonn2plus_v8_dryrun
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
    --max_steps 2 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_pixels 176400 --min_pixels 784 \
    --video_max_frame_pixels 28224 --video_min_frame_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "no" \
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
    2>&1 | tee "$OUTPUT_DIR/dryrun.log"

echo "[v8 dryrun] DONE"
