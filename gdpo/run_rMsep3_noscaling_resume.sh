#!/bin/bash
# ============================================================
# sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling  ── RESUME
#   중단된 학습을 checkpoint-1600(global_step 1600)부터 재개.
#   wandb 는 기존 run(ihj2mjzw)에 이어서 기록 (WANDB_RESUME=must + WANDB_RUN_ID).
#   GPU 0,1 사용. eff batch = grad_accum(2) × GPU(2) = 4 (config_sep3.yaml).
# ============================================================
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /home/team404/miniconda3/etc/profile.d/conda.sh
conda activate salmonn2plus
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

export CUDA_VISIBLE_DEVICES=0,1

# ── wandb 이어서 기록 ──
export WANDB_RUN_ID=ihj2mjzw
export WANDB_RESUME=must

RUN=sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling
OUT=$CKPT_DIR/gdpo/$RUN
mkdir -p "$OUT"

torchrun --standalone --nproc_per_node=2 --master_port=29522 \
  gdpo/gdpo_trainer_batch.py \
  --config       gdpo/config_sep3.yaml \
  --model_path   $CKPT_DIR/base/salmonn2p_7b_unav_v8 \
  --model_base   $CKPT_DIR/base/salmonn2p_7b_unav_v8 \
  --dataset_path $TRAIN_DIR/unpucha_v2.json \
  --reward_module reward_functions_rM_sep3 \
  --tti_mode on \
  --output_dir "$OUT" \
  --run_name "$RUN" \
  --resume_from_checkpoint True \
  2>&1 | tee -a "$OUT/train.log"
