#!/bin/bash
# ============================================================
# run_rMsep3_unpucha_v8_GRPO_resume.sh
#   run_rMsep3_unpucha_v8_GRPO.sh 재개용. 2026-08-30 02:13 머신 shutdown 으로
#   step 1036/5179 에서 죽은 런을 checkpoint-1000 부터 이어서 돌린다.
#
#   원본과 다른 점 3가지 (그 외 100% 동일):
#     1) --resume_from_checkpoint True   → output_dir 내 최신 ckpt(=1000) 자동 재개
#     2) --max_steps 2000                → config 의 5179(=1 epoch 상한) 대신 2000 에서 종료.
#        GDPO 짝(sft_7b_unpucha_v8_..._noscaling)이 828, unav 쪽이 2002 에서 끊긴 것과
#        비교 구간을 맞추기 위함. lr_scheduler=constant_with_warmup 이고 warmup(5%)은
#        이미 지난 지점이라 max_steps 변경이 LR 에 주는 영향 없음(로그상 lr=1e-5 고정).
#     3) tee -a                          → 기존 train.log(5.6MB, step 1~1036) 보존하고 append
#
#   val_metrics.jsonl 은 트레이너가 "a" 모드로 열므로(gdpo_trainer_batch_GRPO.py:1760)
#   step 1200/1400/1600/1800/2000 이 기존 200~1000 뒤에 이어 붙는다.
#   wandb 는 동일 run_name 으로 자동 resume 된다(선례: unav 런 train.log:51692).
# ============================================================
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /home/team404/miniconda3/etc/profile.d/conda.sh
conda activate salmonn2plus
source paths.env

export CUDA_VISIBLE_DEVICES=0

RUN=sft_7b_unpucha_v8_rl_rMsep3_unpucha_batch4_noscaling_GRPO
OUT=$CKPT_DIR/gdpo/$RUN
mkdir -p "$OUT"

torchrun --standalone --nproc_per_node=1 --master_port=29537 \
  gdpo/gdpo_trainer_batch_GRPO.py \
  --config       gdpo/config_sep3_GRPO_1gpu.yaml \
  --model_path   $CKPT_DIR/base \
  --model_base   $CKPT_DIR/base \
  --dataset_path $TRAIN_DIR/unpucha_v2.json \
  --reward_module reward_functions_rM_sep3 \
  --tti_mode on \
  --output_dir "$OUT" \
  --run_name "$RUN" \
  --resume_from_checkpoint True \
  --max_steps 2000 \
  2>&1 | tee -a "$OUT/train.log"
