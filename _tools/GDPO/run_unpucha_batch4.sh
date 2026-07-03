#!/bin/bash
set -eo pipefail
cd /home/team404/workspace/master/Team4
source /home/team404/miniconda3/etc/profile.d/conda.sh
conda activate salmonn2plus
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

# ⚠️ 학습 박스의 여유 GPU 2개로 지정. eff batch = grad_accum(2) × GPU(2) = 4
export CUDA_VISIBLE_DEVICES=0,1

# [resume] 기존 wandb run(n46r3ws9)에 이어서 로깅 (같은 그래프에 append).
#   must=반드시 기존 run 재개(없으면 에러) / allow=있으면 재개 없으면 새로.
export WANDB_RUN_ID=n46r3ws9
export WANDB_RESUME=must

RUN=sft_7b_unav_v8_rl_rMsep2fp_lr_clip_mu2_mlp_headoff_unpucha_batch4
OUT=$CKPT_DIR/gdpo/$RUN
mkdir -p "$OUT"

# grad_accum=2 → video 재디코드 캐시가 있는 gdpo_trainer_batch.py 사용 (host RAM OOM 방지)
torchrun --standalone --nproc_per_node=2 --master_port=29522 \
  _tools/GDPO/gdpo_trainer_batch.py \
  --config       _tools/GDPO/config_sep2fp_lr_mlp_headoff_batch4.yaml \
  --model_path   $CKPT_DIR/base/salmonn2p_7b_unav_v8 \
  --model_base   $CKPT_DIR/base/salmonn2p_7b_unav_v8 \
  --dataset_path $TRAIN_DIR/unpucha_v2.json \
  --reward_module reward_functions_sep2fp \
  --tti_mode on \
  --output_dir "$OUT" \
  --run_name "$RUN" \
  --resume_from_checkpoint True \
  2>&1 | tee -a "$OUT/train.log"
