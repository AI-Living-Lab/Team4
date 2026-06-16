#!/bin/bash
set -eo pipefail
cd /home/aix23102/audiolm/Team404/master/Team4
source paths.env
source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus
export CUDA_VISIBLE_DEVICES=0,4
RUN=sft_7b_unav_v8_rl_rMsep2fp_lr_clip_mu2_mlp_headoff_unpu
torchrun --standalone --nproc_per_node=2 --master_port=29521 \
  _tools/GDPO/gdpo_trainer.py \
  --config _tools/GDPO/config_sep2fp_lr_mlp_headoff.yaml \
  --model_path  $CKPT_DIR/base/salmonn2p_7b_unav_v8 \
  --model_base  $CKPT_DIR/base/salmonn2p_7b_unav_v8 \
  --dataset_path $TRAIN_DIR/unpu_v2.json \
  --reward_module reward_functions_sep2fp \
  --tti_mode on \
  --output_dir $CKPT_DIR/gdpo/$RUN \
  --run_name $RUN
