#!/bin/bash
# ============================================================
# sft_7b_unpucha_v8_rl_rMsep3_unpucha_batch4_noscaling
#   reward_functions_rM_sep3 (fp→precision 연속, len→count) + multi_seg_weight=1.0(no-scaling).
#   데이터셋 unpucha_v2.json 유지 (charades 오디오 제거된 현재 버전).
#   ⚠️ 신규 run — resume/wandb-resume 없음(새 wandb run 자동 생성). SFT 에서 fresh 시작.
#   ⚠️ H100 2장. eff batch = grad_accum(2) × GPU(2) = 4 (config_sep3.yaml).
# ============================================================
set -eo pipefail
cd /home/team404/workspace/master/Team4
source /home/team404/miniconda3/etc/profile.d/conda.sh
conda activate salmonn2plus
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

# H100 2장. eff batch = grad_accum(2) × GPU(2) = 4
export CUDA_VISIBLE_DEVICES=0,1

RUN=sft_7b_unpucha_v8_rl_rMsep3_unpucha_batch4_noscaling
OUT=$CKPT_DIR/gdpo/$RUN
mkdir -p "$OUT"

# grad_accum=2 → video 재디코드 캐시가 있는 gdpo_trainer_batch.py 사용 (host RAM OOM 방지)
torchrun --standalone --nproc_per_node=2 --master_port=29525 \
  _tools/GDPO/gdpo_trainer_batch.py \
  --config       _tools/GDPO/config_sep3.yaml \
  --model_path   $CKPT_DIR/base/salmonn2p_7b_unpucha_v8 \
  --model_base   $CKPT_DIR/base/salmonn2p_7b_unpucha_v8 \
  --dataset_path $TRAIN_DIR/unpucha_v2.json \
  --reward_module reward_functions_rM_sep3 \
  --tti_mode on \
  --output_dir "$OUT" \
  --run_name "$RUN" \
  2>&1 | tee "$OUT/train.log"
