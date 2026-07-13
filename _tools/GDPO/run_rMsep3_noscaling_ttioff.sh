#!/bin/bash
# ============================================================
# sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling_ttioff   (TTI 디버깅 1)
#   = run_rMsep3_noscaling.sh 와 모두 동일. 유일한 차이: **tti_mode on → off**.
#     --tti_mode off  → 모델 config.time_token_id_range 클리어(rope OFF 분기),
#                       데이터 입력 마커 tti_time_format=off (special_token 아님).
#     그 외 전부 baseline 그대로: config_sep3.yaml / reward_functions_rM_sep3 /
#                       model salmonn2p_7b_unav_v8 / dataset unpucha_v2.json.
#   목적: baseline(tti on) 대비 tti off 곡선 비교 = TTI 디버깅 1.
#   TTI_DEBUG=1: rank-0 첫 3 step 동안 TTI 정합성 로그 출력(학습 무영향, 진단용).
#   ⚠️ GPU 2,3 사용 (0,1 은 noprecision 런 점유). master_port 29535 (충돌 회피).
#   ⚠️ 신규 run — resume/wandb-resume 없음(새 wandb run 자동 생성). SFT 에서 fresh 시작.
#   ⚠️ H100 2장. eff batch = grad_accum(2) × GPU(2) = 4 (config_sep3.yaml).
# ============================================================
set -eo pipefail
cd /home/team404/workspace/master/Team4
source /home/team404/miniconda3/etc/profile.d/conda.sh
conda activate salmonn2plus
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

# GPU 2,3 (0,1 은 noprecision 런). eff batch = grad_accum(2) × GPU(2) = 4
export CUDA_VISIBLE_DEVICES=2,3

# TTI 정합성 진단 (rank-0, 첫 3 step 만; 학습에는 영향 없음)
export TTI_DEBUG=1
export TTI_DEBUG_STEPS=3

RUN=sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling_ttioff
OUT=$CKPT_DIR/gdpo/$RUN
mkdir -p "$OUT"

# grad_accum=2 → video 재디코드 캐시가 있는 gdpo_trainer_batch.py 사용 (host RAM OOM 방지)
torchrun --standalone --nproc_per_node=2 --master_port=29535 \
  _tools/GDPO/gdpo_trainer_batch.py \
  --config       _tools/GDPO/config_sep3.yaml \
  --model_path   $CKPT_DIR/base/salmonn2p_7b_unav_v8 \
  --model_base   $CKPT_DIR/base/salmonn2p_7b_unav_v8 \
  --dataset_path $TRAIN_DIR/unpucha_v2.json \
  --reward_module reward_functions_rM_sep3 \
  --tti_mode off \
  --output_dir "$OUT" \
  --run_name "$RUN" \
  2>&1 | tee "$OUT/train.log"
