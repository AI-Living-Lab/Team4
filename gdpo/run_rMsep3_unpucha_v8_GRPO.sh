#!/bin/bash
# ============================================================
# sft_7b_unpucha_v8_rl_rMsep3_unpucha_batch4_noscaling_GRPO
#   run_rMsep3_unpucha_v8.sh(= sft_7b_unpucha_v8_rl_rMsep3_unpucha_batch4_noscaling)와
#   설정 100% 동일, trainer 만 교체:
#     gdpo_trainer_batch.py -> gdpo_trainer_batch_GRPO.py
#   model_path/model_base = $CKPT_DIR/base 는 salmonn2p_7b_unpucha_v8 과 byte-identical
#   (rclone check 로 md5 대조 확인, 2026-08-28) -> SFT 가 이미 머지된 모델. nosft 아님.
#   reward/dataset/tti 는 원본 그대로 재사용 (CLI 로 오버라이드).
#
#   ⚠️ 신규 run — resume/wandb-resume 없음(새 wandb run 자동 생성).
#   ⚠️ GPU 1장(A100 80GB) 전용 — config_sep3_GRPO_1gpu.yaml 사용.
#     원본 config_sep3.yaml(H100 2장 가정, grad_accum=2)과 달리
#     grad_accum=4 로 올려서 effective batch=4 를 그대로 유지
#     (grad_accum(4) x GPU(1) = grad_accum(2) x GPU(2)). max_steps=5179 도 그대로 유효.
#     트레이너 코드 자체는 world_size 를 동적으로 읽어서 1 GPU 에 안전
#     (gdpo_trainer_batch_GRPO.py:652 주석 참고). 코드 수정은 불필요.
#   ⚠️ 포트: 원본 run_rMsep3_unpucha_v8.sh 가 이미 29525 사용 중 -> 여긴 29537 사용(충돌 시 조정).
#   ⚠️ dataset_path($TRAIN_DIR/unpucha_v2.json), val 세트 2종은 준비 완료(2026-08-28 확인).
# ============================================================
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /home/team404/miniconda3/etc/profile.d/conda.sh
conda activate salmonn2plus
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

# GPU 1장(A100) 환경
export CUDA_VISIBLE_DEVICES=0

RUN=sft_7b_unpucha_v8_rl_rMsep3_unpucha_batch4_noscaling_GRPO
OUT=$CKPT_DIR/gdpo/$RUN
mkdir -p "$OUT"

# grad_accum>1 -> video 재디코드 캐시가 있는 batch 트레이너 계열 사용 (host RAM OOM 방지)
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
  2>&1 | tee "$OUT/train.log"
