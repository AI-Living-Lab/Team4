#!/bin/bash
# ============================================================
# sft_7b_unpucha_v8_rl_rMsep3_unpucha_batch4_noscaling_ttifix_nocount
#   = run_rMsep3_unpucha_v8_ttifix.sh 와 모두 동일. 유일한 차이: **count reward 제거** (ablation).
#     reward_module → reward_functions_rM_sep3_nocount (채널 4개: format/global/local/precision)
#     config        → config_sep3_nocount.yaml (reward_weights 4개)
#   (원본 reward_functions_rM_sep3.py / config_sep3.yaml 무수정 — 별도 파일 신설)
#   ⚠️ GPU 0,1 사용 (2,3 은 ttifix 런 점유). master_port 29529 (29527 ttifix, 29525 구 noscaling 충돌 회피).
#   ⚠️ 신규 run — resume/wandb-resume 없음(새 wandb run 자동 생성). SFT 에서 fresh 시작.
#   ⚠️ eff batch = grad_accum(2) × GPU(2) = 4.
# ============================================================
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /home/team404/miniconda3/etc/profile.d/conda.sh
conda activate salmonn2plus
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

# GPU 0,1 (2,3 은 ttifix 런). eff batch = grad_accum(2) × GPU(2) = 4
export CUDA_VISIBLE_DEVICES=0,1

RUN=sft_7b_unpucha_v8_rl_rMsep3_unpucha_batch4_noscaling_ttifix_nocount
OUT=$CKPT_DIR/gdpo/$RUN
mkdir -p "$OUT"

# grad_accum=2 → video 재디코드 캐시가 있는 gdpo_trainer_batch.py 사용 (host RAM OOM 방지)
torchrun --standalone --nproc_per_node=2 --master_port=29529 \
  gdpo/gdpo_trainer_batch.py \
  --config       gdpo/config_sep3_nocount.yaml \
  --model_path   $CKPT_DIR/base/salmonn2p_7b_unpucha_v8 \
  --model_base   $CKPT_DIR/base/salmonn2p_7b_unpucha_v8 \
  --dataset_path $TRAIN_DIR/unpucha_v2.json \
  --reward_module reward_functions_rM_sep3_nocount \
  --tti_mode on \
  --output_dir "$OUT" \
  --run_name "$RUN" \
  2>&1 | tee "$OUT/train.log"
