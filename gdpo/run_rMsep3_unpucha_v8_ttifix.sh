#!/bin/bash
# ============================================================
# sft_7b_unpucha_v8_rl_rMsep3_unpucha_batch4_noscaling_ttifix
#   = run_rMsep3_unpucha_v8.sh 와 동일 설정(reward_functions_rM_sep3, no-scaling).
#   차이점: 오디오 없을 때도 TTI 가 적용되도록 수정된 최신 master/Team4 코드로 재학습.
#     (기존 noscaling 런[GPU0,1]은 fix 이전 코드로 시작돼 미적용 → 이 런이 fix 반영본)
#   reward 채널명은 global/local 유지 (MUSEG 원 용어 존중; global=set-level F1 / local=segment-level NGIoU).
#   ⚠️ GPU 2,3 사용 (0,1 은 기존 noscaling 런 점유). master_port 29527 (29525 충돌 회피).
#   ⚠️ 신규 run — resume/wandb-resume 없음(새 wandb run 자동 생성). SFT 에서 fresh 시작.
#   ⚠️ eff batch = grad_accum(2) × GPU(2) = 4 (config_sep3.yaml).
# ============================================================
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /home/team404/miniconda3/etc/profile.d/conda.sh
conda activate salmonn2plus
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

# GPU 2,3 (0,1 은 기존 noscaling 런). eff batch = grad_accum(2) × GPU(2) = 4
export CUDA_VISIBLE_DEVICES=2,3

RUN=sft_7b_unpucha_v8_rl_rMsep3_unpucha_batch4_noscaling_ttifix
OUT=$CKPT_DIR/gdpo/$RUN
mkdir -p "$OUT"

# grad_accum=2 → video 재디코드 캐시가 있는 gdpo_trainer_batch.py 사용 (host RAM OOM 방지)
torchrun --standalone --nproc_per_node=2 --master_port=29527 \
  gdpo/gdpo_trainer_batch.py \
  --config       gdpo/config_sep3.yaml \
  --model_path   $CKPT_DIR/base/salmonn2p_7b_unpucha_v8 \
  --model_base   $CKPT_DIR/base/salmonn2p_7b_unpucha_v8 \
  --dataset_path $TRAIN_DIR/unpucha_v2.json \
  --reward_module reward_functions_rM_sep3 \
  --tti_mode on \
  --output_dir "$OUT" \
  --run_name "$RUN" \
  2>&1 | tee "$OUT/train.log"
