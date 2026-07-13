#!/bin/bash
# ============================================================
# sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_nosft
#   ★ SFT 없이 base(time-token) 모델에서 바로 GDPO(RL). sep3(noscaling)과 그 외 100% 동일.
#     차이: model_path/model_base = video_salmonn2_plus_7B_time_tokens (SFT-머지본 아님).
#   base 엔 adapter_config 없음 → fresh RL LoRA(경로 B). config 에 time_token_id_range 없어
#     트레이너가 토크나이저에서 (<t0>..<tdot>)=(151666,151676), marker_len=5 로 복원.
#   출력/GT/reward = special_token(그대로) → format reward 콜드스타트 없음.
#   ⚠️ config/trainer/reward/dataset 는 sep3 것 그대로 재사용. CLI 가 model/output/run 오버라이드.
#   ⚠️ 신규 run — resume 없음(base 에서 fresh 시작). 새 wandb run 자동 생성.
#   ⚠️ H100 2장. eff batch = grad_accum(2) × GPU(2) = 4 (config_sep3.yaml).
# ============================================================
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /home/team404/miniconda3/etc/profile.d/conda.sh
conda activate salmonn2plus
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

# ⚠️ 0,1 이 다른 학습 점유 중 → 비어있는 2,3 사용. 바뀌면 조정.
export CUDA_VISIBLE_DEVICES=2,3

RUN=sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_nosft
OUT=$CKPT_DIR/gdpo/$RUN
mkdir -p "$OUT"

# grad_accum=2 → video 재디코드 캐시가 있는 batch 트레이너 사용 (host RAM OOM 방지)
# master_port 는 기존 run 들과 겹치지 않게 29524 사용.
torchrun --standalone --nproc_per_node=2 --master_port=29524 \
  gdpo/gdpo_trainer_batch.py \
  --config       gdpo/config_sep3.yaml \
  --model_path   $CKPT_DIR/base/video_salmonn2_plus_7B_time_tokens \
  --model_base   $CKPT_DIR/base/video_salmonn2_plus_7B_time_tokens \
  --dataset_path $TRAIN_DIR/unpucha_v2.json \
  --reward_module reward_functions_rM_sep3 \
  --tti_mode on \
  --output_dir "$OUT" \
  --run_name "$RUN" \
  2>&1 | tee "$OUT/train.log"
