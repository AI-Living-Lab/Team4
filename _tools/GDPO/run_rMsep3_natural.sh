#!/bin/bash
# ============================================================
# sft_7b_unav_v8_chronus_rl_rMsep3_unpucha_batch4_natural
#   "natural mode" 실험 (chronus SFT warm-start 판) — 입력마커 = natural_text('second{XXX.Y}', 8토큰),
#   출력/GT = chronus('second{start}-second{end}. ...').
#   ★ model = salmonn2p_7b_unav_v8_chronus (chronus 250-step SFT → merge). base-v8 직접 RL 은
#     포맷 콜드스타트로 죽었으므로(natural 1차 시도) chronus SFT 로 포맷을 먼저 데운 모델 사용.
#
#   sep3(run_rMsep3_noscaling.sh) 와 그 외 설정 100% 동일. 다른 점만:
#     · trainer  : gdpo_trainer_batch_natural.py       (GT/val 파싱 chronus, rope natural 분기)
#     · config   : config_sep3_natural.yaml            (tti_time_format=natural_text)
#     · dataset  : unpucha_chronus.json                (출력/GT chronus)
#     · reward   : reward_functions_rM_sep3_natural    (chronus 파서, 채널 5개 의미 동일)
#   ⚠️ 신규 run — resume 없음(SFT 에서 fresh 시작). 새 wandb run 자동 생성.
#   ⚠️ H100 2장. eff batch = grad_accum(2) × GPU(2) = 4 (config_sep3_natural.yaml).
#   ⚠️ 기존 job 과 GPU/포트 충돌 주의 — CUDA_VISIBLE_DEVICES / master_port 는 빈 자원으로 조정.
# ============================================================
set -eo pipefail
cd /home/team404/workspace/master/Team4
source /home/team404/miniconda3/etc/profile.d/conda.sh
conda activate salmonn2plus
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

# H100 2장. ⚠️ 다른 학습이 점유 중이면 비어있는 GPU 로 바꿀 것.
export CUDA_VISIBLE_DEVICES=2,3

RUN=sft_7b_unav_v8_chronus_rl_rMsep3_unpucha_batch4_natural
OUT=$CKPT_DIR/gdpo/$RUN
mkdir -p "$OUT"

# grad_accum=2 → video 재디코드 캐시가 있는 batch 트레이너(natural 변형) 사용 (host RAM OOM 방지)
# master_port 는 기존 sep3(29522) 과 겹치지 않게 29523 사용.
torchrun --standalone --nproc_per_node=2 --master_port=29523 \
  _tools/GDPO/gdpo_trainer_batch_natural.py \
  --config       _tools/GDPO/config_sep3_natural.yaml \
  --model_path   $CKPT_DIR/base/salmonn2p_7b_unav_v8_chronus \
  --model_base   $CKPT_DIR/base/salmonn2p_7b_unav_v8_chronus \
  --dataset_path $TRAIN_DIR/unpucha_chronus.json \
  --reward_module reward_functions_rM_sep3_natural \
  --tti_mode on \
  --output_dir "$OUT" \
  --run_name "$RUN" \
  2>&1 | tee "$OUT/train.log"
