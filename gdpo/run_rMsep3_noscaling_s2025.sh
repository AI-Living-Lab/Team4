#!/bin/bash
# ============================================================
# sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling_s2025
#   run_rMsep3_noscaling.sh 의 시드 복제본 (멀티시드 리버틀용).
#   원본(seed 2024, 2026-07-03) 대비 바뀌는 것은 seed / RUN 이름 / master_port 뿐.
#
#   ⚠️ SFT 는 동일하게 유지된다 — model_path/model_base 모두 원본과 같은
#      salmonn2p_7b_unav_v8 (SFT 머지본). 시드는 RL 단계만 바꾼다:
#      LoRA fresh init + 데이터 순서 + 샘플링.
#   ⚠️ 학습 데이터 파일도 동일(unpucha_v2.json). 시드는 데이터를 다시 뽑지 않는다
#      — 데이터 draw 는 data/train/build_unpucha_v2.py 의 SEED=42 소관.
#
#   ⚠️ --max_steps 를 주지 말 것. warmup_ratio(0.05)가 총 step 수에 곱해지므로
#      max_steps 를 1400 으로 낮추면 warmup 이 259 → 70 step 으로 줄어
#      원본과 다른 LR 스케줄이 된다. config 값(5179) 그대로 두고
#      step 1400 도달 후 수동 종료할 것 (save_steps=200 이라 자동 저장됨).
#
#   ⚠️ 하드웨어가 원본과 다르다. 원본은 H100 2장(world_size=2, grad_accum=2),
#      이 머신은 A100 1장뿐이다. effective batch 를 4 로 맞추기 위해
#      nproc_per_node=1 + grad_accum=4 로 돌린다.
#        eff batch  = grad_accum × num_gpu = 4×1 = 4   (원본 2×2 = 4)
#        block_size = grad_accum × num_processes = 4   (원본과 동일)
#      → optimizer step 당 서로 다른 prompt 4개×μ 라는 구조는 보존된다.
#      단 prompt 가 rank 에 round-robin 분배되는 대신 한 rank 에서 순차 처리되므로
#      비트 단위 동일은 아니다. 하드웨어(A100 vs H100) 차이도 남는다.
#      → 이 런의 차이는 "시드 + 데이터 재구성 + 병렬화/하드웨어" 가 섞인 값이다.
#   ⚠️ 데이터 병렬 상실 + A100/H100 격차로 step 1400 에 10시간 안팎 예상.
# ============================================================
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /home/team404/miniconda3/etc/profile.d/conda.sh
conda activate salmonn2plus
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

# A100 1장. eff batch = grad_accum(4) × GPU(1) = 4  (원본 H100 2장 × accum 2 와 동일)
export CUDA_VISIBLE_DEVICES=0

SEED=2025
RUN=sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling_s${SEED}
OUT=$CKPT_DIR/gdpo/$RUN
mkdir -p "$OUT"

# grad_accum=2 → video 재디코드 캐시가 있는 gdpo_trainer_batch.py 사용 (host RAM OOM 방지)
torchrun --standalone --nproc_per_node=1 --master_port=29525 \
  gdpo/gdpo_trainer_batch.py \
  --config       gdpo/config_sep3.yaml \
  --gradient_accumulation_steps 4 \
  --model_path   $CKPT_DIR/base/salmonn2p_7b_unav_v8 \
  --model_base   $CKPT_DIR/base/salmonn2p_7b_unav_v8 \
  --dataset_path $TRAIN_DIR/unpucha_v2.json \
  --reward_module reward_functions_rM_sep3 \
  --tti_mode on \
  --seed "$SEED" \
  --output_dir "$OUT" \
  --run_name "$RUN" \
  2>&1 | tee "$OUT/train.log"
