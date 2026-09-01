#!/bin/bash
# ============================================================
# sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling_noaudio  ── RESUME
#   = run_rMsep3_noscaling.sh (baseline) 와 동일. 유일한 차이: **오디오 미사용**.
#     dataset  → unpucha_v2_noaudio.json          (audio 키 제거 + use_audio:false)
#     val      → *_noaudio.json 2세트 (config_sep3_noaudio.yaml)
#                → 오디오 없이 학습한 모델을 오디오 없이 평가 (train/eval 조건 일치)
#     그 외 전부 baseline 그대로: reward_functions_rM_sep3 / multi_seg_weight=1.0 /
#                lora r32 MLP포함 / clip ε0.2-0.28 μ=2 / lr 1e-5 / tti_mode on /
#                model salmonn2p_7b_unav_v8.
#
#   ⚠️ max_steps 2000 (baseline 5179 의 앞 2000 구간). baseline 이 2000 step 에서
#      최고 성능이었기에 동일 지점까지만 학습한다. warmup 은 baseline 과 같은
#      **259 step** 으로 고정 (config: warmup_ratio 0.1295×2000=259).
#      → 0~2000 구간 LR 스케줄이 baseline 과 정확히 일치 = 오디오만 단일 변수.
#
#   [왜 use_audio:false 만으로는 안 되는가 — 반드시 읽을 것]
#     qwenvl/data/dataset.py _get_item 은 오디오를 두 군데서 읽는다:
#       757:  if "use_audio" in sources[0] and sources[0]["use_audio"]:   ← 게이트
#       771:  if "audio" in sources[0]:  audio = self.process_audio(...)  ← 게이트를 덮어씀
#     unpucha_v2.json 은 10,358개 전부 "audio" 키를 갖고 wav 도 전부 존재 →
#     charades 3,411개의 use_audio:false 는 무효였고 baseline 은 사실상 전량
#     오디오를 사용했다. 이 런은 "audio" 키 자체를 제거해 audio_lengths=None 경로를 탄다.
#     (data/strip_audio.py 로 생성. dataset.py:205-235 가 그 경로를 1급 지원 —
#      TTI 마커는 그대로 인터리빙되고 audio_pad 만 빠진다.)
#
#   ⚠️ TTI_DEBUG=1 — 오디오를 빼면 마커 인터리빙이 audio 분기(236-263)가 아니라
#      video-only 분기(205-235)를 타므로, 마커가 실제로 들어가는지 반드시 확인한다.
#      로그에서 볼 것 (rank-0):
#        [TTI-DBG] ① load_model  : tti_mode / time_token_id_range 설정
#        [TTI-DBG] ② main        : tti_time_format=special_token ↔ id_range 짝 검증
#        [TTI-DBG] ⑤ step N      : #time_markers_in_prompt  ← **0 이면 실패**
#                                  + prompt[vision_start:+40] 디코드 실물
#      ⑤ 가 0 이거나 "⚠️ WARNING" 이 뜨면 즉시 중단할 것.
#
#   ⚠️ GPU 0,1 (A100 80GB ×2). eff batch = grad_accum(2) × GPU(2) = 4.
#   ⚠️ RESUME — checkpoint-400 에서 재개 (step 400~466 구간은 다시 학습됨).
#      unav-100 평가를 위해 step 466 에서 정지했었다. wandb 는 기존 run(a87crurm)에
#      이어서 기록한다(WANDB_RESUME=must). save_steps 200 이라 최신 ckpt 는 400.
# ============================================================
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /home/team404/miniconda3/etc/profile.d/conda.sh
conda activate salmonn2plus
source paths.env   # BASE_DIR / TRAIN_DIR / CKPT_DIR / WANDB_ENTITY / WANDB_API_KEY

# GPU 0,1. eff batch = grad_accum(2) × GPU(2) = 4
export CUDA_VISIBLE_DEVICES=0,1

# TTI 검증은 최초 런에서 통과 확인됨 → resume 에서는 끈다.
# (다시 보려면 export TTI_DEBUG=1 TTI_DEBUG_STEPS=10)

# HF 의 resume 로그("Continuing training from checkpoint...")는 log_level=passive 라
# 기본적으로 억제된다 → info 로 올려 재개 여부를 눈으로 확인한다.
export TRANSFORMERS_VERBOSITY=info

# ── wandb 이어서 기록 ──
export WANDB_RUN_ID=a87crurm
export WANDB_RESUME=must

SFT_CKPT=$CKPT_DIR/base/salmonn2p_7b_unav_v8
DATASET=$TRAIN_DIR/unpucha_v2_noaudio.json

# ── 사전 확인 — 없으면 여기서 죽는다 (모델 로드 한참 뒤에 실패하지 않도록) ──
#    ⚠️ checkpoints/base/ 에는 salmonn2p_7b_unpucha_v8 이 flat 하게 풀려 있다
#       (rclone copy 로 내용만 복사됨). unav_v8 은 아래 경로에 별도로 받아야 한다:
#         rclone copy gdrive:checkpoints/base/salmonn2p_7b_unav_v8 \
#                     $CKPT_DIR/base/salmonn2p_7b_unav_v8 -P --transfers 16 --fast-list
#       (17.2 GiB — 디스크 여유 확인할 것. 체크포인트 10개에 ~9.4G 더 필요.)
for p in "$SFT_CKPT/config.json" "$DATASET" \
         "$BASE_ROOT/data/val/unav100_tail_val100_noaudio.json" \
         "$BASE_ROOT/data/val/charades_tail_val100_noaudio.json"; do
  [ -e "$p" ] || { echo "❌ 없음: $p"; exit 1; }
done
echo "✅ 모델/데이터 확인 완료"

RUN=sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling_noaudio
OUT=$CKPT_DIR/gdpo/$RUN
mkdir -p "$OUT"

# grad_accum=2 → video 재디코드 캐시가 있는 gdpo_trainer_batch.py 사용 (host RAM OOM 방지)
torchrun --standalone --nproc_per_node=2 --master_port=29540 \
  gdpo/gdpo_trainer_batch.py \
  --config       gdpo/config_sep3_noaudio.yaml \
  --model_path   "$SFT_CKPT" \
  --model_base   "$SFT_CKPT" \
  --dataset_path "$DATASET" \
  --reward_module reward_functions_rM_sep3 \
  --tti_mode on \
  --output_dir "$OUT" \
  --run_name "$RUN" \
  --resume_from_checkpoint "$OUT/checkpoint-400" \
  2>&1 | tee -a "$OUT/train.log"
