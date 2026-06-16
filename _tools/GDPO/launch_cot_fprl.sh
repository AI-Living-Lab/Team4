#!/bin/bash
# ============================================================
# 새 실험 런처 — v8 CoT(P2) + FP페널티↓ + r_L가중↑ (blanket 1세그먼트 붕괴 억제)
#
#   진단: sft_7b_unav_v8_rl_cot_p2_500 이 "긴 1개 blanket" 으로 붕괴
#         → 멀티세그(21.8)·짧은GT(<5s 5~10) 천장. 원인 = FP페널티(0.2) + r_G(coverage) 보상.
#   처방: FP_PENALTY_K↓ (멀티세그 허용) + RM_RG_WEIGHT↓ (r_L↑, blanket 벌점·정밀도↑).
#
#   env 두 knob 만 다르고 나머지(config/데이터/프롬프트/리워드모듈)는 P2와 동일.
#     FP_PENALTY_K  : unmatched pred 1개당 감점 (기존 0.2)
#     RM_RG_WEIGHT  : r_M = w·r_G + (1-w)·r_L 의 r_G 가중 (기존 0.5)
#
#   사용: bash launch_cot_fprl.sh <GPUS(2장)> <FP_K> <RG_W> <PORT> <SUFFIX>
#     예) bash launch_cot_fprl.sh 4,5 0.05 0.3 29510 fp005_rg03
# ============================================================
set -e
GPUS="${1:?GPUS 필요 (예: 4,5)}"
FP="${2:?FP_PENALTY_K 필요 (예: 0.05)}"
RGW="${3:?RM_RG_WEIGHT 필요 (예: 0.3)}"
PORT="${4:?master_port 필요 (예: 29510)}"
SUF="${5:?run suffix 필요 (예: fp005_rg03)}"

source ~/anaconda3/etc/profile.d/conda.sh && conda activate salmonn2plus
cd /home/aix23102/audiolm/Team404/master/Team4
set -a && source paths.env && set +a

RUN=sft_7b_unav_v8_rl_cot_p2_${SUF}_500
OUT=${CKPT_DIR}/gdpo/${RUN}
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=$GPUS FP_PENALTY_K=$FP RM_RG_WEIGHT=$RGW TTI_DEBUG=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid nohup torchrun --standalone --nproc_per_node=2 --master_port=$PORT \
    _tools/GDPO/gdpo_trainer.py \
    --config        _tools/GDPO/config_cot_p2.yaml \
    --model_path    ${CKPT_DIR}/base/salmonn2p_7b_unav_v8 \
    --model_base    ${CKPT_DIR}/base/salmonn2p_7b_unav_v8 \
    --dataset_path  ${TRAIN_DIR}/unav100_v2_p2.json \
    --reward_module reward_functions_cot_p2 \
    --tti_mode      on \
    --output_dir    "$OUT" \
    --run_name      ${RUN} \
    --max_steps     500 \
    < /dev/null > "$OUT/train.log" 2>&1 &

echo "launched: $RUN"
echo "  GPU=$GPUS  FP_PENALTY_K=$FP  RM_RG_WEIGHT=$RGW  port=$PORT"
echo "  log=$OUT/train.log"
