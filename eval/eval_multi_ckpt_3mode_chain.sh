#!/bin/bash
# ============================================================
# 다중 체크포인트 × 3-mode 추론 chain
#
# 동작
#   for ckpt in $CKPT_MODEL_IDS:
#     bash eval_3mode_chain.sh CKPT_MODEL_ID=$ckpt ...
#   각 체크포인트마다 off/special_token/natural_text 세 모드 순차 실행.
#   모든 OUT_DIR 은 eval_salmonn2plus.sh 의 자동 resume 으로 이어 진행되므로,
#   이전에 부분 완료된 폴더는 chunk_idx 부터 재시작됨.
#
# 결과 (최대 |CKPTS| × 3 = N 폴더):
#   outputs/<stage>/<CKPT>/<CKPT_FOLDER>/fps<N>_off/<TESTSET>/
#   outputs/<stage>/<CKPT>/<CKPT_FOLDER>/fps<N>_tti/<TESTSET>/
#   outputs/<stage>/<CKPT>/<CKPT_FOLDER>/fps<N>_natural/<TESTSET>/
#
# 사용법 (KEY=VALUE)
#   bash eval_multi_ckpt_3mode_chain.sh \
#       CKPT_MODEL_IDS="salmonn2p_7b_unav_baseline salmonn2p_7b_unav_tti_smoke" \
#       STAGE=sft CKPT_STEP=1500 \
#       BASE_MODEL_ID=base/video_salmonn2_plus_7B_time_tokens \
#       TESTSET=unav100 GPUS=0
#
# 백그라운드 실행
#   cd /workspace/tti_natural/Team4
#   setsid nohup bash eval/eval_multi_ckpt_3mode_chain.sh \
#       CKPT_MODEL_IDS="salmonn2p_7b_unav_baseline salmonn2p_7b_unav_tti_smoke" \
#       STAGE=sft CKPT_STEP=1500 \
#       BASE_MODEL_ID=base/video_salmonn2_plus_7B_time_tokens \
#       TESTSET=unav100 GPUS=0 \
#       < /dev/null > /tmp/eval_multi_3mode.log 2>&1 &
#   disown
#   echo "PID: $!"
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAIN_SCRIPT="$SCRIPT_DIR/eval_3mode_chain.sh"

# ---- 기본값 ----
CKPT_MODEL_IDS="salmonn2p_7b_unav_baseline salmonn2p_7b_unav_tti_smoke"
STAGE=sft
CKPT_STEP=1500
BASE_MODEL_ID=base/video_salmonn2_plus_7B_time_tokens
TESTSET=unav100
GPUS=0
CONFIG=config.yaml
MODES="off special_token natural_text"

# ---- KEY=VALUE 파싱 ----
for arg in "$@"; do
    case "$arg" in
        CKPT_MODEL_IDS=*)   CKPT_MODEL_IDS="${arg#*=}" ;;
        STAGE=*)            STAGE="${arg#*=}" ;;
        CKPT_STEP=*)        CKPT_STEP="${arg#*=}" ;;
        BASE_MODEL_ID=*)    BASE_MODEL_ID="${arg#*=}" ;;
        TESTSET=*)          TESTSET="${arg#*=}" ;;
        GPUS=*)             GPUS="${arg#*=}" ;;
        CONFIG=*)           CONFIG="${arg#*=}" ;;
        MODES=*)            MODES="${arg#*=}" ;;
        *)
            echo "[에러] 지원하지 않는 인자: $arg" >&2
            echo "지원 인자: CKPT_MODEL_IDS, STAGE, CKPT_STEP, BASE_MODEL_ID, TESTSET, GPUS, CONFIG, MODES" >&2
            exit 1
            ;;
    esac
done

# 콤마 → 공백 변환 (편의)
CKPT_MODEL_IDS="${CKPT_MODEL_IDS//,/ }"

cat <<EOF
==================================================
  MULTI-CKPT × 3-MODE EVAL CHAIN
==================================================
  CHAIN_SCRIPT     : $CHAIN_SCRIPT
  CKPT_MODEL_IDS   : $CKPT_MODEL_IDS
  STAGE            : $STAGE
  CKPT_STEP        : $CKPT_STEP
  BASE_MODEL_ID    : $BASE_MODEL_ID
  TESTSET          : $TESTSET
  GPUS             : $GPUS
  CONFIG           : $CONFIG
  MODES            : $MODES
  START            : $(date -Iseconds)
==================================================
EOF

declare -a OK_PAIRS=()
declare -a FAIL_PAIRS=()

for CKPT in $CKPT_MODEL_IDS; do
    echo ""
    echo "##################################################"
    echo "#  [multi-ckpt] CKPT_MODEL_ID=$CKPT  $(date -Iseconds)"
    echo "##################################################"

    if bash "$CHAIN_SCRIPT" \
        STAGE="$STAGE" \
        CKPT_MODEL_ID="$CKPT" \
        CKPT_STEP="$CKPT_STEP" \
        BASE_MODEL_ID="$BASE_MODEL_ID" \
        TESTSET="$TESTSET" \
        GPUS="$GPUS" \
        CONFIG="$CONFIG" \
        MODES="$MODES"
    then
        echo "[multi-ckpt] CKPT=$CKPT  OK   $(date -Iseconds)"
        OK_PAIRS+=("$CKPT")
    else
        rc=$?
        echo "[multi-ckpt] CKPT=$CKPT  PARTIAL/FAIL (rc=$rc)   $(date -Iseconds)" >&2
        FAIL_PAIRS+=("$CKPT(rc=$rc)")
    fi
done

echo ""
echo "=================================================="
echo "  MULTI-CKPT × 3-MODE COMPLETE  $(date -Iseconds)"
echo "=================================================="
echo "  성공          : ${OK_PAIRS[*]:-(없음)}"
echo "  실패/부분실패 : ${FAIL_PAIRS[*]:-(없음)}"
echo "=================================================="

exit "${#FAIL_PAIRS[@]}"
