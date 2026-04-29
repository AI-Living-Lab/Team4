#!/bin/bash
# ============================================================
# 3-mode 추론 chain — 동일 체크포인트로 off / special_token / natural_text
# 세 가지 추론 모드를 순차 평가.
#
# 동작
#   for mode in $MODES:
#     bash eval_salmonn2plus.sh ... TTI_TIME_FORMAT=$mode
#   결과는 EVAL_TAG (= fps<N>_<format>) 별로 분리되어 저장:
#     outputs/<stage>/<MODEL_ID>/<CKPT_FOLDER>/fps<N>_off/<TESTSET>/
#     outputs/<stage>/<MODEL_ID>/<CKPT_FOLDER>/fps<N>_tti/<TESTSET>/
#     outputs/<stage>/<MODEL_ID>/<CKPT_FOLDER>/fps<N>_natural/<TESTSET>/
#
# 사용법 (KEY=VALUE)
#   bash eval_3mode_chain.sh \
#       STAGE=sft CKPT_MODEL_ID=salmonn2p_7b_unav_baseline CKPT_STEP=1500 \
#       BASE_MODEL_ID=base/video_salmonn2_plus_7B_time_tokens \
#       TESTSET=unav100 GPUS=0
#
# 백그라운드 실행 예시
#   cd /workspace/tti_natural/Team4
#   setsid nohup bash eval/eval_3mode_chain.sh \
#       STAGE=sft CKPT_MODEL_ID=salmonn2p_7b_unav_baseline CKPT_STEP=1500 \
#       BASE_MODEL_ID=base/video_salmonn2_plus_7B_time_tokens \
#       TESTSET=unav100 GPUS=0 \
#       < /dev/null > /tmp/eval_3mode_chain.log 2>&1 &
#   disown
#   echo "PID: $!"
#
# 모드별 한 번씩 실패해도 다음 모드는 계속 진행 (전체 종료 코드는 실패 수).
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/eval_salmonn2plus.sh"

# ---- 기본값 ----
STAGE=sft
CKPT_MODEL_ID=salmonn2p_7b_unav_baseline
CKPT_STEP=
BASE_MODEL_ID=video_salmonn2_plus_7B_time_tokens
TESTSET=unav100
GPUS=0
CONFIG=config.yaml
MODES="off special_token natural_text"

# ---- KEY=VALUE 파싱 ----
for arg in "$@"; do
    case "$arg" in
        STAGE=*)            STAGE="${arg#*=}" ;;
        CKPT_MODEL_ID=*)    CKPT_MODEL_ID="${arg#*=}" ;;
        CKPT_STEP=*)        CKPT_STEP="${arg#*=}" ;;
        BASE_MODEL_ID=*)    BASE_MODEL_ID="${arg#*=}" ;;
        TESTSET=*)          TESTSET="${arg#*=}" ;;
        GPUS=*)             GPUS="${arg#*=}" ;;
        CONFIG=*)           CONFIG="${arg#*=}" ;;
        MODES=*)            MODES="${arg#*=}" ;;
        *)
            echo "[에러] 지원하지 않는 인자: $arg" >&2
            echo "지원 인자: STAGE, CKPT_MODEL_ID, CKPT_STEP, BASE_MODEL_ID, TESTSET, GPUS, CONFIG, MODES" >&2
            exit 1
            ;;
    esac
done

cat <<EOF
==================================================
  3-MODE EVAL CHAIN
==================================================
  EVAL_SCRIPT   : $EVAL_SCRIPT
  STAGE         : $STAGE
  CKPT_MODEL_ID : $CKPT_MODEL_ID
  CKPT_STEP     : ${CKPT_STEP:-<latest>}
  BASE_MODEL_ID : $BASE_MODEL_ID
  TESTSET       : $TESTSET
  GPUS          : $GPUS
  CONFIG        : $CONFIG
  MODES         : $MODES
  START         : $(date -Iseconds)
==================================================
EOF

declare -a OK_MODES=()
declare -a FAIL_MODES=()

for MODE in $MODES; do
    echo ""
    echo "=================================================="
    echo "  [chain] mode=$MODE   start=$(date -Iseconds)"
    echo "=================================================="

    if bash "$EVAL_SCRIPT" \
        STAGE="$STAGE" \
        CKPT_MODEL_ID="$CKPT_MODEL_ID" \
        CKPT_STEP="$CKPT_STEP" \
        BASE_MODEL_ID="$BASE_MODEL_ID" \
        TESTSET="$TESTSET" \
        GPUS="$GPUS" \
        CONFIG="$CONFIG" \
        TTI_TIME_FORMAT="$MODE"
    then
        echo "[chain] mode=$MODE  OK   $(date -Iseconds)"
        OK_MODES+=("$MODE")
    else
        rc=$?
        echo "[chain] mode=$MODE  FAIL (rc=$rc)   $(date -Iseconds)" >&2
        FAIL_MODES+=("$MODE")
    fi
done

echo ""
echo "=================================================="
echo "  3-MODE EVAL CHAIN COMPLETE  $(date -Iseconds)"
echo "=================================================="
echo "  성공  : ${OK_MODES[*]:-(없음)}"
echo "  실패  : ${FAIL_MODES[*]:-(없음)}"
echo "=================================================="

# 종료 코드 = 실패 모드 수 (0 = 전부 성공)
exit "${#FAIL_MODES[@]}"
