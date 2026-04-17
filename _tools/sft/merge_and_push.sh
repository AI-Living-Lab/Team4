#!/bin/bash
# ============================================================
# train_salmonn2plus.sh로 학습된 LoRA 체크포인트를
# 베이스 모델에 merge하고, 선택적으로 HuggingFace Hub에 업로드.
#
# 사용법 (모든 인자 선택, KEY=VALUE 형식):
#   bash merge_and_push.sh \
#       MODEL_ID=salmonn2p7b_unav100_baseline \
#       BASE_MODEL_ID=video_salmonn2_plus_7B_time_tokens \
#       CKPT=5000 \
#       OUTPUT_DIR=/custom/path \
#       REPO_ID=ewhaailab/salmonn2plus-unav100 \
#       PUSH=true \
#       PRIVATE=true \
#       DTYPE=bfloat16
#
# 기본값:
#   CKPT        -> ${CKPT_DIR}/${MODEL_ID} 내 가장 최근 checkpoint-*
#                  (스텝 번호만 넘기면 됨, 예: CKPT=71379)
#   OUTPUT_DIR  -> ${CKPT_DIR}/${MODEL_ID}_merged
#   PUSH        -> false (merge만 하고 업로드 안 함)
#   PRIVATE     -> false
#   DTYPE       -> bfloat16
#
# 인증:
#   사전에 `huggingface-cli login`을 하거나,
#   HF_TOKEN=hf_xxx 환경변수로 넘기세요 (python 스크립트가 자동 인식).
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /workspace/setup.sh
conda activate salmonn2plus

# ---- 기본값 ----
MODEL_ID=salmonn2p7b_unav100_baseline
BASE_MODEL_ID=video_salmonn2_plus_7B_time_tokens
CKPT=
OUTPUT_DIR=
REPO_ID=
PUSH=false
PRIVATE=false
DTYPE=bfloat16

# ---- KEY=VALUE 인자 파싱 ----
for arg in "$@"; do
    case "$arg" in
        MODEL_ID=*)         MODEL_ID="${arg#*=}" ;;
        BASE_MODEL_ID=*)  BASE_MODEL_ID="${arg#*=}" ;;
        CKPT=*)             CKPT="${arg#*=}" ;;
        OUTPUT_DIR=*)       OUTPUT_DIR="${arg#*=}" ;;
        REPO_ID=*)          REPO_ID="${arg#*=}" ;;
        PUSH=*)             PUSH="${arg#*=}" ;;
        PRIVATE=*)          PRIVATE="${arg#*=}" ;;
        DTYPE=*)            DTYPE="${arg#*=}" ;;
        *)
            echo "[에러] 지원하지 않는 인자: $arg"
            echo "지원 인자: MODEL_ID, BASE_MODEL_ID, CKPT, OUTPUT_DIR, REPO_ID, PUSH, PRIVATE, DTYPE"
            exit 1
            ;;
    esac
done

cd "${BASE_DIR}/video_SALMONN2_plus"

BASE_MODEL="${CKPT_DIR}/${BASE_MODEL_ID}"
MODEL_DIR="${CKPT_DIR}/${MODEL_ID}"

# ---- CKPT 미지정 시 가장 최근 체크포인트 자동 선택 ----
# CKPT=71379 처럼 스텝 번호만 넘기면 자동으로 "checkpoint-" 접두사를 붙여준다.
# CKPT=checkpoint-71379 처럼 전체 이름을 넘겨도 동작.
if [ -z "$CKPT" ]; then
    CHECKPOINT_PATH=$(ls -d "$MODEL_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
    if [ -z "$CHECKPOINT_PATH" ]; then
        echo "[에러] $MODEL_DIR 에서 checkpoint-* 를 찾을 수 없습니다"
        exit 1
    fi
else
    if [[ "$CKPT" == checkpoint-* ]]; then
        CHECKPOINT_PATH="${MODEL_DIR}/${CKPT}"
    else
        CHECKPOINT_PATH="${MODEL_DIR}/checkpoint-${CKPT}"
    fi
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${CKPT_DIR}/${MODEL_ID}_merged"
fi

echo "=================================================="
echo "  MODEL_ID        : $MODEL_ID"
echo "  BASE_MODEL      : $BASE_MODEL"
echo "  CHECKPOINT_PATH : $CHECKPOINT_PATH"
echo "  OUTPUT_DIR      : $OUTPUT_DIR"
echo "  DTYPE           : $DTYPE"
echo "  PUSH            : $PUSH"
echo "  REPO_ID         : $REPO_ID"
echo "  PRIVATE         : $PRIVATE"
echo "=================================================="

# ---- PUSH=true일 때만 업로드 인자 구성 ----
PUSH_ARGS=()
if [ "$PUSH" = "true" ] || [ "$PUSH" = "True" ]; then
    if [ -z "$REPO_ID" ]; then
        echo "[에러] PUSH=true일 때 REPO_ID는 필수입니다"
        exit 1
    fi
    PUSH_ARGS+=(--push_to_hub --repo_id "$REPO_ID")
    if [ "$PRIVATE" = "true" ] || [ "$PRIVATE" = "True" ]; then
        PUSH_ARGS+=(--private)
    fi
fi

python "${SCRIPT_DIR}/merge_lora_and_push.py" \
    --base_model_path "$BASE_MODEL" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dtype "$DTYPE" \
    "${PUSH_ARGS[@]}"
