#!/bin/bash
# ============================================================
# 단일 GPU 순차(체인) eval — 2026-06-07
#   GPU 0 사용 (학습은 GPU 2,3 점유 중)
#   각 run 은 eval_salmon2plus_analysis_nochunk_lab.sh 호출.
#   한 run 이 실패해도 다음 run 은 계속 진행한다.
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/eval_salmon2plus_analysis_nochunk_lab.sh"
GPU=0
BASE_MODEL_ID=base/salmonn2p_7b_unav_v8
TESTDIR=/home/aix23102/audiolm/Team404/data/test
# 저장 루트 (lab) — 미지정 시 스크립트가 /workspace/outputs 로 폴백하여 실패함
export OUT_ROOT=/home/aix23102/audiolm/Team404/outputs

run_one () {
    local tag="$1"; shift
    echo ""
    echo "########################################################"
    echo "# [CHAIN] START: $tag  ($(date -Iseconds))"
    echo "########################################################"
    bash "$RUNNER" "$@" GPUS=$GPU \
        && echo "# [CHAIN] DONE: $tag" \
        || echo "# [CHAIN] FAILED: $tag (다음 run 계속)"
}

# 1) B_500 / checkpoint-400  (cot test json)
run_one "B_500/ckpt400 (cot)" \
    STAGE=gdpo CKPT_MODEL_ID=sft_7b_unav_v8_rl_cot_rMfp02_B_500 \
    CKPT_STEP=checkpoint-400 BASE_MODEL_ID=$BASE_MODEL_ID \
    TEST_JSON=$TESTDIR/unav100_v2_500_cot.json

# 2) v2_rM_fp02 / checkpoint-300  (non-cot test json)
run_one "v2_rM_fp02/ckpt300" \
    STAGE=gdpo CKPT_MODEL_ID=sft_7b_unav_v8_rl_v2_rM_fp02 \
    CKPT_STEP=checkpoint-300 BASE_MODEL_ID=$BASE_MODEL_ID \
    TEST_JSON=$TESTDIR/unav100_v2_500.json

# 3) v2_rM_fp02 / checkpoint-400  (non-cot test json)
run_one "v2_rM_fp02/ckpt400" \
    STAGE=gdpo CKPT_MODEL_ID=sft_7b_unav_v8_rl_v2_rM_fp02 \
    CKPT_STEP=checkpoint-400 BASE_MODEL_ID=$BASE_MODEL_ID \
    TEST_JSON=$TESTDIR/unav100_v2_500.json

echo ""
echo "########################################################"
echo "# [CHAIN] ALL DONE  ($(date -Iseconds))"
echo "########################################################"
