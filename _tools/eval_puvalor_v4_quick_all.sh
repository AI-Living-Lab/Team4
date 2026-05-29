#!/bin/bash
# ============================================================
# salmonn2plus_puvalor_v4 quick eval wrapper
#   - 2 ckpts (10612, 12128) × 3 prompt variants (full/none/min) = 6 runs
#   - ckpt 두 개를 병렬로 (GPU 5, GPU 7) — 각 GPU 안에서 3 variant sequential
#   - merge step skip (merged 디렉터리 존재 가정 — 이미 trained q_tokens 반영됨)
# ============================================================
set -eo pipefail

cd /home/aix23102/audiolm/vS2_eunji

V4_DIR=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_v4
DATA_DIR=/home/aix23102/audiolm/vS2_eunji/data

VARIANTS=(full none min)

LOG_DIR=/home/aix23102/audiolm/vS2_eunji/_tools/eval_v4_quick_logs
mkdir -p "$LOG_DIR"

SUMMARY=$LOG_DIR/summary.txt
echo "=== eval_puvalor_v4_quick_all.sh started at $(date) ===" | tee -a "$SUMMARY"

# Per-ckpt sequential runner (3 variants on a single GPU)
run_ckpt() {
    local C=$1
    local GPU=$2
    local MERGED="$V4_DIR/checkpoint-${C}_merged"
    for V in "${VARIANTS[@]}"; do
        local TEST="$DATA_DIR/puvalor_test_quick_${V}.json"
        local LOG="$LOG_DIR/ckpt${C}_${V}.log"
        echo "" | tee -a "$SUMMARY"
        echo "----------------------------------------------------" | tee -a "$SUMMARY"
        echo "[$(date)] [GPU $GPU] START ckpt-${C}  variant=${V}" | tee -a "$SUMMARY"
        echo "  test=$TEST" | tee -a "$SUMMARY"
        echo "  log =$LOG"  | tee -a "$SUMMARY"
        if [ ! -f "$TEST" ]; then
            echo "[SKIP] test json missing: $TEST" | tee -a "$SUMMARY"
            continue
        fi
        EVAL_GPU=$GPU bash _tools/eval_puvalor_v4_quick.sh "$MERGED" "$TEST" "$V" 2>&1 | tee "$LOG"
        echo "[$(date)] [GPU $GPU] DONE  ckpt-${C}  variant=${V}" | tee -a "$SUMMARY"
    done
}

# Launch two ckpt branches in parallel
run_ckpt 10612 5 &
PID_A=$!
run_ckpt 12128 7 &
PID_B=$!

echo "[wrapper] ckpt-10612 PID=$PID_A on GPU 5  |  ckpt-12128 PID=$PID_B on GPU 7" | tee -a "$SUMMARY"

wait $PID_A
RC_A=$?
wait $PID_B
RC_B=$?

echo "" | tee -a "$SUMMARY"
echo "[wrapper] ckpt-10612 exit=$RC_A   ckpt-12128 exit=$RC_B" | tee -a "$SUMMARY"
echo "=== eval_puvalor_v4_quick_all.sh finished at $(date) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "Result dirs:" | tee -a "$SUMMARY"
ls -d /home/aix23102/audiolm/vS2_eunji/eval/results/puvalor_v4_quick_* 2>/dev/null | tee -a "$SUMMARY"
