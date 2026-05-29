#!/bin/bash
# ============================================================
# salmonn2plus_puvalor_v4: 선택 체크포인트 순차 평가 wrapper
#   - eval_puvalor_v4.sh 를 ckpt-10612, ckpt-12128 두 개에 대해 순차 호출
#   - 각 라운드마다 GPU 4,5 모두 사용 → 동시 실행 불가, sequential
#   - Test set: puvalor_test_gout_hinted.json (format 예시 추가됨)
# ============================================================
set -eo pipefail

cd /home/aix23102/audiolm/vS2_eunji

V4_DIR=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_v4

# 평가 대상 체크포인트 2개 (epoch 0.7, 0.8 — eval_loss 가장 낮은 두 개)
CKPTS=(
    "$V4_DIR/checkpoint-10612"
    "$V4_DIR/checkpoint-12128"
)

LOG_DIR=/home/aix23102/audiolm/vS2_eunji/_tools/eval_v4_logs
mkdir -p "$LOG_DIR"

SUMMARY=$LOG_DIR/summary.txt
echo "=== eval_puvalor_v4_all.sh started at $(date) ===" | tee -a "$SUMMARY"

for CKPT in "${CKPTS[@]}"; do
    NAME=$(basename "$CKPT")
    LOG=$LOG_DIR/${NAME}.log
    echo "" | tee -a "$SUMMARY"
    echo "--------------------------------------------------------------------" | tee -a "$SUMMARY"
    echo "[$(date)] START $NAME" | tee -a "$SUMMARY"
    echo "log: $LOG" | tee -a "$SUMMARY"
    if [ ! -d "$CKPT" ]; then
        echo "[SKIP] $CKPT not found" | tee -a "$SUMMARY"
        continue
    fi
    bash _tools/eval_puvalor_v4.sh "$CKPT" 2>&1 | tee "$LOG"
    echo "[$(date)] DONE  $NAME" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "=== eval_puvalor_v4_all.sh finished at $(date) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "Result dirs:" | tee -a "$SUMMARY"
ls -d /home/aix23102/audiolm/vS2_eunji/eval/results/puvalor_v4_checkpoint-* 2>/dev/null | tee -a "$SUMMARY"
