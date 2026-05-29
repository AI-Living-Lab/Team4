#!/bin/bash
# ============================================================
# v4 ckpt-6064 (4+1, no q_tokens sidecar) vs v5_b ckpt-4500 (3+1, dist-NLL ord)
# 동일 sub500 test sample 에서 mIoU 비교. GPU 6,7 (다른 사용자와 공유, 19GB free).
#
# - v4 ckpt-6064 → eval_v5_sub500.sh TOK=4_1
# - v5_b ckpt-4500 → eval_v5_sub500.sh TOK=3_1
# - 순차 실행 (동시 실행 시 OOM 위험)
# ============================================================
set -eo pipefail
source /home/aix23102/anaconda3/etc/profile.d/conda.sh
conda activate salmonn2plus

LOG_DIR=/home/aix23102/audiolm/vS2_eunji/_tools/eval_v4_v5b_compare_ep04_logs
mkdir -p "$LOG_DIR"
echo "[start] $(date)"

V4_CKPT=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_v4/checkpoint-6064
V5B_CKPT=/data0/aix23102/checkpoints_open_aligner/salmonn2plus_v5_b/checkpoint-4500

# Stage 1: v4 ckpt-6064
echo "========== v4 ckpt-6064 (TOK=4_1) =========="
EVAL_GPUS="6,7" bash /home/aix23102/audiolm/vS2_eunji/_tools/eval_v5_sub500.sh \
    "$V4_CKPT" "4_1" "v4_step6064_ep04" 2>&1 | tee "$LOG_DIR/v4_step6064.log"

# Stage 2: v5_b ckpt-4500
echo "========== v5_b ckpt-4500 (TOK=3_1) =========="
EVAL_GPUS="6,7" bash /home/aix23102/audiolm/vS2_eunji/_tools/eval_v5_sub500.sh \
    "$V5B_CKPT" "3_1" "v5b_step4500_ep04" 2>&1 | tee "$LOG_DIR/v5b_step4500.log"

echo "[done] $(date)"
echo
echo "=== Summary ==="
for d in \
    /data0/aix23102/checkpoints_open_aligner/salmonn2plus_puvalor_v4/eval_sub500_checkpoint-6064_v4_step6064_ep04/eval_miou_summary.json \
    /data0/aix23102/checkpoints_open_aligner/salmonn2plus_v5_b/eval_sub500_checkpoint-4500_v5b_step4500_ep04/eval_miou_summary.json; do
    if [ -f "$d" ]; then
        echo "--- $d ---"
        cat "$d"
        echo
    fi
done
