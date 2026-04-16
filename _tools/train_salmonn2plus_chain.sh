#!/bin/bash
# ============================================================
# PU-VALOR 학습 완료 대기 → UnAV-100 fine-tuning 자동 시작
# ============================================================
set -eo pipefail

PUVALOR_PID=$(ps aux | grep "train_qwen.py.*salmonn2plus_puvalor" | grep -v grep | head -1 | awk '{print $2}')

if [ -z "$PUVALOR_PID" ]; then
    echo "[INFO] PU-VALOR training not running. Starting UnAV-100 directly."
else
    echo "[INFO] Waiting for PU-VALOR training (PID: $PUVALOR_PID) to finish..."
    while kill -0 "$PUVALOR_PID" 2>/dev/null; do
        sleep 60
    done
    echo "[INFO] PU-VALOR training finished!"
fi

echo "[INFO] Starting UnAV-100 fine-tuning..."
bash /home/aix23102/audiolm/vS2_eunji/_tools/train_salmonn2plus_unav100.sh
