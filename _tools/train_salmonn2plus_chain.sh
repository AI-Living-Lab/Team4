#!/bin/bash
# ============================================================
# PU-VALOR 0.5ep → UnAV-100 1ep → 전체 checkpoint eval
# ============================================================
set -eo pipefail

echo "[INFO] $(date) Starting PU-VALOR 0.5ep training..."
bash /home/aix23102/audiolm/vS2_eunji/_tools/train_salmonn2plus_puvalor.sh
echo "[INFO] $(date) PU-VALOR training finished!"

echo "[INFO] $(date) Starting UnAV-100 1ep fine-tuning..."
bash /home/aix23102/audiolm/vS2_eunji/_tools/train_salmonn2plus_unav100.sh
echo "[INFO] $(date) UnAV-100 training finished!"

echo "[INFO] $(date) Starting evaluation on all checkpoints..."
bash /home/aix23102/audiolm/vS2_eunji/_tools/eval_all_checkpoints.sh
echo "[INFO] $(date) All done!"
