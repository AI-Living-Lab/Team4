#!/bin/bash
set -e
PID="${1:-0}"
OUT_DIR=/workspace/jsy/output/avicuna_unavqa_json_hybrid
PRED="$OUT_DIR/predictions.jsonl"
LOG="$OUT_DIR/watcher.log"
PY=/workspace/miniconda3/envs/chronusomni/bin/python

mkdir -p "$OUT_DIR"
echo "[watcher $(date -u +'%Y-%m-%dT%H:%M:%SZ')] polling PID=$PID" | tee -a "$LOG"

while kill -0 "$PID" 2>/dev/null; do sleep 30; done

echo "[watcher $(date -u +'%Y-%m-%dT%H:%M:%SZ')] PID $PID exited" | tee -a "$LOG"
if [ ! -f "$PRED" ]; then
    echo "[watcher ERROR] pred missing" | tee -a "$LOG"; exit 1
fi

N=$(wc -l < "$PRED")
echo "[watcher] pred n=$N" | tee -a "$LOG"

echo "[watcher $(date -u +'%Y-%m-%dT%H:%M:%SZ')] running eval" | tee -a "$LOG"
"$PY" /workspace/jsy/scripts/eval_miou_multiseg_avicuna_json.py \
    --predictions "$PRED" \
    --max_time 60 \
    --out_dir "$OUT_DIR" 2>&1 | tee -a "$LOG"

echo "[watcher $(date -u +'%Y-%m-%dT%H:%M:%SZ')] DONE" | tee -a "$LOG"
cat "$OUT_DIR/eval_miou_summary.json" 2>/dev/null | tee -a "$LOG"
