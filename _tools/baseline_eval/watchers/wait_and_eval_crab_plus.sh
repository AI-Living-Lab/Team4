#!/bin/bash
# Crab+ inference PID 종료 → 자동 eval
# usage: bash wait_and_eval_crab_plus.sh <PID>
set -e
PID="${1:-0}"
OUT_DIR=/workspace/jsy/outputs/base/CrabPlus/Unav100QA
RESULTS="$OUT_DIR/test_results_rank0.json"
TEST_JSON=/workspace/jsy/Crab_Plus/data_local/AVUIE_2/unav/test.json
EVAL_SCRIPT=/workspace/jsy/scripts/eval_miou_multiseg_crab_plus.py
LOG="$OUT_DIR/watcher.log"
PY=/workspace/miniconda3/envs/chronusomni/bin/python

mkdir -p "$OUT_DIR"
echo "[watcher $(date -u +'%Y-%m-%dT%H:%M:%SZ')] polling PID=$PID" | tee -a "$LOG"

while kill -0 "$PID" 2>/dev/null; do
    sleep 30
done

echo "[watcher $(date -u +'%Y-%m-%dT%H:%M:%SZ')] PID $PID exited" | tee -a "$LOG"

if [ ! -f "$RESULTS" ]; then
    echo "[watcher ERROR] results missing: $RESULTS" | tee -a "$LOG"
    exit 1
fi

N=$("$PY" -c "import json;print(len(json.load(open('$RESULTS'))))")
echo "[watcher] results n=$N" | tee -a "$LOG"

echo "[watcher $(date -u +'%Y-%m-%dT%H:%M:%SZ')] running eval" | tee -a "$LOG"
"$PY" "$EVAL_SCRIPT" \
    --results "$RESULTS" \
    --test_json "$TEST_JSON" \
    --max_time 60 \
    --out_dir "$OUT_DIR" 2>&1 | tee -a "$LOG"

echo "[watcher $(date -u +'%Y-%m-%dT%H:%M:%SZ')] DONE" | tee -a "$LOG"
cat "$OUT_DIR/eval_miou_summary.json" | tee -a "$LOG"
