#!/usr/bin/env bash
# run_all.sh
# ----------
# TTI 검증 전체를 순서대로 실행. 하나라도 실패하면 즉시 종료 (exit 1).
#
# 사용:
#   bash _tools/tti/run_all.sh <model_path>
#
# 예:
#   bash _tools/tti/run_all.sh /workspace/checkpoints/base/video_salmonn2_plus_7B_time_tokens
#
# 주의: 실행 전 conda env 를 먼저 활성화해야 한다.
#   source /workspace/setup.sh && conda activate salmonn2plus

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: bash $0 <model_path>"
  exit 2
fi

MODEL_PATH="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {
  local name="$1"; shift
  echo ""
  echo "================================================================"
  echo "  $name"
  echo "================================================================"
  python "$@"
}

run "01  tokenizer + config IDs"      "$HERE/01_check_tokenizer_ids.py" --model_path "$MODEL_PATH"
run "02  dataset interleave"          "$HERE/02_check_dataset_interleave.py"
run "03  rope with time tokens"       "$HERE/03_check_rope_with_time_tokens.py"
run "04  rope regression (off path)"  "$HERE/04_check_rope_regression.py"
run "05  modeling delegate shim"      "$HERE/05_check_modeling_delegate.py"
run "06  rope natural_text interleave" "$HERE/06_check_natural_text_interleave.py"
run "07  off mode (no markers)"        "$HERE/07_check_off_mode.py"

echo ""
echo "================================================================"
echo "  ALL 7 CHECKS PASSED"
echo "================================================================"
