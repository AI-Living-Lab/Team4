#!/usr/bin/env bash
# JSON 파일 내 하드코딩된 경로를 paths.env 환경변수 기반으로 변환
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/paths.env"

OLD_PREFIX="/data0/aix23102"
NEW_PREFIX="${DATA_DIR}"   # /workspace/datasets

JSON_FILES=(
  "${SCRIPT_DIR}/data/unav100_train_multiseg_salmonn2plus.json"
  "${SCRIPT_DIR}/data/unav100_test_multiseg_sub80.json"
)

for f in "${JSON_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[SKIP] 파일 없음: $f"
    continue
  fi
  echo "[FIX]  $f"
  echo "       ${OLD_PREFIX} -> ${NEW_PREFIX}"
  sed -i "s|${OLD_PREFIX}|${NEW_PREFIX}|g" "$f"
done

echo "Done."
