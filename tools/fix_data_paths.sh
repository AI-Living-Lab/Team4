#!/usr/bin/env bash
# data/{train,test} 내 모든 JSON 의 하드코딩된 원격 경로를 paths.env 기반 로컬 경로로 변환
# 예) "/workspace/datasets/unav_100/videos/X.mp4" -> "${DATA_DIR}/videos/X.mp4"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${PROJECT_DIR}/paths.env"

OLD_PREFIX="/workspace/datasets/unav_100"
NEW_PREFIX="${DATA_DIR}"   # Lab: /data0/aix23102/unav_100

DIRS=(
  "${TRAIN_DIR}"
  "${TEST_DIR}"
)

for d in "${DIRS[@]}"; do
  if [[ ! -d "$d" ]]; then
    echo "[SKIP] 디렉터리 없음: $d"
    continue
  fi
  for f in "$d"/*.json; do
    [[ -e "$f" ]] || continue
    n=$(grep -c "${OLD_PREFIX}" "$f" || true)
    if [[ "$n" -eq 0 ]]; then
      echo "[SKIP] 대상 경로 없음: $f"
      continue
    fi
    echo "[FIX]  $f  (${n} matches)"
    echo "       ${OLD_PREFIX} -> ${NEW_PREFIX}"
    sed -i "s|${OLD_PREFIX}|${NEW_PREFIX}|g" "$f"
  done
done

echo "Done."
