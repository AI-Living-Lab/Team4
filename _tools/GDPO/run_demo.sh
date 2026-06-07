#!/bin/bash
# ============================================================
# video-SALMONN2+ Temporal Grounding 데모 실행 (껐다 켜도 재사용)
#   사용: bash _tools/GDPO/run_demo.sh
#   (tmux 안에서 돌리면 SSH 끊겨도 유지 — 아래 주석 참고)
# ============================================================
set -e

# ── 1) conda 초기화 + 환경 (새 컨테이너면 conda 가 PATH 에 없으므로 직접 source) ──
if ! command -v conda >/dev/null 2>&1; then
    source /workspace/home/miniconda3/etc/profile.d/conda.sh
fi
conda activate hyj

cd /workspace/hyj/Team4
set -a && source paths.env && set +a

# ── 2) 실행 (모델/데이터 경로는 paths.env 의 SFT_CKPT/BASE_MODEL) ──
#   DEMO_TTI_MODE / CUDA_VISIBLE_DEVICES 는 호출 시 덮어쓸 수 있음.
DEMO_MODEL_PATH="${SFT_CKPT}" \
DEMO_MODEL_BASE="${BASE_MODEL}" \
DEMO_TTI_MODE="${DEMO_TTI_MODE:-off}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONUNBUFFERED=1 \
python _tools/GDPO/demo_app.py
