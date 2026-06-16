#!/bin/bash
# ============================================================
# video-SALMONN2+ Temporal Grounding 데모 실행 (껐다 켜도 재사용)
#   사용: bash _tools/GDPO/run_demo.sh
#   (tmux 안에서 돌리면 SSH 끊겨도 유지 — 아래 주석 참고)
# ============================================================
set -e

# ── 1) conda 초기화 + 환경 ───────────────────────────────────────────────────
#   비대화형 셸(스크립트)에선 conda 가 PATH 에 있어도 `conda activate` 함수가
#   로드돼 있지 않아 "Run 'conda init'..." 에러가 난다. → conda.sh 를 항상 source.
#   base 경로는 conda info --base 로 자동 탐색, 실패 시 알려진 경로들 fallback.
_CONDA_BASE="$(conda info --base 2>/dev/null)"
for _sh in "${_CONDA_BASE:+$_CONDA_BASE/etc/profile.d/conda.sh}" \
           /workspace/home/miniconda3/etc/profile.d/conda.sh \
           /opt/conda/etc/profile.d/conda.sh \
           "$HOME/miniconda3/etc/profile.d/conda.sh"; do
    if [ -n "$_sh" ] && [ -f "$_sh" ]; then
        source "$_sh"
        break
    fi
done
conda activate hyj

cd /workspace/hyj/Team4
set -a && source paths.env && set +a

# ── 2) 모델 경로 (RL 결과를 데모에 띄움) ───────────────────────────────────────
#   MODEL_BASE : 가중치를 통째로 로드하는 곳(= 학습 때 --model_base). RL 어댑터가 얹힐
#                풀 모델(architectures=video_SALMONN2_plus)이어야 함. ★ RL 런과 동일해야 함.
#   MODEL_PATH : 그 base 위에 얹을 RL LoRA 어댑터(= GDPO --output_dir). adapter_config.json
#                있는 폴더여야 데모 로더가 경로 A 로 RL LoRA 를 base 위에 적용함.
#                여기를 SFT 머지본으로 두면(=base 와 동일) RL 없이 SFT 단독이 됨.
#
#   ↓↓↓ 보여줄 RL 런에 맞춰 이 두 줄만 바꾸세요 (v5b 관례 예시) ↓↓↓
DEMO_RL_BASE=/workspace/checkpoints/base/salmonn2p_7b_v5b_unav_v1
DEMO_RL_ADAPTER=/workspace/hyj/output/sft_7b_v5b_unav_v1_rl_rM_<TAG>_500   # ← <TAG> 채우기 (또는 .../checkpoint-XXXX)
#   ↑↑↑ 옛 puvalor SFT 단독을 보려면: BASE=ADAPTER=${BASE_MODEL} (=paths.env) 로 두면 됨 ↑↑↑

# ── 3) 실행 ───────────────────────────────────────────────────────────────────
#   DEMO_TTI_MODE / CUDA_VISIBLE_DEVICES 는 호출 시 덮어쓸 수 있음.
DEMO_MODEL_PATH="${DEMO_RL_ADAPTER}" \
DEMO_MODEL_BASE="${DEMO_RL_BASE}" \
DEMO_TTI_MODE="${DEMO_TTI_MODE:-off}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONUNBUFFERED=1 \
python _tools/GDPO/demo_app.py
