#!/bin/bash
# ============================================================
# eval_2gpu_merged.sh
#   여러 GPU split 을 기존 eval.sh 로 병렬 추론한 뒤, 결과를 하나로 합쳐
#   단일 평가(summary 3종)를 낸다. → "GPU 별로 나뉜 결과를 단일 결과로".
#
#   ⚠️ 기존 스크립트(eval.sh / eval_miou.py / _chunk_helpers.py / config.yaml)는
#      읽기만 하고 절대 수정하지 않는다. 새 결과는 <FINAL_TAG> 폴더에만 생성.
#
#   동작:
#     1) SPLITS 의 각 항목을 짝지은 GPUS 에서 eval.sh CHUNK=on 으로 병렬 추론
#        (각 split 은 청크 자동 누적 + resume → split 당 test_results_rank0.json)
#     2) split 결과들을 concat (각 샘플이 자기 GT(ref)를 들고 있어 병합 안전)
#     3) 합친 결과에 eval_miou.py 1회 → <FINAL_TAG>/{pairwise,union,sample}_miou_summary.json
#
#   사용 예 (charades, 2GPU):
#     bash eval_2gpu_merged.sh \
#       STAGE=gdpo CKPT_MODEL_ID=sft_7b_unav_v8_rl_rMsep2fp_lr_clip_mu2_mlp_headoff_unpu \
#       CKPT_STEP=1000 BASE_MODEL_ID=base/salmonn2p_7b_unav_v8 \
#       SPLITS="charades_rlp_audio_s0 charades_rlp_audio_s1" GPUS="0 1" \
#       FINAL_TAG=charades_rlp_audio \
#       FULL_TEST_JSON=$TEST_DIR/charades_rlp_audio.json
# ============================================================
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 기본값 ----
STAGE=gdpo
CKPT_MODEL_ID=
CKPT_STEP=
BASE_MODEL_ID=base/salmonn2p_7b_unav_v8
SPLITS=                 # 공백구분 TESTSET 디렉토리명들 (data/test/<...>/)
GPUS=                   # 공백구분 GPU 번호들 (SPLITS 와 개수 동일)
SEQUENTIAL=off          # on=split 들을 순차 실행(GPU 1개로 여러 split 처리 시). off=병렬
FINAL_TAG=              # 최종 병합 결과 폴더/태그 (suffix 없는 이름)
FULL_TEST_JSON=         # (선택) 최종 평가 GT. 생략 시 ref 자동감지
LABEL=                  # (선택) summary label. 기본=CKPT_MODEL_ID
TTI_TIME_FORMAT=special_token
NATURAL=off
CONFIG=config.yaml

for arg in "$@"; do case "$arg" in
    STAGE=*) STAGE="${arg#*=}";;
    CKPT_MODEL_ID=*) CKPT_MODEL_ID="${arg#*=}";;
    CKPT_STEP=*) CKPT_STEP="${arg#*=}";;
    BASE_MODEL_ID=*) BASE_MODEL_ID="${arg#*=}";;
    SPLITS=*) SPLITS="${arg#*=}";;
    GPUS=*) GPUS="${arg#*=}";;
    SEQUENTIAL=*) SEQUENTIAL="${arg#*=}";;
    FINAL_TAG=*) FINAL_TAG="${arg#*=}";;
    FULL_TEST_JSON=*) FULL_TEST_JSON="${arg#*=}";;
    LABEL=*) LABEL="${arg#*=}";;
    TTI_TIME_FORMAT=*) TTI_TIME_FORMAT="${arg#*=}";;
    NATURAL=*) NATURAL="${arg#*=}";;
    CONFIG=*) CONFIG="${arg#*=}";;
    *) echo "[에러] 모르는 인자: $arg"; exit 1;;
esac; done

[ -n "$CKPT_MODEL_ID" ] || { echo "[에러] CKPT_MODEL_ID 필요"; exit 1; }
[ -n "$CKPT_STEP" ]     || { echo "[에러] CKPT_STEP 필요"; exit 1; }
[ -n "$SPLITS" ]        || { echo "[에러] SPLITS 필요"; exit 1; }
[ -n "$GPUS" ]          || { echo "[에러] GPUS 필요"; exit 1; }
[ -n "$FINAL_TAG" ]     || { echo "[에러] FINAL_TAG 필요"; exit 1; }

# ---- paths.env (EVAL_DIR/TEST_DIR 등) ----
source "$SCRIPT_DIR/../paths.env"
EVAL_DIR="${OUT_ROOT:-${EVAL_DIR}}"

# ---- EVAL_TAG = fps<N>_<format> (eval.sh 와 동일 규칙) ----
_cfgval() { grep -E "^$1:" "$SCRIPT_DIR/$CONFIG" | head -1 | sed 's/^[^:]*: *//' | tr -d '"'\'' '; }
BASE_INTERVAL="$(_cfgval BASE_INTERVAL)"; BASE_INTERVAL="${BASE_INTERVAL:-0.2}"
MAX_TIME="$(_cfgval MAX_TIME)"; MAX_TIME="${MAX_TIME:-9999.9}"
FPS_INT=$(awk -v b="$BASE_INTERVAL" 'BEGIN{printf "%d",(1.0/b)+0.5}')
case "$TTI_TIME_FORMAT" in
    off) FT=off;; natural_text) FT=natural;; special_token) FT=tti;; from_to) FT=fromto;; *) FT="$TTI_TIME_FORMAT";;
esac
EVAL_TAG="fps${FPS_INT}_${FT}"

if [[ "$CKPT_STEP" == checkpoint-* ]]; then CKPT_FOLDER="$CKPT_STEP"; else CKPT_FOLDER="checkpoint-${CKPT_STEP}"; fi
BRANCH="${STAGE}/${CKPT_MODEL_ID}/${CKPT_FOLDER}"
[ -z "$LABEL" ] && LABEL="$CKPT_MODEL_ID"

read -ra SPLIT_ARR <<< "$SPLITS"
read -ra GPU_ARR   <<< "$GPUS"
[ "${#SPLIT_ARR[@]}" -eq "${#GPU_ARR[@]}" ] || { echo "[에러] SPLITS(${#SPLIT_ARR[@]}) 와 GPUS(${#GPU_ARR[@]}) 개수 불일치"; exit 1; }

cat <<EOF
==================================================
  eval_2gpu_merged
  MODEL          : ${STAGE}/${CKPT_MODEL_ID}/${CKPT_FOLDER}
  BASE           : ${BASE_MODEL_ID}
  EVAL_TAG       : ${EVAL_TAG}
  SPLITS         : ${SPLITS}
  GPUS           : ${GPUS}
  FINAL_TAG      : ${FINAL_TAG}
  FULL_TEST_JSON : ${FULL_TEST_JSON:-(ref 자동감지)}
  EVAL_DIR       : ${EVAL_DIR}
==================================================
EOF

# ============================================================
# [1/3] split 별 병렬 추론 (각 split = 1 GPU, eval.sh CHUNK=on)
# ============================================================
echo "=== [1/3] split 추론 시작 (SEQUENTIAL=$SEQUENTIAL) $(date -Iseconds) ==="
FAIL=0
if [ "$SEQUENTIAL" = "on" ]; then
    # 순차: split 을 하나씩 끝내고 다음으로 (GPU 1개로 여러 split 처리 시 OOM 방지)
    for idx in "${!SPLIT_ARR[@]}"; do
        sp="${SPLIT_ARR[$idx]}"; g="${GPU_ARR[$idx]}"
        splog="$SCRIPT_DIR/_2gpu_${FINAL_TAG}_${sp}_gpu${g}.log"
        echo "  [seq] split=$sp GPU=$g → $splog  $(date -Iseconds)"
        if ! bash "$SCRIPT_DIR/eval.sh" \
                CHUNK=on STAGE="$STAGE" \
                CKPT_MODEL_ID="$CKPT_MODEL_ID" CKPT_STEP="$CKPT_STEP" \
                BASE_MODEL_ID="$BASE_MODEL_ID" TESTSET="$sp" GPUS="$g" \
                TTI_TIME_FORMAT="$TTI_TIME_FORMAT" NATURAL="$NATURAL" CONFIG="$CONFIG" \
                > "$splog" 2>&1; then
            echo "[에러] split '$sp' (GPU $g) 실패 → 로그: $splog"; FAIL=1; break
        fi
    done
else
    # 병렬: split 마다 다른 GPU 에서 동시에
    PIDS=(); LOGS=()
    for idx in "${!SPLIT_ARR[@]}"; do
        sp="${SPLIT_ARR[$idx]}"; g="${GPU_ARR[$idx]}"
        splog="$SCRIPT_DIR/_2gpu_${FINAL_TAG}_${sp}_gpu${g}.log"
        LOGS+=("$splog")
        echo "  - split=$sp GPU=$g → $splog"
        bash "$SCRIPT_DIR/eval.sh" \
            CHUNK=on STAGE="$STAGE" \
            CKPT_MODEL_ID="$CKPT_MODEL_ID" CKPT_STEP="$CKPT_STEP" \
            BASE_MODEL_ID="$BASE_MODEL_ID" TESTSET="$sp" GPUS="$g" \
            TTI_TIME_FORMAT="$TTI_TIME_FORMAT" NATURAL="$NATURAL" CONFIG="$CONFIG" \
            > "$splog" 2>&1 &
        PIDS+=($!)
    done
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            echo "[에러] split '${SPLIT_ARR[$i]}' (GPU ${GPU_ARR[$i]}) 추론 실패 → 로그: ${LOGS[$i]}"
            FAIL=1
        fi
    done
fi
[ "$FAIL" = "0" ] || { echo "[중단] 일부 split 실패. 고친 뒤 재실행하면 CHUNK resume 됨."; exit 1; }
echo "=== split 추론 모두 완료 $(date -Iseconds) ==="

# ============================================================
# [2/3] split 결과 concat → 단일 master
# ============================================================
echo "=== [2/3] 결과 병합 ==="
FINAL_DIR="${EVAL_DIR}/${BRANCH}/${EVAL_TAG}/${FINAL_TAG}"
mkdir -p "$FINAL_DIR"
MERGED="$FINAL_DIR/test_results_rank0.json"

SPLIT_RESULTS=()
for sp in "${SPLIT_ARR[@]}"; do
    r="${EVAL_DIR}/${BRANCH}/${EVAL_TAG}/${sp}/test_results_rank0.json"
    [ -f "$r" ] || { echo "[에러] split 결과 없음: $r"; exit 1; }
    SPLIT_RESULTS+=("$r")
done

python3 - "$MERGED" "${SPLIT_RESULTS[@]}" <<'PY'
import json, sys
out, parts = sys.argv[1], sys.argv[2:]
merged = []
for p in parts:
    d = json.load(open(p))
    merged.extend(d if isinstance(d, list) else [d])
with open(out, "w") as f:
    json.dump(merged, f, ensure_ascii=False)
print(f"[merge] {len(parts)} split → {len(merged)} samples → {out}")
PY

# ============================================================
# [3/3] 합친 결과 단일 평가 → summary 3종
# ============================================================
echo "=== [3/3] 단일 평가 ==="
if ! declare -F conda >/dev/null 2>&1; then
    _cb="$(conda info --base 2>/dev/null)"
    for _c in "$_cb" "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        [ -n "$_c" ] && [ -f "$_c/etc/profile.d/conda.sh" ] && source "$_c/etc/profile.d/conda.sh" && break
    done
fi
conda activate "${CONDA_ENV:-salmonn2p}" 2>/dev/null || true

EXTRA=""; [ "$NATURAL" = "on" ] && EXTRA="--natural"
python3 "$SCRIPT_DIR/eval_miou.py" "$MERGED" \
    ${FULL_TEST_JSON:+--test_json "$FULL_TEST_JSON"} $EXTRA \
    --max_time "$MAX_TIME" --out_dir "$FINAL_DIR" \
    --label "$LABEL" --testset "$FINAL_TAG"

echo ""
echo "[완료] 단일 결과 → $FINAL_DIR"
ls "$FINAL_DIR"/*_miou_summary.json
