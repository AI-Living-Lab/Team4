#!/bin/bash
# ============================================================
# SALMONN2+ 통합 Eval 런처 (eval.sh)
#
# 하나의 스크립트로 다음을 모두 지원:
#   - CHUNK=on|off       : 청크 추론(+자동 resume) / 단일 JSON 추론
#   - 모델 종류          : base+LoRA(no-merged) / full-merged(MERGED_MODEL=) / base(CKPT_STEP=base)
#   - MODE=infer|eval    : 추론+평가 / 이미 있는 결과파일만 평가(GPU 불필요)
#   - NATURAL=on|off     : 관대 파싱(HH:MM:SS/M:SS/소수초/second{}...) / 시간토큰만(기본)
#   - CoT 출력           : <answer>...</answer> 안만 자동 파싱 (eval_miou.py 내부 처리)
#
# 산출물(결과 폴더):
#   test_results_rank0.json   : 추론 결과 (MODE=infer 일 때)
#   pairwise_miou_summary.json  union_miou_summary.json  sample_miou_summary.json
#   inference.log (infer), eval_miou_progress.jsonl (chunk 진행 스냅샷)
#
# ── 사용 예 ──────────────────────────────────────────────
#  # 1) base+LoRA, 단일 JSON 추론+평가
#  bash eval.sh STAGE=gdpo CKPT_MODEL_ID=sft_7b_unav_v8_rl_v2_rMsep_clip_mu2 CKPT_STEP=1000 \
#       BASE_MODEL_ID=base/salmonn2p_7b_puvalor_off_v2 \
#       TEST_JSON=/workspace/data/test/unav100_v2_500.json GPUS=0
#
#  # 2) 청크 추론 (대용량 testset; data/test/<TESTSET>/chunk_*.json 필요)
#  bash eval.sh CHUNK=on STAGE=sft CKPT_MODEL_ID=salmonn2p_7b_unav_baseline CKPT_STEP=1500 \
#       BASE_MODEL_ID=base/video_salmonn2_plus_7B_time_tokens TESTSET=unav100 GPUS=0
#
#  # 3) full-merged 체크포인트 직접 평가 (LoRA 머지 X)
#  bash eval.sh MERGED_MODEL=/workspace/checkpoints/base/salmonn2p_7b_puvalor_off_v2 \
#       TEST_JSON=/workspace/data/test/unav100_v2_cot_50.json GPUS=0
#
#  # 4) 이미 추론된 결과만 재평가 (GPU 불필요). CoT/자연어면 NATURAL=on
#  bash eval.sh MODE=eval RESULTS=/workspace/outputs/.../test_results_rank0.json \
#       TEST_JSON=/workspace/data/test/unav100_v2_500.json NATURAL=on
# ============================================================
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 기본값 (CLI KEY=VALUE 로 override) ----
MODE=infer                 # infer | eval
CHUNK=off                  # on | off
CHUNK_SIZE=500
STAGE=sft                  # sft | gdpo
CKPT_MODEL_ID=
CKPT_STEP=                 # 빈값=latest checkpoint-*, base/no=base, 숫자/checkpoint-N
BASE_MODEL_ID=video_salmonn2_plus_7B_time_tokens
MERGED_MODEL=              # 설정 시 이 경로를 직접 로드(lora_ckpt=No), STAGE/CKPT_* 무시
TEST_JSON=                 # CHUNK=off / MODE=eval 의 GT(+추론 입력)
TESTSET=                   # CHUNK=on 의 chunks 디렉토리 이름 (data/test/<TESTSET>/)
TESTSET_TAG=               # 빈값이면 TEST_JSON 파일명 또는 TESTSET 에서 자동
RESULTS=                   # MODE=eval 에서 평가할 결과파일/폴더 (생략 시 OUT_DIR 추정)
OUT_DIR=                   # 빈값이면 규칙대로 자동 생성
LABEL=                     # summary 라벨 (빈값이면 CKPT_MODEL_ID/MERGED/base 명)
NATURAL=off                # on | off
GT_REF=off                 # on=GT 를 ref(토큰)에서 파싱(강제). 미지정 시 자동감지(embedded>ref>test_json)
PRED_PERCENT=off           # on=pred 구간을 %로 보고 duration 곱해 초 환산 (avicuna)
GPUS=0
CONFIG=config.yaml
TTI_TIME_FORMAT_CLI=

for arg in "$@"; do
    case "$arg" in
        MODE=*) MODE="${arg#*=}" ;;
        CHUNK=*) CHUNK="${arg#*=}" ;;
        CHUNK_SIZE=*) CHUNK_SIZE="${arg#*=}" ;;
        STAGE=*) STAGE="${arg#*=}" ;;
        CKPT_MODEL_ID=*) CKPT_MODEL_ID="${arg#*=}" ;;
        CKPT_STEP=*) CKPT_STEP="${arg#*=}" ;;
        BASE_MODEL_ID=*) BASE_MODEL_ID="${arg#*=}" ;;
        MERGED_MODEL=*) MERGED_MODEL="${arg#*=}" ;;
        TEST_JSON=*) TEST_JSON="${arg#*=}" ;;
        TESTSET=*) TESTSET="${arg#*=}" ;;
        TESTSET_TAG=*) TESTSET_TAG="${arg#*=}" ;;
        RESULTS=*) RESULTS="${arg#*=}" ;;
        OUT_DIR=*) OUT_DIR="${arg#*=}" ;;
        LABEL=*) LABEL="${arg#*=}" ;;
        NATURAL=*) NATURAL="${arg#*=}" ;;
        GT_REF=*) GT_REF="${arg#*=}" ;;
        PRED_PERCENT=*) PRED_PERCENT="${arg#*=}" ;;
        GPUS=*) GPUS="${arg#*=}" ;;
        CONFIG=*) CONFIG="${arg#*=}" ;;
        TTI_TIME_FORMAT=*) TTI_TIME_FORMAT_CLI="${arg#*=}" ;;
        *) echo "[에러] 지원하지 않는 인자: $arg"; exit 1 ;;
    esac
done

EVAL_PY="${SCRIPT_DIR}/eval_miou.py"
HELPER="${SCRIPT_DIR}/_chunk_helpers.py"
MAKETABLE="${SCRIPT_DIR}/maketable.py"

# eval 완료 후 카테고리(gdpo/base/sft/merged) 루트의 table.txt 를 자동 갱신.
#   결과경로에서 outputs 바로 아래 카테고리 폴더를 찾아 maketable.py 를 그 폴더에 실행한다.
#   예) .../outputs/gdpo/<run>/checkpoint-200 → .../outputs/gdpo/table.txt
#       .../outputs/base/MUSEG/Unav100        → .../outputs/base/table.txt
#   (maketable 은 stdlib 만 사용 → conda 불필요. 실패해도 eval 결과엔 영향 X.)
update_table() {
    local _dir="$1"
    [ -f "$_dir" ] && _dir="$(dirname "$_dir")"
    local _abs; _abs="$(cd "$_dir" 2>/dev/null && pwd)" || _abs="$_dir"
    local _acc="" _root="" _p _oldIFS="$IFS"
    IFS='/'
    for _p in $_abs; do
        [ -z "$_p" ] && continue
        _acc="$_acc/$_p"
        case "$_p" in gdpo|base|sft|merged) _root="$_acc"; break ;; esac
    done
    IFS="$_oldIFS"
    if [ -z "$_root" ]; then
        echo "[table] 카테고리(gdpo/base/sft/merged) 루트 못 찾음 → table 갱신 skip: $_abs"; return 0
    fi
    echo "[table] maketable → ${_root}/table.txt"
    python3 "$MAKETABLE" "$_root" || echo "[table] maketable 실패(무시): $_root"
}

EXTRA_FLAGS=""
[ "$NATURAL" = "on" ] && EXTRA_FLAGS="$EXTRA_FLAGS --natural"
[ "$GT_REF" = "on" ] && EXTRA_FLAGS="$EXTRA_FLAGS --gt_ref"
[ "$PRED_PERCENT" = "on" ] && EXTRA_FLAGS="$EXTRA_FLAGS --pred_percent"

# ---- config.yaml 로드 (flat KEY: VALUE) ----
CONFIG_PATH="${SCRIPT_DIR}/${CONFIG}"
[ -f "$CONFIG_PATH" ] || { echo "[에러] config 없음: $CONFIG_PATH"; exit 1; }
while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"; line="${line%%#*}"
    [[ -z "${line// }" ]] && continue
    key="${line%%:*}"; val="${line#*:}"
    key="$(echo "$key" | awk '{$1=$1;print}')"
    val="$(echo "$val" | awk '{$1=$1;print}')"
    val="${val#\"}"; val="${val%\"}"; val="${val#\'}"; val="${val%\'}"
    [[ -z "$key" ]] && continue
    eval "$key=\"\$val\""
done < "$CONFIG_PATH"
[ -n "$TTI_TIME_FORMAT_CLI" ] && TTI_TIME_FORMAT="$TTI_TIME_FORMAT_CLI"
MAX_TIME="${MAX_TIME:-9999.9}"

# ============================================================
# MODE=eval : 추론 생략, 이미 있는 결과만 평가 (GPU 불필요)
# ============================================================
if [ "$MODE" = "eval" ]; then
    EVAL_TARGET="${RESULTS:-$OUT_DIR}"
    [ -n "$EVAL_TARGET" ] || { echo "[에러] MODE=eval 은 RESULTS= 또는 OUT_DIR= 필요"; exit 1; }
    [ -e "$EVAL_TARGET" ] || { echo "[에러] 평가 대상 없음: $EVAL_TARGET"; exit 1; }
    [ -z "$TESTSET_TAG" ] && [ -n "$TEST_JSON" ] && TESTSET_TAG=$(basename "$TEST_JSON" .json)
    [ -z "$LABEL" ] && LABEL="${CKPT_MODEL_ID:-eval}"
    echo "[MODE=eval] target=$EVAL_TARGET  natural=$NATURAL"
    python3 "$EVAL_PY" "$EVAL_TARGET" \
        ${TEST_JSON:+--test_json "$TEST_JSON"} $EXTRA_FLAGS \
        --max_time "$MAX_TIME" --label "$LABEL" --testset "$TESTSET_TAG"
    update_table "$EVAL_TARGET"
    echo "[완료] eval-only"
    exit 0
fi

# ============================================================
# MODE=infer : 추론 + 평가
# ============================================================
# runpod 전용 setup.sh 는 있으면만 source (랩 서버엔 없음)
[ -f /workspace/setup.sh ] && source /workspace/setup.sh
source "$SCRIPT_DIR/../paths.env"
# conda 'activate' 셸 함수 로드 (비대화형 셸은 바이너리만 PATH 에 있어 init 필요)
if ! declare -F conda >/dev/null 2>&1; then
    _cbase="$(conda info --base 2>/dev/null)"
    for _c in "$_cbase" "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        [ -n "$_c" ] && [ -f "$_c/etc/profile.d/conda.sh" ] && source "$_c/etc/profile.d/conda.sh" && break
    done
fi
conda activate "${CONDA_ENV:-salmonn2p}"
# 저장 루트: OUT_ROOT > paths.env 의 EVAL_DIR > (최후) /workspace/outputs
export EVAL_DIR="${OUT_ROOT:-${EVAL_DIR:-/workspace/outputs}}"
export ARNOLD_WORKER_NUM=1 ARNOLD_ID=0 METIS_WORKER_0_HOST=localhost

NUM_GPUS=$(echo "$GPUS" | awk -F',' '{print NF}')
export CUDA_VISIBLE_DEVICES=$GPUS
export ARNOLD_WORKER_GPU=$NUM_GPUS
FIRST_GPU=$(echo "$GPUS" | awk -F',' '{print $1}')
MASTER_PORT=$((12900 + FIRST_GPU))

# ---- EVAL_TAG = fps<N>_<format> ----
FPS_INT=$(awk -v b="${BASE_INTERVAL:-0.2}" 'BEGIN { printf "%d", (1.0/b)+0.5 }')
case "${TTI_TIME_FORMAT:-off}" in
    off) FORMAT_TAG=off ;; natural_text) FORMAT_TAG=natural ;;
    special_token) FORMAT_TAG=tti ;; from_to) FORMAT_TAG=fromto ;;
    *) FORMAT_TAG="${TTI_TIME_FORMAT:-off}" ;;
esac
EVAL_TAG="fps${FPS_INT}_${FORMAT_TAG}"

BASE_CODE="${BASE_DIR}/video_SALMONN2_plus"
MODEL_BASE="${CKPT_DIR}/${BASE_MODEL_ID}"

# ---- 모델 해석: MERGED_MODEL > CKPT_STEP=base > base+LoRA ----
# IS_BASE_EVAL=true → LoRA 머지 불필요 (model_name 직접 로드, lora_ckpt=No)
IS_BASE_EVAL=false
if [ -n "$MERGED_MODEL" ]; then
    [ -f "$MERGED_MODEL/config.json" ] || { echo "[에러] merged 모델 없음: $MERGED_MODEL"; exit 1; }
    LORA_CKPT="No"; IS_BASE_EVAL=true
    MODEL_LOAD="$MERGED_MODEL"
    OUT_BRANCH="merged/$(basename "$MERGED_MODEL")"
    [ -z "$LABEL" ] && LABEL="$(basename "$MERGED_MODEL")"
else
    STAGE=$(echo "$STAGE" | tr '[:upper:]' '[:lower:]')
    [ "$STAGE" = "sft" ] || [ "$STAGE" = "gdpo" ] || { echo "[에러] STAGE는 sft|gdpo"; exit 1; }
    [ -n "$CKPT_MODEL_ID" ] || { echo "[에러] CKPT_MODEL_ID 필요 (또는 MERGED_MODEL=)"; exit 1; }
    MODEL_DIR="${CKPT_DIR}/${STAGE}/${CKPT_MODEL_ID}"
    CKPT_STEP_LOWER=$(echo "$CKPT_STEP" | tr '[:upper:]' '[:lower:]')
    _root_ckpt_folder() {
        if [ -f "$MODEL_DIR/latest" ]; then
            local _s; _s=$(sed -n 's/.*global_step//p' "$MODEL_DIR/latest")
            echo "checkpoint-${_s:-root}"
        else echo "root"; fi
    }
    if [ -z "$CKPT_STEP" ]; then
        LORA_CKPT=$(ls -d "$MODEL_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1) || true
        if [ -n "$LORA_CKPT" ]; then CKPT_FOLDER=$(basename "$LORA_CKPT")
        elif [ -f "$MODEL_DIR/adapter_config.json" ]; then LORA_CKPT="$MODEL_DIR"; CKPT_FOLDER=$(_root_ckpt_folder)
        else echo "[에러] $MODEL_DIR 에 checkpoint-* / adapter_config.json 없음"; exit 1; fi
    elif [ "$CKPT_STEP_LOWER" = "base" ] || [ "$CKPT_STEP_LOWER" = "no" ]; then
        LORA_CKPT="No"; CKPT_FOLDER="base"; IS_BASE_EVAL=true
    elif [ "$CKPT_STEP_LOWER" = "root" ] || [ "$CKPT_STEP_LOWER" = "self" ]; then
        [ -f "$MODEL_DIR/adapter_config.json" ] || { echo "[에러] $MODEL_DIR adapter 없음"; exit 1; }
        LORA_CKPT="$MODEL_DIR"; CKPT_FOLDER=$(_root_ckpt_folder)
    else
        if [[ "$CKPT_STEP" == checkpoint-* ]]; then CKPT_FOLDER="$CKPT_STEP"; else CKPT_FOLDER="checkpoint-${CKPT_STEP}"; fi
        LORA_CKPT="${MODEL_DIR}/${CKPT_FOLDER}"
        [ -d "$LORA_CKPT" ] || { echo "[에러] 체크포인트 없음: $LORA_CKPT"; exit 1; }
    fi
    MODEL_LOAD="$MODEL_BASE"
    if [ "$IS_BASE_EVAL" = "true" ]; then OUT_BRANCH="base/${BASE_MODEL_ID#base/}"
    else OUT_BRANCH="${STAGE}/${CKPT_MODEL_ID}/${CKPT_FOLDER}"; fi
    [ -z "$LABEL" ] && LABEL="${CKPT_MODEL_ID:-${BASE_MODEL_ID#base/}}"
fi

# ---- testset 소스 / 태그 / OUT_DIR ----
if [ "$CHUNK" = "on" ]; then
    [ -n "$TESTSET" ] || { echo "[에러] CHUNK=on 은 TESTSET= 필요 (data/test/<TESTSET>/)"; exit 1; }
    CHUNKS_DIR="${TEST_DIR}/${TESTSET}"
    [ -d "$CHUNKS_DIR" ] || { echo "[에러] chunks 디렉토리 없음: $CHUNKS_DIR (먼저 _chunk_helpers.py split)"; exit 1; }
    [ -z "$TESTSET_TAG" ] && TESTSET_TAG="$TESTSET"
    # _full.json (eval GT 용; 없으면 생성)
    TEST_JSON="${CHUNKS_DIR}/_full.json"
    python3 "$HELPER" build_full --chunks_dir "$CHUNKS_DIR" >/dev/null
    mapfile -t CHUNK_FILES < <(ls "$CHUNKS_DIR"/chunk_*.json 2>/dev/null | sort)
    N_CHUNKS=${#CHUNK_FILES[@]}
    [ "$N_CHUNKS" -gt 0 ] || { echo "[에러] $CHUNKS_DIR 에 chunk_*.json 없음"; exit 1; }
else
    [ -f "$TEST_JSON" ] || { echo "[에러] TEST_JSON 없음: $TEST_JSON"; exit 1; }
    [ -z "$TESTSET_TAG" ] && TESTSET_TAG=$(basename "$TEST_JSON" .json)
fi

[ -z "$OUT_DIR" ] && OUT_DIR="${EVAL_DIR}/${OUT_BRANCH}/${EVAL_TAG}/${TESTSET_TAG}"
mkdir -p "$OUT_DIR"
MASTER_RESULT="$OUT_DIR/test_results_rank0.json"
INFER_LOG="$OUT_DIR/inference.log"
PROGRESS_LOG="$OUT_DIR/eval_miou_progress.jsonl"

cat <<EOF
==================================================
  통합 eval (MODE=infer)
  CHUNK          : $CHUNK   $( [ "$CHUNK" = on ] && echo "($N_CHUNKS chunks)" )
  MODEL_LOAD     : $MODEL_LOAD
  LORA_CKPT      : $LORA_CKPT   (base_eval=$IS_BASE_EVAL)
  LABEL          : $LABEL
  TEST           : ${TEST_JSON}  (tag=$TESTSET_TAG)
  EVAL_TAG       : $EVAL_TAG  (BASE_INTERVAL=$BASE_INTERVAL, TTI=${TTI_TIME_FORMAT:-off}, NATURAL=$NATURAL)
  GPUS           : $GPUS (count=$NUM_GPUS)
  OUT_DIR        : $OUT_DIR
==================================================
EOF

# ---- 추론 1회 호출 (입력 json, 모델, lora 를 인자로) ----
# 사용: run_inference <dataset_json> <model_name_or_path> <lora_ckpt> <output_dir>
run_inference() {
    local _ds="$1" _model="$2" _lora="$3" _outdir="$4"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        qwenvl/train/train_qwen.py \
        --model_base "$MODEL_BASE" \
        --run_test True --pred_rank 0 \
        --deepspeed "$DEEPSPEED_CONFIG" \
        --model_name_or_path "$_model" \
        --dataset_use "$_ds" \
        --bf16 \
        --output_dir "$_outdir" \
        --num_train_epochs "$NUM_TRAIN_EPOCHS" \
        --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
        --max_pixels "$MAX_PIXELS" --min_pixels "$MIN_PIXELS" \
        --video_max_frame_pixels "$VIDEO_MAX_FRAME_PIXELS" \
        --video_min_frame_pixels "$VIDEO_MIN_FRAME_PIXELS" \
        --eval_strategy "$EVAL_STRATEGY" --save_strategy "$SAVE_STRATEGY" \
        --learning_rate "$LEARNING_RATE" \
        --model_max_length "$MODEL_MAX_LENGTH" \
        --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
        --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
        --run_name "." --report_to "$REPORT_TO" \
        --video_min_frames "$VIDEO_MIN_FRAMES" --video_max_frames "$VIDEO_MAX_FRAMES" \
        --base_interval "$BASE_INTERVAL" \
        --lora_ckpt "$_lora" \
        --no_audio "$NO_AUDIO" \
        --tti_time_format "${TTI_TIME_FORMAT:-off}" \
        2>&1 | tee -a "$INFER_LOG"
}

# 평가 1회 호출 → 3개 summary 갱신
run_eval() {
    local _quiet="$1"
    python3 "$EVAL_PY" "$MASTER_RESULT" --test_json "$TEST_JSON" $EXTRA_FLAGS \
        --max_time "$MAX_TIME" --out_dir "$OUT_DIR" \
        --label "$LABEL" --testset "$TESTSET_TAG" \
        --progress_log "$PROGRESS_LOG" $_quiet
}

cd "$BASE_CODE"

if [ "$CHUNK" != "on" ]; then
    # ---------- 단일 JSON ----------
    echo "[1/2] inference (tti=${TTI_TIME_FORMAT:-off})" | tee "$INFER_LOG"
    run_inference "$TEST_JSON" "$MODEL_LOAD" "$LORA_CKPT" "$OUT_DIR"
    [ -f "$MASTER_RESULT" ] || { echo "[에러] 결과 없음: $MASTER_RESULT"; exit 1; }
    rm -rf "$OUT_DIR/generation_0" 2>/dev/null || true
    echo "[2/2] eval → 3 summaries"
    run_eval ""
    update_table "$OUT_DIR"
    echo "[완료] $OUT_DIR"; exit 0
fi

# ---------- 청크 (자동 resume + LoRA merge 캐시 재사용) ----------
CHUNK_IDX_FILE="$OUT_DIR/.chunk_idx"
CHUNK_WORKDIR="$OUT_DIR/.chunk_workdir"
MERGED_CACHE="$OUT_DIR/.merged_model"

RESUME_OUT=$(python3 "$HELPER" resume_offset --master "$MASTER_RESULT" \
    --chunks_dir "$CHUNKS_DIR" --chunk_idx_file "$CHUNK_IDX_FILE")
read START_CHUNK N_MASTER EXPECTED_N <<<"$RESUME_OUT"
if [ "$N_MASTER" -gt "$EXPECTED_N" ]; then
    python3 "$HELPER" truncate_master --master "$MASTER_RESULT" --keep "$EXPECTED_N" >/dev/null
    echo "[CLEAN] truncated master $N_MASTER -> $EXPECTED_N"
fi

SUCCESS=0
cleanup_all() {
    rm -rf "$CHUNK_WORKDIR" 2>/dev/null || true
    if [ "$SUCCESS" = "1" ]; then
        rm -rf "$MERGED_CACHE" 2>/dev/null || true
        rm -f "$CHUNK_IDX_FILE" 2>/dev/null || true
    fi
}
trap cleanup_all EXIT

if [ "$START_CHUNK" -ge "$N_CHUNKS" ]; then
    echo "[SKIP] 모든 $N_CHUNKS chunk 완료됨; 최종 평가만"
    run_eval ""; SUCCESS=1; update_table "$OUT_DIR"; echo "[완료] $OUT_DIR"; exit 0
fi

for ((i=START_CHUNK; i<N_CHUNKS; i++)); do
    chunk_json="${CHUNK_FILES[$i]}"
    chunk_id=$(basename "$chunk_json" .json | sed 's/^chunk_//')
    if [ -d "$MERGED_CACHE" ] && [ "$IS_BASE_EVAL" != "true" ]; then
        _model="$MERGED_CACHE"; _lora="No"
    else
        _model="$MODEL_LOAD"; _lora="$LORA_CKPT"
    fi
    rm -rf "$CHUNK_WORKDIR"; mkdir -p "$CHUNK_WORKDIR"
    echo "=== [CHUNK $((i+1))/$N_CHUNKS] id=$chunk_id $(date -Iseconds) ===" | tee -a "$INFER_LOG"
    run_inference "$chunk_json" "$_model" "$_lora" "$CHUNK_WORKDIR"

    chunk_result="$CHUNK_WORKDIR/test_results_rank0.json"
    [ -f "$chunk_result" ] || { echo "[에러] chunk $chunk_id 결과 없음"; exit 1; }
    # 첫 LoRA merge 산출물을 캐시로 보존(이후 chunk 재사용)
    if [ "$IS_BASE_EVAL" != "true" ] && [ ! -d "$MERGED_CACHE" ] && [ -d "$CHUNK_WORKDIR/generation_0" ]; then
        mv "$CHUNK_WORKDIR/generation_0" "$MERGED_CACHE"; echo "[MERGE] cached -> $MERGED_CACHE"
    fi
    AGG_OUT=$(python3 "$HELPER" append --master "$MASTER_RESULT" --chunk_results "$chunk_result")
    read N_AFTER N_ADDED <<<"$AGG_OUT"
    echo "$((i+1))" > "$CHUNK_IDX_FILE.tmp" && mv "$CHUNK_IDX_FILE.tmp" "$CHUNK_IDX_FILE"
    run_eval "--quiet"
    echo "[CHUNK $((i+1))/$N_CHUNKS] master_n=$N_AFTER added=$N_ADDED $(date -Iseconds)"
done

echo ""; echo "=== FINAL EVAL $(date -Iseconds) ==="
run_eval ""
SUCCESS=1
update_table "$OUT_DIR"
echo "[완료] $OUT_DIR"
