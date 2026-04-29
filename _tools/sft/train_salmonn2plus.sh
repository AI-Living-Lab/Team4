#!/bin/bash
# ============================================================
# SALMONN2+ SFT 학습 런처
#   - 데이터: UnAV-100 multi-segment QA (기본값)
#   - 하이퍼파라미터: config.yaml 에서 관리 (CONFIG=... 으로 교체 가능)
#
# 사용법 (모든 인자는 선택, KEY=VALUE 형식):
#   bash train_salmonn2plus.sh \
#       STAGE=sft \
#       MODEL_ID=salmonn2p_unav_baseline \
#       TRAINSET_FILE=unav100_train_multiseg_salmonn2plus.json \
#       BASE_MODEL_ID=video_salmonn2_plus_7B_time_tokens \
#       GPUS=0,1,2 \
#       CONFIG=config.yaml
#
# 지원 인자:
#   STAGE         : sft | gdpo  (대소문자 무관, 기본값 sft)
#                   -> 체크포인트 저장 위치의 하위 폴더를 결정
#   MODEL_ID      : 학습 산출물 식별자.
#                   네이밍 규칙: {model}_{size}_{dataset}_{설정tag}
#                   예) salmonn2p_7b_unav_baseline, salmonn2p_7b_unav_tti
#   TRAINSET_FILE  : ${JSON_DIR} 아래 학습 json 파일명
#   BASE_MODEL_ID : ${CKPT_DIR} 아래 베이스 모델 폴더명
#   GPUS          : 사용할 GPU id 목록 (콤마 구분, 예: "0" / "0,1,2,3")
#   CONFIG        : 하이퍼파라미터 yaml 파일명 (스크립트와 같은 폴더 기준)
#
# 체크포인트 저장 구조:
#   Team4/
#   └── checkpoints/              <- ${CKPT_DIR}
#       ├── sft/
#       │   ├── salmonn2p_7b_unav_baseline/
#       │   │   ├── checkpoint-01000/
#       │   │   ├── checkpoint-02000/
#       │   │   └── ...
#       │   └── salmonn2p_7b_unav_tti/
#       └── gdpo/
#           └── salmonn2p_7b_unav_baseline/
#               ├── checkpoint-00500/
#               └── checkpoint-01000/
#
#   즉, MODEL_DIR = ${CKPT_DIR}/${STAGE(소문자)}/${MODEL_ID}
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /workspace/setup.sh
conda activate salmonn2plus

# ---- 기본값 ----
STAGE=sft
MODEL_ID=salmonn2p_unav_baseline
TRAINSET_FILE=unav100_train_multiseg_salmonn2plus.json
BASE_MODEL_ID=video_salmonn2_plus_7B_time_tokens
GPUS=0
CONFIG=config.yaml

# ---- KEY=VALUE 인자 파싱 ----
for arg in "$@"; do
    case "$arg" in
        STAGE=*)            STAGE="${arg#*=}" ;;
        MODEL_ID=*)         MODEL_ID="${arg#*=}" ;;
        TRAINSET_FILE=*)     TRAINSET_FILE="${arg#*=}" ;;
        BASE_MODEL_ID=*)    BASE_MODEL_ID="${arg#*=}" ;;
        GPUS=*)             GPUS="${arg#*=}" ;;
        CONFIG=*)           CONFIG="${arg#*=}" ;;
        *)
            echo "[에러] 지원하지 않는 인자: $arg"
            echo "지원 인자: STAGE, MODEL_ID, TRAINSET_FILE, BASE_MODEL_ID, GPUS, CONFIG"
            exit 1
            ;;
    esac
done

# ---- STAGE 검증 (소문자로 정규화 후 sft/gdpo만 허용) ----
STAGE=$(echo "$STAGE" | tr '[:upper:]' '[:lower:]')
if [ "$STAGE" != "sft" ] && [ "$STAGE" != "gdpo" ]; then
    echo "[에러] STAGE는 'sft' 또는 'gdpo' 여야 합니다 (받은 값: $STAGE)"
    exit 1
fi

# ---- config.yaml 전체 경로 조립 (스크립트 폴더 기준) ----
CONFIG_DIR="${SCRIPT_DIR}/${CONFIG}"

# ---- YAML 설정 로드 (flat KEY: VALUE 형식만 지원) ----
if [ ! -f "$CONFIG_DIR" ]; then
    echo "[에러] config 파일을 찾을 수 없음: $CONFIG_DIR"
    exit 1
fi

while IFS= read -r line || [ -n "$line" ]; do
    # Windows CR, 인라인 '#' 주석 제거
    line="${line%$'\r'}"
    line="${line%%#*}"
    # 빈 줄 스킵
    [[ -z "${line// }" ]] && continue
    # 첫 ':' 기준으로 key/value 분리
    key="${line%%:*}"
    val="${line#*:}"
    # 양쪽 공백 제거
    key="$(echo "$key" | awk '{$1=$1;print}')"
    val="$(echo "$val" | awk '{$1=$1;print}')"
    # 값 감싸는 따옴표 제거
    val="${val#\"}"; val="${val%\"}"
    val="${val#\'}"; val="${val%\'}"
    [[ -z "$key" ]] && continue
    eval "$key=\"\$val\""
done < "$CONFIG_DIR"

# ---- GPU 개수 자동 계산 ----
NUM_GPUS=$(echo "$GPUS" | awk -F',' '{print NF}')

export CUDA_VISIBLE_DEVICES=$GPUS
export ARNOLD_WORKER_GPU=$NUM_GPUS
export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

cd "${BASE_DIR}/video_SALMONN2_plus"

# ---- 경로 조립 ----
MODEL=${CKPT_DIR}/${BASE_MODEL_ID}
MODEL_BASE=${CKPT_DIR}/${BASE_MODEL_ID}
DATASET=${JSON_DIR}/${TRAINSET_FILE}
# 체크포인트는 stage 별 폴더에 분리 저장: checkpoints/{sft|gdpo}/{MODEL_ID}/
MODEL_DIR=${CKPT_DIR}/${STAGE}/${MODEL_ID}

echo "=================================================="
echo "  STAGE           : $STAGE"
echo "  CONFIG          : $CONFIG_DIR"
echo "  MODEL_ID        : $MODEL_ID"
echo "  TRAINSET_FILE    : $TRAINSET_FILE"
echo "  BASE_MODEL_ID   : $BASE_MODEL_ID"
echo "  GPUS            : $GPUS  (count=$NUM_GPUS)"
echo "  MODEL           : $MODEL"
echo "  DATASET         : $DATASET"
echo "  MODEL_DIR       : $MODEL_DIR"
echo "=================================================="

# ---- 출력 디렉토리 생성 및 재현용 config 스냅샷 저장 ----
mkdir -p "$MODEL_DIR"
cp "$CONFIG_DIR" "$MODEL_DIR/config.used.yaml"

torchrun --standalone --nproc_per_node=$ARNOLD_WORKER_GPU \
    qwenvl/train/train_qwen.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path "$MODEL" \
    --model_base "$MODEL_BASE" \
    --dataset_use "$DATASET" \
    --tune_mm_vision "$TUNE_MM_VISION" \
    --tune_mm_mlp "$TUNE_MM_MLP" \
    --tune_mm_llm "$TUNE_MM_LLM" \
    --tune_mm_audio "$TUNE_MM_AUDIO" \
    --tune_mm_qformer "$TUNE_MM_QFORMER" \
    --tune_lm_head "$TUNE_LM_HEAD" \
    --use_lora "$USE_LORA" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --bf16 \
    --output_dir "$MODEL_DIR" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --max_pixels "$MAX_PIXELS" \
    --min_pixels "$MIN_PIXELS" \
    --video_max_frame_pixels "$VIDEO_MAX_FRAME_PIXELS" \
    --video_min_frame_pixels "$VIDEO_MIN_FRAME_PIXELS" \
    --eval_strategy "$EVAL_STRATEGY" \
    --save_strategy "$SAVE_STRATEGY" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_ratio "$WARMUP_RATIO" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --logging_steps "$LOGGING_STEPS" \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --run_name "$MODEL_ID" \
    --report_to "$REPORT_TO" \
    --video_min_frames "$VIDEO_MIN_FRAMES" \
    --video_max_frames "$VIDEO_MAX_FRAMES" \
    --base_interval "$BASE_INTERVAL" \
    --train_type "$TRAIN_TYPE" \
    --no_audio "$NO_AUDIO" \
    --tti_time_format "${TTI_TIME_FORMAT:-off}" \
    2>&1 | tee -a "$MODEL_DIR/train.log"
