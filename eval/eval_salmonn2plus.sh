#!/bin/bash
# ============================================================
# SALMONN2+ Eval 런처
#   - 학습된 LoRA 체크포인트 또는 베이스 모델 단독으로 추론
#   - 평가 지표: Union-IoU 기반 mIoU + Recall@{0.3,0.5,0.7} + FP_rate
#     · mIoU(union)  : GT 하나당 겹치는 pred 합집합과의 IoU 평균 (쪼개기 손해 방지)
#     · Recall@θ     : Union-IoU ≥ θ 인 GT segment 비율
#     · FP_rate      : GT와 전혀 겹치지 않는 pred 비율 (과잉/무관 예측 페널티)
#   - 하이퍼파라미터는 config.yaml 에서 관리 (CONFIG=... 으로 교체 가능)
#
# 사용법 (모든 인자 선택, KEY=VALUE 형식):
#   bash eval_salmonn2plus.sh \
#       STAGE=sft \
#       CKPT_MODEL_ID=salmonn2p_7b_unav_baseline \
#       CKPT_STEP=5000 \
#       BASE_MODEL_ID=video_salmonn2_plus_7B_time_tokens \
#       TESTSET_FILE=unav100_test_multiseg_sub80.json \
#       GPUS=0 \
#       CONFIG=config.yaml
#
# 지원 인자:
#   STAGE         : sft | gdpo  (대소문자 무관, 기본값 sft)
#                   -> 어느 stage 체크포인트를 쓸지 결정
#   CKPT_MODEL_ID : 평가할 학습 산출물 식별자 (train 의 MODEL_ID 와 동일)
#                   네이밍 규칙: {model}_{size}_{dataset}_{설정tag}
#                   예) salmonn2p_7b_unav_baseline
#   CKPT_STEP     : 평가할 체크포인트 스텝 번호.
#                   - 생략            -> 해당 모델의 가장 최근 checkpoint-*
#                   - 5000            -> checkpoint-5000
#                   - checkpoint-5000 -> 같은 의미
#                   - base / no       -> LoRA 없이 베이스 모델만 평가
#   BASE_MODEL_ID : ${CKPT_DIR} 아래 베이스 모델 폴더명
#   TESTSET_FILE  : ${BASE_DIR}/data 아래 테스트 json 파일명
#                   (확장자 제외한 이름이 결과 하위 폴더가 됨)
#   GPUS          : 사용할 GPU id 목록 (콤마 구분, 예: "0" / "0,1")
#   CONFIG        : 하이퍼파라미터 yaml 파일명 (스크립트와 같은 폴더 기준)
#
# 결과 저장 구조 예시:
#   (testset 하위 폴더명 = TESTSET_FILE 에서 .json 만 제거)
#   Team4/
#   └── outputs/
#       ├── sft/
#       │   └── salmonn2p_7b_unav_baseline/              <- {CKPT_MODEL_ID}
#       │       ├── checkpoint-500/
#       │       │   └── unav100_test_sub80/              <- {TESTSET_NAME}
#       │       │       ├── eval_miou_summary.json
#       │       │       ├── test_results_rank0.json
#       │       │       └── inference.log
#       │       └── checkpoint-1000/
#       ├── gdpo/
#       │   └── salmonn2p_7b_unav_baseline/
#       │       └── checkpoint-500/
#       │           └── unav100_test_full/
#       │               ├── eval_miou_summary.json
#       │               ├── test_results_rank0.json
#       │               └── inference.log
#       └── base/                                         <- CKPT_STEP=base 일 때
#           └── video_salmonn2_plus_7B_time_tokens/       <- {BASE_MODEL_ID}
#               └── unav100_test_sub80/
#                   ├── eval_miou_summary.json
#                   ├── test_results_rank0.json
#                   └── inference.log
#
# 참고: 업스트림 train_qwen.py 는 test_results_rank0.json 을
#   os.path.join(output_dir, run_name, ...) 로 저장하기 때문에,
#   flat 구조를 유지하려면 --run_name "." 로 넘겨야 합니다.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /workspace/setup.sh
conda activate salmonn2plus

export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

# ---- 기본값 ----
STAGE=sft
CKPT_MODEL_ID=salmonn2p_7b_unav_baseline
CKPT_STEP=
BASE_MODEL_ID=video_salmonn2_plus_7B_time_tokens
TESTSET_FILE=unav100_test_multiseg_sub80.json
GPUS=0
CONFIG=config.yaml

# ---- KEY=VALUE 인자 파싱 ----
for arg in "$@"; do
    case "$arg" in
        STAGE=*)            STAGE="${arg#*=}" ;;
        CKPT_MODEL_ID=*)    CKPT_MODEL_ID="${arg#*=}" ;;
        CKPT_STEP=*)        CKPT_STEP="${arg#*=}" ;;
        BASE_MODEL_ID=*)    BASE_MODEL_ID="${arg#*=}" ;;
        TESTSET_FILE=*)     TESTSET_FILE="${arg#*=}" ;;
        GPUS=*)             GPUS="${arg#*=}" ;;
        CONFIG=*)           CONFIG="${arg#*=}" ;;
        *)
            echo "[에러] 지원하지 않는 인자: $arg"
            echo "지원 인자: STAGE, CKPT_MODEL_ID, CKPT_STEP, BASE_MODEL_ID, TESTSET_FILE, GPUS, CONFIG"
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

# ---- config.yaml 로드 ----
CONFIG_DIR="${SCRIPT_DIR}/${CONFIG}"
if [ ! -f "$CONFIG_DIR" ]; then
    echo "[에러] config 파일을 찾을 수 없음: $CONFIG_DIR"
    exit 1
fi

while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    line="${line%%#*}"
    [[ -z "${line// }" ]] && continue
    key="${line%%:*}"
    val="${line#*:}"
    key="$(echo "$key" | awk '{$1=$1;print}')"
    val="$(echo "$val" | awk '{$1=$1;print}')"
    val="${val#\"}"; val="${val%\"}"
    val="${val#\'}"; val="${val%\'}"
    [[ -z "$key" ]] && continue
    eval "$key=\"\$val\""
done < "$CONFIG_DIR"

# ---- GPU 개수 자동 계산 ----
NUM_GPUS=$(echo "$GPUS" | awk -F',' '{print NF}')
export CUDA_VISIBLE_DEVICES=$GPUS
export ARNOLD_WORKER_GPU=$NUM_GPUS

# ---- 경로 조립 ----
BASE_CODE="${BASE_DIR}/video_SALMONN2_plus"
MODEL_BASE="${CKPT_DIR}/base/${BASE_MODEL_ID}"
TEST_JSON="${BASE_DIR}/data/${TESTSET_FILE}"
MODEL_DIR="${CKPT_DIR}/${STAGE}/${CKPT_MODEL_ID}"
EVAL_SCRIPT="${BASE_DIR}/eval/eval_miou_multiseg.py"

# 테스트 json 존재 확인
if [ ! -f "$TEST_JSON" ]; then
    echo "[에러] 테스트 json 없음: $TEST_JSON"
    exit 1
fi

# 결과 하위 폴더명: TESTSET_FILE 에서 .json 제거
TESTSET_NAME="${TESTSET_FILE%.json}"

# ---- CKPT_STEP → LoRA 경로 / 체크포인트 폴더명 결정 ----
# 1) 생략       -> 가장 최근 checkpoint-*
# 2) base/no    -> LoRA 미적용 (베이스 모델만)
# 3) 그 외      -> checkpoint-<CKPT_STEP>
CKPT_STEP_LOWER=$(echo "$CKPT_STEP" | tr '[:upper:]' '[:lower:]')
IS_BASE_EVAL=false
if [ -z "$CKPT_STEP" ]; then
    LORA_CKPT=$(ls -d "$MODEL_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
    if [ -z "$LORA_CKPT" ]; then
        echo "[에러] $MODEL_DIR 에서 checkpoint-* 를 찾을 수 없습니다"
        exit 1
    fi
    CKPT_FOLDER=$(basename "$LORA_CKPT")
elif [ "$CKPT_STEP_LOWER" = "base" ] || [ "$CKPT_STEP_LOWER" = "no" ]; then
    LORA_CKPT="No"
    CKPT_FOLDER="base"
    IS_BASE_EVAL=true
else
    if [[ "$CKPT_STEP" == checkpoint-* ]]; then
        CKPT_FOLDER="$CKPT_STEP"
    else
        CKPT_FOLDER="checkpoint-${CKPT_STEP}"
    fi
    LORA_CKPT="${MODEL_DIR}/${CKPT_FOLDER}"
    if [ ! -d "$LORA_CKPT" ]; then
        echo "[에러] 체크포인트 폴더 없음: $LORA_CKPT"
        exit 1
    fi
fi

# ---- 결과 디렉토리 ----
# 일반:     outputs/{stage}/{ckpt_model_id}/{checkpoint-xxxx}/{testset}/
# 베이스:   outputs/base/{base_model_id}/{testset}/
if [ "$IS_BASE_EVAL" = "true" ]; then
    OUT_DIR="${BASE_DIR}/outputs/base/${BASE_MODEL_ID}/${TESTSET_NAME}"
else
    OUT_DIR="${BASE_DIR}/outputs/${STAGE}/${CKPT_MODEL_ID}/${CKPT_FOLDER}/${TESTSET_NAME}"
fi

echo "=================================================="
echo "  STAGE           : $STAGE"
echo "  CKPT_MODEL_ID   : $CKPT_MODEL_ID"
echo "  CKPT_STEP       : ${CKPT_STEP:-<latest>}  ->  $CKPT_FOLDER"
echo "  LORA_CKPT       : $LORA_CKPT"
echo "  BASE_MODEL_ID   : $BASE_MODEL_ID"
echo "  TESTSET         : $TEST_JSON"
echo "  GPUS            : $GPUS  (count=$NUM_GPUS)"
echo "  CONFIG          : $CONFIG_DIR"
echo "  OUT_DIR         : $OUT_DIR"
echo "=================================================="

mkdir -p "$OUT_DIR"

# ---- 종료 시 merge된 임시 모델(generation_*) 자동 정리 ----
# train_qwen.py 가 LoRA merge 결과를 $OUT_DIR/generation_0/ 에 ~15G 저장하는데
# 평가 이후에는 불필요. 스크립트가 실패해도 정리되도록 trap 사용.
cleanup_gen() { rm -rf "$OUT_DIR"/generation_* 2>/dev/null || true; }
trap cleanup_gen EXIT

cd "$BASE_CODE"

# torchrun 포트 (GPUS 첫 id 로 충돌 회피)
FIRST_GPU=$(echo "$GPUS" | awk -F',' '{print $1}')
MASTER_PORT=$((12900 + FIRST_GPU))

# ---- 추론 실행 ----
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    qwenvl/train/train_qwen.py \
    --model_base "$MODEL_BASE" \
    --run_test True \
    --pred_rank 0 \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path "$MODEL_BASE" \
    --dataset_use "$TEST_JSON" \
    --bf16 \
    --output_dir "$OUT_DIR" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --max_pixels "$MAX_PIXELS" \
    --min_pixels "$MIN_PIXELS" \
    --video_max_frame_pixels "$VIDEO_MAX_FRAME_PIXELS" \
    --video_min_frame_pixels "$VIDEO_MIN_FRAME_PIXELS" \
    --eval_strategy "$EVAL_STRATEGY" \
    --save_strategy "$SAVE_STRATEGY" \
    --learning_rate "$LEARNING_RATE" \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --run_name "." \
    --report_to "$REPORT_TO" \
    --video_min_frames "$VIDEO_MIN_FRAMES" \
    --video_max_frames "$VIDEO_MAX_FRAMES" \
    --base_interval "$BASE_INTERVAL" \
    --lora_ckpt "$LORA_CKPT" \
    --no_audio "$NO_AUDIO" \
    2>&1 | tee "$OUT_DIR/inference.log"

# ---- mIoU 평가 ----
RESULT_FILE=$(find "$OUT_DIR" -name "test_results_rank0.json" 2>/dev/null | head -1)
if [ ! -f "$RESULT_FILE" ]; then
    echo "[에러] 추론 결과 파일을 찾을 수 없습니다: $OUT_DIR/test_results_rank0.json"
    exit 1
fi

python3 "$EVAL_SCRIPT" \
    --results "$RESULT_FILE" \
    --test_json "$TEST_JSON" \
    --max_time "$MAX_TIME" \
    --out_dir "$OUT_DIR"

echo "[완료] $OUT_DIR 에 평가 결과 저장 (eval_miou_summary.json)"
