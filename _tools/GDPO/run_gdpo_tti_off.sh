#!/bin/bash
# ============================================================
# SALMONN2+ GDPO 학습 런처 (TTI-OFF 모드)
#   - SFT LoRA adapter → GDPO 강화학습
#   - 하이퍼파라미터: config.yaml 에서 관리 (CONFIG=... 으로 교체 가능)
#   - 코드 스타일은 _tools/sft/train_salmonn2plus.sh 와 통일
#
# 사용법 (모든 인자는 선택, KEY=VALUE 형식):
#   bash _tools/GDPO/run_gdpo_tti_off.sh \
#       MODEL_ID=gdpo_7b_unav_fps5_off_fmtiou_g8_l256_b004 \
#       SFT_CKPT_ID=sft/salmonn2p_7b_puvalor_fps5_off/checkpoint-2000 \
#       BASE_MODEL_ID=base/video_salmonn2_plus_7B_time_tokens \
#       TRAINSET_FILE=unav100_rl.json \
#       GPUS=0 \
#       CONFIG=config.yaml
#
# 지원 인자:
#   STAGE         : 'gdpo' 만 허용 (이 스크립트는 GDPO 전용).
#   MODEL_ID      : 학습 산출물 식별자.
#                   네이밍 규칙: gdpo_<size>_<dataset>_fps<N>_<format>_<rewards>_g<N>_l<N>_b<TAG>
#                     - g<N>     : config.gdpo.num_generations
#                     - l<N>     : config.gdpo.max_completion_length
#                     - b<TAG>   : config.gdpo.beta (점 제거. 0.04→004, 0.1→01)
#                   비워두면 config.logging.run_name 을 그대로 사용.
#                   config 와 MODEL_ID 의 g/l/b 토큰이 어긋나면 경고.
#   TRAINSET_FILE : ${TRAIN_DIR} (= ${JSON_DIR}/train) 아래 학습 json 파일명.
#   BASE_MODEL_ID : ${CKPT_DIR} 아래 베이스 모델 폴더명 (time_tokens 추가된 버전).
#   SFT_CKPT_ID   : ${CKPT_DIR} 아래 SFT 체크포인트 상대 경로
#                   (예: sft/salmonn2p_7b_puvalor_fps5_off/checkpoint-2000)
#   GPUS          : 사용할 GPU id 목록 (콤마 구분, 예: "0" / "0,1,2,3").
#   CONFIG        : 하이퍼파라미터 yaml 파일명 (스크립트와 같은 폴더 기준).
#
# 체크포인트 저장 구조:
#   Team4/
#   └── checkpoints/              <- ${CKPT_DIR}
#       └── gdpo/
#           └── ${MODEL_ID}/
#               ├── config.used.yaml      (재현용 스냅샷)
#               ├── train.log             (실시간 학습 로그)
#               ├── checkpoint-00500/
#               └── ...
# ============================================================

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAM_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source /workspace/setup.sh
conda activate salmonn2plus

# paths.env 로드 (BASE_DIR / CKPT_DIR / JSON_DIR / TRAIN_DIR / WANDB_* 등 정의)
if [ -f "$TEAM_ROOT/paths.env" ]; then
    source "$TEAM_ROOT/paths.env"
else
    echo "[에러] paths.env 없음: $TEAM_ROOT/paths.env" >&2
    exit 1
fi

# ---- 기본값 ----
STAGE=gdpo
MODEL_ID=                              # 비워두면 config.logging.run_name 사용
TRAINSET_FILE=unav100_rl.json
BASE_MODEL_ID=base/video_salmonn2_plus_7B_time_tokens
SFT_CKPT_ID=sft/salmonn2p_7b_puvalor_fps5_off/checkpoint-2000
GPUS=0
CONFIG=config.yaml

# ---- KEY=VALUE 인자 파싱 ----
for arg in "$@"; do
    case "$arg" in
        STAGE=*)            STAGE="${arg#*=}" ;;
        MODEL_ID=*)         MODEL_ID="${arg#*=}" ;;
        TRAINSET_FILE=*)    TRAINSET_FILE="${arg#*=}" ;;
        BASE_MODEL_ID=*)    BASE_MODEL_ID="${arg#*=}" ;;
        SFT_CKPT_ID=*)      SFT_CKPT_ID="${arg#*=}" ;;
        GPUS=*)             GPUS="${arg#*=}" ;;
        CONFIG=*)           CONFIG="${arg#*=}" ;;
        *)
            echo "[에러] 지원하지 않는 인자: $arg"
            echo "지원 인자: STAGE, MODEL_ID, TRAINSET_FILE, BASE_MODEL_ID, SFT_CKPT_ID, GPUS, CONFIG"
            exit 1
            ;;
    esac
done

# ---- STAGE 검증 (GDPO 전용) ----
STAGE=$(echo "$STAGE" | tr '[:upper:]' '[:lower:]')
if [ "$STAGE" != "gdpo" ]; then
    echo "[에러] 이 스크립트는 GDPO 전용입니다. STAGE=gdpo 만 허용 (받은 값: $STAGE)"
    exit 1
fi

# ---- config 경로 ----
CONFIG_PATH="${SCRIPT_DIR}/${CONFIG}"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "[에러] config 파일을 찾을 수 없음: $CONFIG_PATH"
    exit 1
fi

# ---- config.yaml 파싱 (nested 구조라 python yaml 사용) ----
#   - GDPO 핵심 하이퍼파라미터 (num_generations / max_completion_length / beta) 추출
#   - MODEL_ID (인자값 또는 config.run_name) 와의 토큰 일치성 검증
VALIDATE=$(python - "$CONFIG_PATH" "$MODEL_ID" <<'PY'
import sys, yaml, re
cfg_path = sys.argv[1]
model_id_arg = sys.argv[2].strip()
with open(cfg_path) as f:
    c = yaml.safe_load(f)

ng    = c["gdpo"]["num_generations"]
ml    = c["gdpo"]["max_completion_length"]
beta  = c["gdpo"]["beta"]
rn    = c["logging"]["run_name"]

# beta 표현 규칙: 점 제거 (0.04 → "004", 0.1 → "01", 1.0 → "10")
beta_tag = str(beta).replace(".", "").replace("-", "")

# MODEL_ID 인자가 없으면 config.run_name 사용
mid = model_id_arg or rn

issues = []
if not re.search(rf"_g{ng}(?![0-9])", mid):
    issues.append(f"MODEL_ID 에 'g{ng}' 가 없음  (config.gdpo.num_generations={ng})")
if not re.search(rf"_l{ml}(?![0-9])", mid):
    issues.append(f"MODEL_ID 에 'l{ml}' 가 없음  (config.gdpo.max_completion_length={ml})")
if not re.search(rf"_b{beta_tag}(?![0-9])", mid):
    issues.append(f"MODEL_ID 에 'b{beta_tag}' 가 없음  (config.gdpo.beta={beta})")

print(f"MODEL_ID={mid}")
print(f"NG={ng}")
print(f"ML={ml}")
print(f"BETA={beta}")
print(f"BETA_TAG={beta_tag}")
print("ISSUES_BEGIN")
for it in issues:
    print(it)
print("ISSUES_END")
PY
)

# ---- VALIDATE 결과 추출 ----
MODEL_ID=$(echo "$VALIDATE" | awk -F= '/^MODEL_ID=/{print $2}')
NG=$(echo "$VALIDATE" | awk -F= '/^NG=/{print $2}')
ML=$(echo "$VALIDATE" | awk -F= '/^ML=/{print $2}')
BETA=$(echo "$VALIDATE" | awk -F= '/^BETA=/{print $2}')
BETA_TAG=$(echo "$VALIDATE" | awk -F= '/^BETA_TAG=/{print $2}')

# ---- MODEL_ID vs config.yaml 일치성 검증 (경고만, 학습은 계속 진행) ----
# grep 이 매치 없을 때 exit 1을 내서 set -e 가 스크립트를 죽이는 걸 방지 (|| true).
ISSUE_LINES=$(echo "$VALIDATE" | sed -n '/ISSUES_BEGIN/,/ISSUES_END/p' | grep -v 'ISSUES_' || true)
if [ -n "$ISSUE_LINES" ]; then
    cat >&2 <<EOF

[경고] MODEL_ID 와 config.yaml 의 GDPO 하이퍼파라미터가 어긋나 보입니다.
  MODEL_ID    : ${MODEL_ID}
  config.yaml : num_generations=${NG}        →  'g${NG}'
                max_completion_length=${ML}  →  'l${ML}'
                beta=${BETA}                 →  'b${BETA_TAG}'
  발견된 문제 :
EOF
    echo "$ISSUE_LINES" | sed 's/^/    - /' >&2
    cat >&2 <<EOF
  권장 이름   : <prefix>_g${NG}_l${ML}_b${BETA_TAG}

  ※ 학습은 그대로 진행됩니다. 의도한 네이밍이면 무시,
    실수라면 Ctrl+C 후 MODEL_ID 또는 config.yaml 을 다시 지정해 재실행하세요.

EOF
fi

# ---- GPU 개수 자동 계산 ----
NUM_GPUS=$(echo "$GPUS" | awk -F',' '{print NF}')

export CUDA_VISIBLE_DEVICES=$GPUS
export ARNOLD_WORKER_GPU=$NUM_GPUS
export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=localhost

# ---- 환경변수 안전장치 ----
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# 메모리 단편화 완화 (단일 GPU에서 7B + 8 generations 학습 시 도움)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# wandb run name 충돌 방지 (config / CLI 인자에서 지정)
unset WANDB_NAME
# WANDB_API_KEY 가 없을 때만 offline fallback. 있으면 online 그대로.
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[run_gdpo_tti_off] WANDB_API_KEY 없음 → WANDB_MODE=offline 으로 강제."
    export WANDB_MODE=offline
fi

# gdpo_trainer.py 내부의 from-import 가 TEAM_ROOT 기준으로 동작
cd "$TEAM_ROOT"
export PYTHONPATH="$TEAM_ROOT:$PYTHONPATH"

# ---- 경로 조립 ----
MODEL=${CKPT_DIR}/${SFT_CKPT_ID}        # SFT LoRA adapter
MODEL_BASE=${CKPT_DIR}/${BASE_MODEL_ID}  # base 모델 (time_tokens 포함)
DATASET=${TRAIN_DIR}/${TRAINSET_FILE}
if [ ! -f "$DATASET" ]; then
    echo "[에러] 학습 JSON 없음: $DATASET" >&2
    echo "  TRAIN_DIR=${TRAIN_DIR}" >&2
    echo "  TRAINSET_FILE=${TRAINSET_FILE}" >&2
    exit 1
fi
if [ ! -d "$MODEL" ]; then
    echo "[에러] SFT 체크포인트 없음: $MODEL" >&2
    exit 1
fi
if [ ! -d "$MODEL_BASE" ]; then
    echo "[에러] BASE 모델 없음: $MODEL_BASE" >&2
    exit 1
fi
# 체크포인트는 stage 별 폴더에 분리 저장: checkpoints/gdpo/${MODEL_ID}/
MODEL_DIR=${CKPT_DIR}/${STAGE}/${MODEL_ID}

echo "=================================================="
echo "  STAGE           : $STAGE"
echo "  CONFIG          : $CONFIG_PATH"
echo "  MODEL_ID        : $MODEL_ID"
echo "  TRAINSET_FILE   : $TRAINSET_FILE"
echo "  BASE_MODEL_ID   : $BASE_MODEL_ID"
echo "  SFT_CKPT_ID     : $SFT_CKPT_ID"
echo "  GPUS            : $GPUS  (count=$NUM_GPUS)"
echo "  MODEL (SFT ckpt): $MODEL"
echo "  MODEL_BASE      : $MODEL_BASE"
echo "  DATASET         : $DATASET"
echo "  MODEL_DIR       : $MODEL_DIR"
echo "  GDPO HP         : g=$NG  l=$ML  b=$BETA  (tag=b${BETA_TAG})"
echo "=================================================="

# ---- 출력 디렉토리 생성 및 재현용 config 스냅샷 저장 ----
mkdir -p "$MODEL_DIR"
cp "$CONFIG_PATH" "$MODEL_DIR/config.used.yaml"

torchrun --standalone --nproc_per_node=$ARNOLD_WORKER_GPU \
    _tools/GDPO/gdpo_trainer.py \
    --config "$CONFIG_PATH" \
    --model_path "$MODEL" \
    --model_base "$MODEL_BASE" \
    --dataset_path "$DATASET" \
    --output_dir "$MODEL_DIR" \
    --run_name "$MODEL_ID" \
    2>&1 | tee -a "$MODEL_DIR/train.log"
