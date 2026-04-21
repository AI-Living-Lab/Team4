#!/bin/bash
# ============================================================
# 로컬 체크포인트 -> HuggingFace Hub 업로드
#
# paths.env의 HF_TOKEN / HF_NAMESPACE / HF_PRIVATE를 읽어 사용한다.
#
# CKPT 경로에서 repo 이름과 path_in_repo를 자동 유도한다.
#   CKPT=sft/salmonn2p_7b_unav_baseline/checkpoint-1500
#     -> repo_id       = ${HF_NAMESPACE}/sft_salmonn2p_7b_unav_baseline
#        path_in_repo  = checkpoint-1500
#   CKPT=sft/exp_v1 (하위에 checkpoint-* 이 없고 단일 폴더)
#     -> repo_id       = ${HF_NAMESPACE}/sft_exp_v1
#        path_in_repo  = <root>
#
# 사용법 (KEY=VALUE):
#   bash _tools/push_checkpoint.sh CKPT=sft/salmonn2p_7b_unav_baseline/checkpoint-1500
#
#   # 자동 유도 대신 수동으로 덮어쓰기
#   bash _tools/push_checkpoint.sh \
#       CKPT=gdpo/exp_v1/checkpoint-500 \
#       REPO_NAME=salmonn2p-gdpo-v1 \
#       PATH_IN_REPO=step-500
#
#   # 완전히 다른 namespace/레포로
#   bash _tools/push_checkpoint.sh \
#       CKPT=sft/.../checkpoint-1500 \
#       REPO_ID=someone-else/custom-repo
#
#   # 기본 IGNORE 외에 추가로 제외
#   bash _tools/push_checkpoint.sh \
#       CKPT=sft/.../checkpoint-1500 \
#       IGNORE="extra_file.bin"
#
#   # 기본 IGNORE 전부 해제하고 전체 업로드 (SFT resume용 등)
#   bash _tools/push_checkpoint.sh \
#       CKPT=sft/.../checkpoint-1500 \
#       NO_DEFAULT_IGNORE=true
#
# 기본 IGNORE (GRPO 백본/공유 용도에서 불필요하고, PEFT가 깨뜨리는 파일들):
#   global_step*/*       DeepSpeed ZeRO 옵티마이저 상태 (수 GB~수십 GB, resume 전용)
#   latest               DeepSpeed 포인터 파일
#   zero_to_fp32.py      DeepSpeed 변환 헬퍼
#   README.md            PEFT 자동 생성, base_model에 로컬 경로가 박혀 HF YAML 검증 실패
#
# 인자:
#   CKPT                업로드할 체크포인트 경로 (절대경로 or CKPT_DIR 기준 상대경로)
#   REPO_ID             HF 레포 풀 id. 지정 시 REPO_NAME/HF_NAMESPACE 무시
#   REPO_NAME           레포 이름만 지정. HF_NAMESPACE와 합쳐짐
#   PATH_IN_REPO        레포 내 서브폴더 (자동 유도 덮어쓰기)
#   PRIVATE             true/false. 미지정 시 HF_PRIVATE
#   MESSAGE             커밋 메시지
#   IGNORE              기본 IGNORE에 추가할 glob (공백 구분)
#   NO_DEFAULT_IGNORE   true면 기본 IGNORE 적용 안 함 (IGNORE=만 사용)
#   ALLOW               업로드 포함 glob (지정 시 이 패턴만 업로드)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# conda env 활성화 (huggingface_hub 이 salmonn2plus env에 설치되어 있음)
if [ -f /workspace/setup.sh ]; then
    # shellcheck disable=SC1091
    source /workspace/setup.sh
fi
if command -v conda >/dev/null 2>&1; then
    conda activate "${CONDA_ENV:-salmonn2plus}"
fi

if [ -f "${BASE_DIR}/paths.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${BASE_DIR}/paths.env"
    set +a
else
    echo "[에러] paths.env를 찾을 수 없습니다: ${BASE_DIR}/paths.env"
    exit 1
fi

# ---- 기본값 ----
CKPT=
REPO_ID=
REPO_NAME=
PATH_IN_REPO=
PRIVATE="${HF_PRIVATE:-false}"
MESSAGE=
IGNORE=
NO_DEFAULT_IGNORE=false
ALLOW=

# 공유/추론/GRPO 백본 용도에서 불필요하고, PEFT README는 HF YAML 검증에서 튕기는 파일들.
# resume이 필요하면 NO_DEFAULT_IGNORE=true로 해제.
DEFAULT_IGNORE="global_step*/* latest zero_to_fp32.py README.md"

# ---- KEY=VALUE 파싱 ----
for arg in "$@"; do
    case "$arg" in
        CKPT=*)         CKPT="${arg#*=}" ;;
        REPO_ID=*)      REPO_ID="${arg#*=}" ;;
        REPO_NAME=*)    REPO_NAME="${arg#*=}" ;;
        PATH_IN_REPO=*) PATH_IN_REPO="${arg#*=}" ;;
        PRIVATE=*)      PRIVATE="${arg#*=}" ;;
        MESSAGE=*)      MESSAGE="${arg#*=}" ;;
        IGNORE=*)            IGNORE="${arg#*=}" ;;
        NO_DEFAULT_IGNORE=*) NO_DEFAULT_IGNORE="${arg#*=}" ;;
        ALLOW=*)             ALLOW="${arg#*=}" ;;
        *)
            echo "[에러] 지원하지 않는 인자: $arg"
            echo "지원: CKPT, REPO_ID, REPO_NAME, PATH_IN_REPO, PRIVATE, MESSAGE, IGNORE, NO_DEFAULT_IGNORE, ALLOW"
            exit 1
            ;;
    esac
done

# ---- 기본 IGNORE 병합 ----
if [ "$NO_DEFAULT_IGNORE" != "true" ] && [ "$NO_DEFAULT_IGNORE" != "True" ]; then
    IGNORE="$(echo "$DEFAULT_IGNORE $IGNORE" | xargs)"  # 중복 공백 정리
fi

# ---- 검증 ----
if [ -z "$CKPT" ]; then
    echo "[에러] CKPT= 는 필수입니다"
    exit 1
fi
if [ -z "$HF_TOKEN" ]; then
    echo "[에러] paths.env에 HF_TOKEN이 설정되지 않았습니다"
    exit 1
fi

# ---- 절대경로 정규화 ----
if [[ "$CKPT" = /* ]]; then
    CKPT_ABS="$CKPT"
else
    CKPT_ABS="${CKPT_DIR}/${CKPT}"
fi

if [ ! -d "$CKPT_ABS" ]; then
    echo "[에러] 체크포인트 디렉터리가 없습니다: $CKPT_ABS"
    exit 1
fi

# ---- CKPT_DIR 기준 상대경로에서 repo_name / path_in_repo 자동 유도 ----
# 규칙: 마지막 세그먼트는 path_in_repo, 나머지는 '_'로 이어붙여 repo_name.
# 단, 세그먼트가 하나뿐이면 그것이 repo_name이고 path_in_repo는 root.
if [[ "$CKPT_ABS" == "$CKPT_DIR"/* ]]; then
    REL="${CKPT_ABS#"$CKPT_DIR"/}"
    LAST="$(basename "$REL")"
    PARENT="$(dirname "$REL")"
    if [ "$PARENT" = "." ]; then
        REPO_NAME_AUTO="$LAST"
        PATH_IN_REPO_AUTO=""
    else
        REPO_NAME_AUTO="${PARENT//\//_}"
        PATH_IN_REPO_AUTO="$LAST"
    fi
else
    REPO_NAME_AUTO=""
    PATH_IN_REPO_AUTO=""
fi

# ---- repo_id 확정 ----
if [ -z "$REPO_ID" ]; then
    if [ -z "$REPO_NAME" ]; then
        REPO_NAME="$REPO_NAME_AUTO"
    fi
    if [ -z "$REPO_NAME" ]; then
        echo "[에러] repo_name을 유도할 수 없습니다. CKPT가 CKPT_DIR 밖에 있으면"
        echo "        REPO_ID= 또는 REPO_NAME= 을 직접 지정하세요."
        exit 1
    fi
    if [ -z "$HF_NAMESPACE" ]; then
        echo "[에러] paths.env에 HF_NAMESPACE가 설정되지 않았습니다 (또는 REPO_ID=로 풀 id 지정)"
        exit 1
    fi
    REPO_ID="${HF_NAMESPACE}/${REPO_NAME}"
fi

# ---- path_in_repo 확정 (사용자 지정 > 자동 유도) ----
if [ -z "$PATH_IN_REPO" ]; then
    PATH_IN_REPO="$PATH_IN_REPO_AUTO"
fi

# ---- Python 인자 구성 ----
PY_ARGS=(--ckpt_path "$CKPT_ABS" --repo_id "$REPO_ID")
[ -n "$PATH_IN_REPO" ] && PY_ARGS+=(--path_in_repo "$PATH_IN_REPO")
[ -n "$MESSAGE" ]      && PY_ARGS+=(--commit_message "$MESSAGE")
if [ "$PRIVATE" = "true" ] || [ "$PRIVATE" = "True" ]; then
    PY_ARGS+=(--private)
fi
# shellcheck disable=SC2206
[ -n "$IGNORE" ] && PY_ARGS+=(--ignore_patterns $IGNORE)
# shellcheck disable=SC2206
[ -n "$ALLOW" ]  && PY_ARGS+=(--allow_patterns $ALLOW)

echo "=================================================="
echo "  CKPT         : $CKPT_ABS"
echo "  REPO_ID      : $REPO_ID"
echo "  PATH_IN_REPO : ${PATH_IN_REPO:-<root>}"
echo "  PRIVATE      : $PRIVATE"
echo "  IGNORE       : ${IGNORE:-<none>}"
echo "  ALLOW        : ${ALLOW:-<none>}"
echo "=================================================="

python "${SCRIPT_DIR}/push_checkpoint.py" "${PY_ARGS[@]}"
