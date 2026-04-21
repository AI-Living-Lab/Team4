# Team4 — video-SALMONN-2+ UnAV-100 Pipeline

UnAV-100 오디오-비주얼 이벤트 로컬라이제이션을 위한 video-SALMONN-2+ 학습/평가/배포 파이프라인.

## 📁 UnAV-100 Dataset

### 🎬 Raw Video
- [Download Raw Videos](https://drive.google.com/drive/folders/1YtKugZrNJ8iCEtncyMCPdQ1Qdrfal9_U)

### 🔎 Features Extracted with I3D + VGGish
- [Download Features](https://drive.google.com/drive/folders/1xcNnXLVfd7cJEoGUvJYHerCXQnKFoPS-)

### 🧠 Features Extracted with ONE-PEACE
- [Download Features](https://drive.google.com/drive/folders/1wKnNlNU1FHiw3lN7kaT09frlM0bg4XjI)
- Files included:
  - `av_features.tar.gz` — Video features
  - `av_features_audio_al_retrieval.tar.gz` — Audio features

### 📄 학습/평가용 JSON (`data/`)

| 파일 | 용도 | 크기 |
|---|---|---|
| `unav100_train.json` | SFT 학습 (멀티세그먼트 QA) | — |
| `unav100_train_dense.json` | dense 형식 학습 (이벤트 JSON 출력) | — |
| `unav100_test_sub80.json` | 소규모 평가셋 | 119 샘플 |
| `unav100_test_full.json` | 전체 평가셋 (dense → 라벨별 변환) | 3455 샘플 |
| `unav100_test_dense.json` | dense 형식 테스트 원본 | 2167 비디오 |

## 🐟 video-SALMONN-2+ Setup

Repository reference: [video-SALMONN-2 GitHub](https://github.com/bytedance/video-SALMONN-2)

### 📥 Download Checkpoints

베이스 모델 `video_salmonn2_plus_7B_time_tokens`은 video-SALMONN-2 위에 VTG-LLM time token (`<t0>`~`<t9>`, `<tdot>`)을 추가한 변형입니다.

**옵션 A: 팀 HF 레포에서 통째로 다운로드 (권장)**

```bash
huggingface-cli download ewha-team404/video_salmonn2_plus_7B_time_tokens \
    --local-dir ${CKPT_DIR}/video_salmonn2_plus_7B_time_tokens
```

**옵션 B: 원본을 받아 직접 time token 추가**

```bash
# 1) 원본 video-SALMONN-2 다운로드
huggingface-cli download tsinghua-ee/video-SALMONN-2 \
    --local-dir ${CKPT_DIR}/video_salmonn2_plus_7B

# 2) time token 추가
python _tools/sft/add_time_tokens_salmonn2plus.py \
    --model_path ${CKPT_DIR}/video_salmonn2_plus_7B \
    --output_path ${CKPT_DIR}/video_salmonn2_plus_7B_time_tokens
```

### 📥 학습된 체크포인트 다운로드 (HF 레포)

이미 학습된 SFT/GDPO 결과를 받아 GRPO 백본·평가에 쓸 때:

```bash
# 특정 체크포인트만
huggingface-cli download ewha-team404/sft_salmonn2p_7b_unav_baseline \
    --include "checkpoint-1500/*" \
    --local-dir ${CKPT_DIR}/sft/salmonn2p_7b_unav_baseline

# Python에서
from huggingface_hub import snapshot_download
snapshot_download(
    "ewha-team404/sft_salmonn2p_7b_unav_baseline",
    allow_patterns="checkpoint-1500/*",
    local_dir=f"{os.environ['CKPT_DIR']}/sft/salmonn2p_7b_unav_baseline",
)
```

> 비공개 레포는 `huggingface-cli login` 또는 `HF_TOKEN` 환경변수 필요.

### 🔧 Environment Variables (`paths.env`)

`paths.env.example` 를 복사하여 `paths.env` 를 만들고 본인 환경에 맞게 채웁니다 (`paths.env` 는 `.gitignore` 처리되어 있어 토큰 포함 가능).

**주요 변수:**

| 변수 | 용도 |
|---|---|
| `BASE_DIR` | 이 레포 루트 |
| `DATA_DIR` | 데이터셋 루트 |
| `MODEL_DIR` | HF 모델 캐시/베이스 모델 루트 |
| `HF_HOME` | HuggingFace 캐시 (`${MODEL_DIR}/.cache/huggingface`) |
| `CKPT_DIR` | 체크포인트 저장 루트 (학습 산출물 포함) |
| `JSON_DIR` | 학습/평가 JSON 디렉토리 (`${BASE_DIR}/data`) |
| `EVAL_DIR` | 평가 결과 디렉토리 (`${BASE_DIR}/outputs`) |
| `UNAV_VIDEO_DIR` / `UNAV_AUDIO_DIR` | UnAV-100 원본 (전처리 시) |
| `HF_TOKEN` | HuggingFace write 토큰 (체크포인트 업로드용) |
| `HF_NAMESPACE` | HF 유저/조직명. 업로드 레포 이름은 CKPT 경로에서 자동 유도 |
| `HF_PRIVATE` | 새 레포 private 여부 (`true`/`false`) |

### 📂 Directory Structure

```bash
Team4/
├── checkpoints/                               # ${CKPT_DIR}
│   ├── video_salmonn2_plus_7B_time_tokens/    # 베이스 모델
│   ├── sft/                                   # SFT 학습 산출물
│   │   └── {MODEL_ID}/
│   │       ├── checkpoint-00500/
│   │       ├── checkpoint-01000/
│   │       └── config.used.yaml               # 재현용 스냅샷
│   └── gdpo/
│       └── {MODEL_ID}/
├── outputs/                                   # ${EVAL_DIR}
│   ├── sft/
│   │   └── {MODEL_ID}/{checkpoint-xxxx}/{testset}/
│   │       ├── eval_miou_summary.json
│   │       ├── test_results_rank0.json
│   │       └── inference.log
│   ├── gdpo/
│   └── base/                                  # CKPT_STEP=base 일 때
├── data/                                      # ${JSON_DIR}
├── _tools/
│   ├── push_checkpoint.sh                     # 체크포인트 → HF 업로드 래퍼
│   ├── push_checkpoint.py                     # 업로드 Python 구현
│   ├── build_pu_valor.py                      # PU-VALOR 데이터 전처리
│   ├── fix_json_paths.sh                      # JSON 내부 경로 일괄 치환
│   ├── sft/
│   │   ├── train_salmonn2plus.sh              # SFT 학습 런처
│   │   ├── config.yaml                        # 학습 하이퍼파라미터
│   │   ├── merge_lora_and_push.py             # LoRA merge + HF 업로드
│   │   ├── merge_and_push.sh                  # merge 래퍼
│   │   └── add_time_tokens_salmonn2plus.py    # 베이스 모델에 time token 추가
│   └── GDPO/                                  # GDPO 학습 코드
└── eval/
    ├── eval_salmonn2plus.sh                   # 평가 런처
    ├── config.yaml                            # 평가 하이퍼파라미터
    └── eval_miou_multiseg.py                  # mIoU / R@k 계산
```

## 🚀 Usage

모든 런처는 `KEY=VALUE` 스타일 인자를 받으며, 생략 시 기본값이 적용됩니다.

### 1) SFT 학습

```bash
bash _tools/sft/train_salmonn2plus.sh \
    MODEL_ID=salmonn2p_7b_unav_baseline \
    TRAINSET_FILE=unav100_train.json \
    GPUS=0,1
```

**지원 인자**: `STAGE` (sft/gdpo), `MODEL_ID`, `TRAINSET_FILE`, `BASE_MODEL_ID`, `GPUS`, `CONFIG`

**저장 위치**: `${CKPT_DIR}/${STAGE}/${MODEL_ID}/checkpoint-xxxxx/`

**MODEL_ID 네이밍 규칙**: `{model}_{size}_{dataset}_{설정tag}`
예) `salmonn2p_7b_unav_baseline`, `salmonn2p_7b_unav_tti`

### 2) 평가

```bash
# 가장 최근 체크포인트 평가
bash eval/eval_salmonn2plus.sh CKPT_MODEL_ID=salmonn2p_7b_unav_baseline

# 특정 스텝 + 전체 testset
bash eval/eval_salmonn2plus.sh \
    CKPT_MODEL_ID=salmonn2p_7b_unav_baseline \
    CKPT_STEP=5000 \
    TESTSET_FILE=unav100_test_full.json

# GDPO 체크포인트
bash eval/eval_salmonn2plus.sh \
    STAGE=gdpo \
    CKPT_MODEL_ID=salmonn2p_7b_unav_baseline \
    CKPT_STEP=500

# 베이스 모델만 (LoRA 없이)
bash eval/eval_salmonn2plus.sh CKPT_STEP=base
```

**지원 인자**: `STAGE`, `CKPT_MODEL_ID`, `CKPT_STEP`, `BASE_MODEL_ID`, `TESTSET_FILE`, `GPUS`, `CONFIG`

**결과 저장 위치**:
- 일반: `outputs/{stage}/{CKPT_MODEL_ID}/{checkpoint-xxxx}/{testset}/`
- 베이스: `outputs/base/{BASE_MODEL_ID}/{testset}/`

### 3) 체크포인트 HuggingFace 업로드

학습된 체크포인트(LoRA 어댑터 그대로)를 HF 레포에 올립니다. 레포 이름과 path는 CKPT 경로에서 자동 유도됩니다.

```bash
# 가장 단순한 사용 (paths.env의 HF_TOKEN/HF_NAMESPACE 사용)
bash _tools/push_checkpoint.sh CKPT=sft/salmonn2p_7b_unav_baseline/checkpoint-1500
# -> https://huggingface.co/${HF_NAMESPACE}/sft_salmonn2p_7b_unav_baseline/tree/main/checkpoint-1500

# 전체 업로드 (DeepSpeed optimizer state 포함, SFT resume용)
bash _tools/push_checkpoint.sh \
    CKPT=sft/salmonn2p_7b_unav_baseline/checkpoint-1500 \
    NO_DEFAULT_IGNORE=true

# 다른 레포로
bash _tools/push_checkpoint.sh \
    CKPT=gdpo/exp_v1 \
    REPO_ID=ewha-team404/salmonn2p-gdpo
```

**자동 유도 규칙** (CKPT_DIR 기준 상대경로):
- 마지막 세그먼트 → `path_in_repo`
- 나머지 세그먼트들을 `_`로 결합 → repo 이름

**기본 IGNORE** (GRPO 백본/공유 용도에서 불필요한 파일들 자동 제외):
- `global_step*/*` — DeepSpeed ZeRO 옵티마이저 상태 (수~수십 GB)
- `latest`, `zero_to_fp32.py` — DeepSpeed 부속 파일
- `README.md` — PEFT 자동 생성, base_model에 로컬 경로가 박혀 HF 검증 실패

**주요 인자**: `CKPT`, `REPO_ID`, `REPO_NAME`, `PATH_IN_REPO`, `PRIVATE`, `MESSAGE`, `IGNORE`, `NO_DEFAULT_IGNORE`, `ALLOW`

### 4) LoRA Merge + HuggingFace 업로드

LoRA 어댑터를 베이스 모델에 merge한 **독립 실행 모델**로 올리고 싶을 때 (어댑터 분리 없이 `AutoModel.from_pretrained` 한 줄로 로드):

```bash
# merge만 (기본: 가장 최근 checkpoint)
bash _tools/sft/merge_and_push.sh MODEL_ID=salmonn2p_7b_unav_baseline CKPT=5000

# merge + HF 업로드
bash _tools/sft/merge_and_push.sh \
    MODEL_ID=salmonn2p_7b_unav_baseline \
    CKPT=5000 \
    PUSH=true PRIVATE=true \
    REPO_ID=ewha-team404/salmonn2p-unav-merged
```

**인증**: `paths.env`의 `HF_TOKEN`을 자동 사용. 또는 `huggingface-cli login`.

## 🛠 Config 파일

하이퍼파라미터는 각 파이프라인 폴더의 `config.yaml` 에서 관리합니다.

- `_tools/sft/config.yaml` — 학습 (LoRA, optimizer, batch 등)
- `eval/config.yaml` — 평가 (해상도, 프레임, deepspeed 등)

변경 없이 다른 실험을 돌리고 싶으면 `config.yaml` 을 복사해 수정 후 `CONFIG=my_config.yaml` 로 지정합니다.
