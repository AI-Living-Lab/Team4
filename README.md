# Team4 — video-SALMONN-2+ UnAV-100 Pipeline

오디오-비주얼 TVG multiseg Task + Time-Token Interleaving (TTI) 적용버전의
video-SALMONN-2+ 학습/평가 파이프라인.

## 📁 UnAV-100 Dataset

### 🎬 Raw Video / Features
- [Raw Videos](https://drive.google.com/drive/folders/1YtKugZrNJ8iCEtncyMCPdQ1Qdrfal9_U)


### 📄 학습/평가 JSON

| 경로 | 용도 |
|---|---|
| `data/train/unav100_sft.json`        | SFT 학습 (`TRAINSET_FILE` 인자) |
| `data/test/unav100/chunk_*.json`     | chunk 단위 평가 (`TESTSET=unav100`) |
| `data/test/unav100/_full.json`       | 전체 평가셋 (mIoU 계산용 — 자동 생성) |
| `data/debug_interleave_samples.json` | 디버그 dump 용 샘플 8개 |

> chunk 단위 평가 json은 `python3 eval/_chunk_helpers.py split --test_json <원본> --chunks_dir data/test/<NAME>/` 로 생성.

## 🐟 video-SALMONN-2+ Setup

Repository reference: [video-SALMONN-2 GitHub](https://github.com/bytedance/video-SALMONN-2)

### 📥 Download Checkpoints
1. **video-SALMONN2_plus_7B_full-** — [huggingface](https://huggingface.co/tsinghua-ee/video-SALMONN2_plus_7B_full)
2. **베이스 + time tokens (pre-baked)**: `_tools/sft/add_time_tokens_salmonn2plus.py` 로 11개 time-token 추가 후
   `${CKPT_DIR}/base/video_salmonn2_plus_7B_time_tokens` 에 저장.

### 🔧 Environment Variables

`paths.env.example` 를 복사하여 `paths.env` 를 만들고 수정:

```bash
cp paths.env.example paths.env
# 필요 시 HF_TOKEN 채우고, BASE_DIR / DATA_DIR / CKPT_DIR 경로 조정
```

| 변수 | 의미 |
|---|---|
| `BASE_DIR`     | 이 레포 루트 |
| `DATA_DIR`     | UnAV-100 등 외부 데이터 (`/workspace/datasets`) |
| `CKPT_DIR`     | 체크포인트 루트 (`/workspace/checkpoints`) |
| `JSON_DIR`     | `${BASE_DIR}/data` |
| `TRAIN_DIR`    | `${JSON_DIR}/train` — 학습 JSON (`TRAINSET_FILE` 검색 위치) |
| `TEST_DIR`     | `${JSON_DIR}/test` — 평가 chunk 디렉토리 |
| `EVAL_DIR`     | `${BASE_DIR}/outputs` — 평가 결과 |
| `HF_HOME`      | HuggingFace 캐시 |
| `HF_TOKEN`     | HF Hub 업로드용 (선택) |

### 📂 Directory Structure

```
Team4/
├── checkpoints/                                       # ${CKPT_DIR}
│   ├── base/video_salmonn2_plus_7B_time_tokens/       # pre-baked 베이스
│   ├── sft/{MODEL_ID}/                                # SFT 산출물
│   │   ├── checkpoint-00500/ ...
│   │   └── config.used.yaml                           # 재현용 스냅샷
│   └── gdpo/{MODEL_ID}/
├── outputs/                                           # ${EVAL_DIR}
│   ├── sft/{MODEL_ID}/{checkpoint-N}/{EVAL_TAG}/{TESTSET}/
│   │   ├── eval_miou_summary.json
│   │   ├── test_results_rank0.json
│   │   └── inference.log
│   └── base/{BASE_MODEL_ID}/{EVAL_TAG}/{TESTSET}/     # CKPT_STEP=base
├── data/                                              # ${JSON_DIR}
│   ├── train/unav100_sft.json                         # ${TRAIN_DIR}
│   ├── test/{TESTSET}/chunk_*.json                    # ${TEST_DIR}
│   └── debug_interleave_samples.json
├── _tools/
│   ├── sft/                                           # SFT 학습 + LoRA merge 도구
│   ├── tti/                                           # TTI 회귀 검증 (run_all.sh)
│   └── debug/                                         # debug_interleave dump
└── eval/
    ├── eval_salmonn2plus.sh                           # chunked 평가 + 자동 resume
    ├── eval_3mode_chain.sh                            # off/special_token/natural_text 순차 평가
    ├── config.yaml
    └── eval_miou_multiseg.py
```

## 🎯 Time-Token Interleaving (TTI)

비디오/오디오 청크 사이에 시간 마커를 끼워넣어 temporal grounding 성능을 높이는 실험.
`tti_time_format` 플래그로 3가지 모드 지원:

| 모드             | 청크당 마커 | 예시 (1.5s)         | 설명 |
|---|---|---|---|
| `off` (기본)     | 0 토큰    | (없음)              | Qwen2.5-VL 베이스라인 |
| `special_token` | 6 토큰    | `<t0><t0><t0><t1><tdot><t5>` | VTG-LLM 식 special token |
| `natural_text`  | 9 토큰    | `second{0001.5}`    | ChromosOmni 식 자연어 (zero-pad) |

> 출력(GT) 형식은 모드와 무관 — 항상 `<tD><tD><tD><tD><tdot><tD>` (베이스라인 모델이 time-token 임베딩을 갖고 있음).

검증: `bash _tools/tti/run_all.sh ${CKPT_DIR}/base/video_salmonn2_plus_7B_time_tokens` (7/7 PASS 면 OK)

## 🚀 Usage

모든 런처는 `KEY=VALUE` 인자, 생략 시 기본값 적용.

### 1) SFT 학습

```bash
bash _tools/sft/train_salmonn2plus.sh \
    MODEL_ID=salmonn2p_7b_unav_fps5_off \
    TRAINSET_FILE=unav100_sft.json \
    GPUS=0,1
```

**지원 인자**: `STAGE` (sft/gdpo), `MODEL_ID`, `TRAINSET_FILE`, `BASE_MODEL_ID`, `GPUS`, `CONFIG`

**MODEL_ID 네이밍 규칙**: `<base>_fps<N>_<off|natural|tti>`
- `<N>` = `round(1 / config.BASE_INTERVAL)`
- `<format tag>` ↔ `config.TTI_TIME_FORMAT` 매핑:
  `off → off`, `natural_text → natural`, `special_token → tti`
- 불일치 시 학습 스크립트가 친절한 경고 메시지로 권장 이름을 출력 (학습은 진행).

예) `BASE_INTERVAL=0.1, TTI_TIME_FORMAT=special_token` → `salmonn2p_7b_unav_fps10_tti`

**저장 위치**: `${CKPT_DIR}/${STAGE}/${MODEL_ID}/checkpoint-N/`

### 2) LoRA Merge + HuggingFace 업로드

```bash
bash _tools/sft/merge_and_push.sh \
    MODEL_ID=salmonn2p_7b_unav_fps5_off \
    CKPT=5000 \
    PUSH=true PRIVATE=true \
    REPO_ID=ewhaailab/salmonn2p-unav
```

인증: `huggingface-cli login` 또는 `HF_TOKEN=hf_xxx`.

### 3) 평가

```bash
# 가장 최근 체크포인트
bash eval/eval_salmonn2plus.sh CKPT_MODEL_ID=salmonn2p_7b_unav_fps5_off

# 특정 step
bash eval/eval_salmonn2plus.sh CKPT_MODEL_ID=salmonn2p_7b_unav_fps5_off CKPT_STEP=1500

# 추론 모드만 다르게 (config.yaml override)
bash eval/eval_salmonn2plus.sh \
    CKPT_MODEL_ID=salmonn2p_7b_unav_fps5_off CKPT_STEP=1500 \
    TTI_TIME_FORMAT=natural_text

# 베이스 모델만
bash eval/eval_salmonn2plus.sh CKPT_STEP=base
```

**지원 인자**: `STAGE`, `CKPT_MODEL_ID`, `CKPT_STEP`, `BASE_MODEL_ID`, `TESTSET`, `GPUS`, `CONFIG`,
`TTI_TIME_FORMAT` (config.yaml 보다 우선)

**결과 저장 위치** (EVAL_TAG = `fps<N>_<format>` 자동 계산):
- LoRA: `outputs/<stage>/<MODEL_ID>/<checkpoint-N>/<EVAL_TAG>/<TESTSET>/`
- Base: `outputs/base/<BASE_MODEL_ID>/<EVAL_TAG>/<TESTSET>/`

같은 체크포인트를 여러 추론 모드로 비교하면 EVAL_TAG 가 달라 결과가 분리됨.

### 4) 3-mode 추론 chain

같은 체크포인트로 `off`, `special_token`, `natural_text` 세 모드를 순차 평가.

```bash
cd /workspace/tti_natural/Team4
setsid nohup bash eval/eval_3mode_chain.sh \
    STAGE=sft \
    CKPT_MODEL_ID=salmonn2p_7b_unav_fps5_off \
    CKPT_STEP=1500 \
    BASE_MODEL_ID=base/video_salmonn2_plus_7B_time_tokens \
    TESTSET=unav100 GPUS=0 \
    < /dev/null > /tmp/eval_3mode_chain.log 2>&1 &
disown
```

각 모드 결과는 다른 EVAL_TAG 폴더에 분리 저장 (`fps5_off/`, `fps5_tti/`, `fps5_natural/`).
한 모드 실패해도 다음 모드는 계속 진행.

## 🛠 Config 파일

하이퍼파라미터는 각 파이프라인 폴더의 `config.yaml` 에서 관리.

| 파일 | 용도 | TTI 관련 키 |
|---|---|---|
| `_tools/sft/config.yaml` | 학습 (LoRA, optimizer, batch, …) | `BASE_INTERVAL`, `TTI_TIME_FORMAT` |
| `eval/config.yaml`        | 평가 (해상도, 프레임, deepspeed) | `BASE_INTERVAL`, `TTI_TIME_FORMAT` |

다른 실험은 `cp config.yaml my_config.yaml` 후 수정해서 `CONFIG=my_config.yaml` 로 지정.

## 🧪 Debug / 검증

```bash
# 3 모드 모두 8개 샘플 dump (json + txt)
bash _tools/debug/smoke_dump_all_modes.sh

# 특정 모드의 sweep (BASE_INTERVAL × VIDEO_MAX_FRAMES 조합)
TTI_TIME_FORMAT=special_token OUT_BASE=_debug_out/sweep_special_token \
    bash _tools/debug/sweep_dump.sh

# sweep 결과 비교 csv/md 생성
python _tools/debug/compare.py --in_dir _debug_out/sweep_special_token \
    --format csv --out _debug_out/sweep_special_token/compare.csv

# TTI rope/dataset/modeling 회귀 검증 (7개)
bash _tools/tti/run_all.sh ${CKPT_DIR}/base/video_salmonn2_plus_7B_time_tokens
```
