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

`paths.example.env` 를 복사하여 `paths.env` 를 만들고 각 서버 경로에 맞게 수정:

```bash
cp paths.example.env paths.env
# BASE_ROOT / DATA_ROOT 두 루트만 잡으면 나머지는 자동 파생. WANDB_API_KEY 채우기.
```

> ⚠️ `paths.env` 는 `.gitignore` 대상(개인 키 포함) — 커밋 금지. 템플릿만 `paths.example.env` 로 관리.

| 변수 | 의미 |
|---|---|
| `BASE_ROOT`    | 프로젝트/데이터 JSON 의 공통 루트 |
| `DATA_ROOT`    | 체크포인트·원본 데이터(영상 등)의 공통 루트 |
| `BASE_DIR`     | 이 레포 루트 (`${BASE_ROOT}/master/Team4`) |
| `CKPT_DIR`     | 체크포인트 루트 (`${DATA_ROOT}/checkpoints`) |
| `DATA_DIR`     | UnAV-100 등 원본 데이터 (`${DATA_ROOT}/unav_100`) |
| `TRAIN_DIR`    | 학습 JSON (`TRAINSET_FILE` 검색 위치) |
| `TEST_DIR`     | 평가 chunk 디렉토리 |
| `EVAL_DIR`     | 평가 결과 (`${BASE_ROOT}/outputs`) |
| `WANDB_ENTITY` / `WANDB_API_KEY` | WandB 로그 (학습 시) |

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
├── data/                                              # ${BASE_ROOT}/data
│   ├── train/unav100_sft.json                         # ${TRAIN_DIR}
│   ├── test/{TESTSET}/chunk_*.json                    # ${TEST_DIR}
│   └── debug_interleave_samples.json
├── _tools/
│   ├── sft/                                           # SFT 학습 (train_salmonn2plus.sh) + time-token 추가
│   ├── GDPO/                                          # RL(GDPO) 학습 — GDPO학습방법총정리 문서 참고
│   ├── tti/                                           # TTI 회귀 검증 (run_all.sh)
│   └── debug/                                         # debug_interleave dump
└── eval/
    ├── eval.sh                                        # 통합 런처 (추론+resume / 평가)
    ├── eval_miou.py                                   # 통합 평가기 (3종 summary JSON)
    ├── _chunk_helpers.py                              # 청크 split/append/resume
    ├── config.yaml
    └── 통합평가방법총정리(26-06-08).md                # 평가 상세 가이드
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

### 2) RL 미세조정 (GDPO)

SFT 정책을 temporal-IoU reward 로 강화학습. 상세는
[_tools/GDPO/GDPO학습방법총정리(26-06-04).md](_tools/GDPO/GDPO학습방법총정리(26-06-04).md) 참고.

```bash
python _tools/GDPO/gdpo_trainer.py \
    --config       _tools/GDPO/config.yaml \
    --model_path   ${CKPT_DIR}/sft/salmonn2p_7b_unav_fps5_off \
    --model_base   ${CKPT_DIR}/video_salmonn2_plus_7B_time_tokens \
    --dataset_path ${TRAIN_DIR}/unav100_v2.json
```

### 3) 평가

통합 런처 `eval/eval.sh` + 평가기 `eval/eval_miou.py` 로 추론·평가를 수행한다.
인자/출력 규칙 전체는 [eval/통합평가방법총정리(26-06-08).md](eval/통합평가방법총정리\(26-06-08\).md) 참고.

```bash
cd eval

# base+LoRA, 단일 JSON 추론+평가
bash eval.sh STAGE=sft CKPT_MODEL_ID=salmonn2p_7b_unav_fps5_off CKPT_STEP=1500 \
     TEST_JSON=${TEST_DIR}/unav100_v2_500.json GPUS=0

# 추론 모드만 다르게 (config 보다 우선)
bash eval.sh STAGE=sft CKPT_MODEL_ID=salmonn2p_7b_unav_fps5_off CKPT_STEP=1500 \
     TEST_JSON=${TEST_DIR}/unav100_v2_500.json TTI_TIME_FORMAT=natural_text

# 베이스 모델만
bash eval.sh CKPT_STEP=base TEST_JSON=${TEST_DIR}/unav100_v2_500.json

# 이미 추론된 결과만 재평가 (GPU 불필요)
bash eval.sh MODE=eval RESULTS=<out_dir>/test_results_rank0.json TEST_JSON=<GT>.json
```

> LoRA→base 머지는 `eval.sh` 가 추론 직전 **자동** 수행(청크 시 `.merged_model/` 캐시 재사용)한다.
> 별도 merge/push 스크립트는 없다.

**주요 인자**: `MODE`, `CHUNK`, `STAGE`, `CKPT_MODEL_ID`, `CKPT_STEP`, `BASE_MODEL_ID`,
`MERGED_MODEL`, `TEST_JSON`/`TESTSET`, `GPUS`, `TTI_TIME_FORMAT`, `NATURAL` (전체 표는 평가 문서 §3).

**결과 저장 위치**: `${EVAL_DIR}/<branch>/fps<N>_<format>/<TESTSET_TAG>/` —
같은 체크포인트를 여러 추론 모드로 비교하면 `fps<N>_<format>` 태그가 달라 결과가 분리됨.

## 🛠 Config 파일

하이퍼파라미터는 각 파이프라인 폴더의 `config.yaml` 에서 관리.

| 파일 | 용도 | TTI 관련 키 |
|---|---|---|
| `_tools/sft/config.yaml`  | SFT 학습 (LoRA, optimizer, batch, …) | `BASE_INTERVAL`, `TTI_TIME_FORMAT` |
| `_tools/GDPO/config.yaml` | GDPO(RL) 학습 (reward, clip, num_generations, …) | `tti_mode` |
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
