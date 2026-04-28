# RunPod Baseline 측정 — `_tools/baseline_eval/`

> 2026-04-22 ~ 04-23 RunPod A100 80GB 환경에서 진행한 prior-work baseline 측정 작업 일체.
> 우리 팀 video-SALMONN-2+ SFT/GDPO 결과와 head-to-head 비교 위한 reference baseline.
> 저자 jsy + Claude 협업.

---

## 측정 결과 요약 (UnAV-100 full 3455 samples, Union-IoU + FP_rate + FN_rate)

| 순위 | 모델 | mIoU(union) | R@0.3 | R@0.5 | R@0.7 | FSR | FP | FN |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 🥇 1 | ChronusOmni hybrid | **36.78%** | 43.61 | 33.49 | 25.58 | 100% | 5.46% | 22.00% |
| 🥈 2 | ARC-Hunyuan hybrid | 35.46% | 41.88 | 31.64 | 24.59 | 100% | **4.78%** | **18.59%** |
| 🥉 3 | Avicuna stage4 (LongVALE tail) | 30.42% | 35.51 | 28.20 | 22.92 | 100% | 22.92% | 40.93% |
| 4 | Avicuna stage4 (JSON tail) | 27.97% | 33.98 | 23.93 | 17.71 | 70.51% | 43.63% | 40.25% |
| 5 | SALMONN base V2 (+hint) | 16.97% | 21.90 | 11.85 | 5.85 | 97.16% | 27.31% | 36.15% |
| 6 | Qwen-Omni raw V1 (no tail) | 11.24% | 14.63 | 9.05 | 4.98 | 75.31% | 28.21% | 65.47% |
| 7 | SALMONN base V1 (no hint) | 0.35% | 0.41 | 0.25 | 0.17 | 1.74% | 18.33% | **98.90%** |
| skip | Crab+ | — | — | — | — | — | — | — | 10s training cap (구조적 한계) |
| skip | LongVALE-LLM / AVST-Zero | — | — | — | — | — | — | — | 체크포인트 미공개 |

각 모델별 자세한 측정 세팅은 아래 폴더 참조.

---

## 폴더 구조

```
_tools/baseline_eval/
├── README.md                    ← 이 파일
├── inference/                   ← 6개 추론 runner (모델별)
├── eval/                        ← 8개 평가 스크립트 (Union-IoU + FP + FN)
├── convert/                     ← 6개 변환기 (팀 JSON → 각 모델 입력 포맷)
├── watchers/                    ← 5개 watcher 스크립트 (추론 종료 후 자동 eval)
├── utils/                       ← 3개 데이터 prep 유틸
└── test_inputs/                 ← 5개 변환된 test JSON (재현용)
```

추가 위치:
- `Team4/notes/CLAUDE.md` — 측정 작업 마스터 노트 (의사결정 히스토리 전부)
- `Team4/notes/memory/*.md` — 11개 의사결정 메모 (정책/한계/관찰)
- `Team4/outputs/baseline_eval_runpod/` — 각 모델 eval_miou_summary.json

---

## inference/ — 모델별 추론 runner

전부 cdh 표준 출력 contract 준수: `test_results_rank0.json` (list, test JSON 순서 유지, 각 row 에 `pred` 필드).

| 파일 | 모델 | 비고 |
|---|---|---|
| `inference_chronusomni.py` | ChronusOmni (Ola-7b + SFT+GRPO) | 저자 `inference/eval.py` plumbing 우리가 cdh 표준으로 wrap. 출력: `From second{X} to second{Y}` |
| `inference_arc_hunyuan.py` | ARC-Hunyuan-Video-7B (Hunyuan-7B VL + Whisper-v3) | flash_attention_2, stopping_criteria on `</answer>`, 저자 build_prompt 사용. 출력: `<answer><span>HH:MM:SS - HH:MM:SS</span></answer>` |
| `inference_crab_plus.py` | Crab+ (Qwen2.5-Omni-7B + I-LoRA) | sdpa, peft_hyper I-LoRA 3-way routing 적용. 출력: `<event>{label}, (s e)(s e)</event>` |
| `inference_qwen_omni_raw.py` | Qwen2.5-Omni-7B raw (foundation, LoRA 없음) | sdpa, Crab+ 와 동일 backbone but I-LoRA 미적용. 출력: 자유 자연어 |
| `inference_unavqa.py` | Avicuna stage4 (Vicuna-7B + CLIP+CLAP features) | 기존 확장. test JSON 의 `conversations[0].value` 를 verbatim 으로 사용 |
| `inference_longvale.py` | Avicuna stage4 — LongVALE 측정용 | 공식 LongVALE prompt + 1k subset |

### 공통 사용 패턴
```bash
# 예시: ChronusOmni full 3455 launch
cd /workspace/jsy/Chronus  # checkpoint 심볼릭 링크 위치 (./checkpoints 참조)
python inference_chronusomni.py \
    --json_file /path/to/test.json \
    --output_dir /path/to/output \
    --model_path ./checkpoints
```

---

## eval/ — 평가 스크립트

전부 cdh `eval_miou_multiseg.py` 인터페이스 준수: `--results --test_json --max_time --out_dir`.
모두 `eval_utils.py` (Union-IoU + FP_rate + FN_rate 공통 라이브러리) 사용.

| 파일 | 대상 | 파서 |
|---|---|---|
| `eval_utils.py` | 모든 eval 공통 lib | Union-IoU compute, FP/FN counting, summarize/print helpers |
| `eval_miou_multiseg_chronusomni.py` | ChronusOmni | `second\{X\}\s*(?:to\|-)\s*second\{Y\}` 정규식 |
| `eval_miou_multiseg_arc_hunyuan.py` | ARC-Hunyuan | `<answer>` 블록 안 `HH:MM:SS - HH:MM:SS` |
| `eval_miou_multiseg_crab_plus.py` | Crab+ | `<event>{L}, (s e)(s e)</event>` + fuzzy label 매칭 + 10s cap breakdown |
| `eval_miou_multiseg_avicuna.py` | Avicuna (LongVALE tail) | `from XX to YY` pct → seconds (× duration / 100) |
| `eval_miou_multiseg_avicuna_json.py` | Avicuna (JSON tail) | JSON list 파싱 + `event` fuzzy match + `timestamps "from X to Y"` |
| `eval_miou_nl_boosted.py` | SALMONN base / Qwen-Omni raw 등 NL 출력 | from-to / dash / starts-until / first-N / throughout / HH:MM:SS / between 7개 패턴 |
| `eval_longvale_tvg.py` | Avicuna LongVALE 1k subset | 공식 LongVALE strict regex |

### 파서 정책 (중요)
- **공식 출력 포맷 있는 모델** (ChronusOmni, ARC-Hunyuan, Crab+, Avicuna): **각자 native parser 만**. boosted parser 사용 금지 (혼용 시 unfair advantage)
- **공식 포맷 없는 foundation/base 모델** (SALMONN base, Qwen-Omni raw): **boosted NL parser** 사용
- 자세한 정책: `notes/memory/fair_parser_policy_2026_04_23.md` 참조

### 사용 예시
```bash
python eval_miou_multiseg_chronusomni.py \
    --results /path/to/test_results_rank0.json \
    --test_json /path/to/test.json \
    --max_time 60 \
    --out_dir /path/to/output
# → output/eval_miou_summary.json 생성 (mIoU_union_%, Recall_%, FP_rate_%, FN_rate_%, FSR_% 포함)
```

---

## convert/ — 입력 데이터 변환기

팀 UnAV-100 / LongVALE JSON → 각 모델 inference 가 요구하는 입력 포맷.
**Hybrid 정책 적용**: 팀 question wording 본문 + 각 모델 native format-tail.

| 파일 | 변환 |
|---|---|
| `convert_unav_to_chronusomni.py` | 팀 UnAV multiseg JSON + ChronusOmni LongVALE-style tail (`Output in the format of 'From second{start_time} to second{end_time}'.`) |
| `convert_unav_to_arc_hunyuan.py` | 팀 UnAV + ARC-Hunyuan native (build_prompt wrap 이 자동 추가, tail 없음) |
| `convert_unav_to_crab_plus.py` | 팀 UnAV + Crab+ AVE/UnAV native tail (`Please describe the events and time range...`) |
| `convert_unav_to_qwen_omni.py` | 팀 UnAV + neutral seconds hint (foundation, no native) |
| `convert_longvale_to_avicuna.py` | LongVALE 공식 → Avicuna stage4 입력 |
| `convert_longvale_to_salmonn.py` | LongVALE 공식 → SALMONN-2+ 입력 |

### 정책 메모
- 자세한 hybrid 전략 history: `notes/memory/feedback_headtohead_prompts.md`
- Avicuna 는 LongVALE-tail 과 JSON-tail 두 변형 측정 (single-seg vs multi-seg capability 비교)

---

## watchers/ — 자동 eval 트리거

추론 PID 종료 시 polling 으로 감지 → eval 자동 실행 → `eval_miou_summary.json` 저장.
Pod 자리 비울 때 / 모바일 관찰 시 유용.

| 파일 | 모델 | 사용 |
|---|---|---|
| `wait_and_eval_chronusomni.sh` | ChronusOmni | `bash wait_and_eval_chronusomni.sh <PID>` |
| `wait_and_eval_arc_hunyuan.sh` | ARC-Hunyuan | 동일 |
| `wait_and_eval_crab_plus.sh` | Crab+ | 동일 |
| `wait_and_eval_qwen_omni_raw.sh` | Qwen-Omni raw | 동일 |
| `wait_and_eval_avicuna_json.sh` | Avicuna (JSON variant) | 동일 |

---

## utils/ — 데이터 prep 유틸리티

| 파일 | 용도 |
|---|---|
| `cache_unav_durations.py` | UnAV-100 video duration 캐시 생성 (decord). Avicuna inference 가 `durations.json` 필요 |
| `extract_wav_longvale.sh` | LongVALE mp4 → 48kHz mono wav 일괄 추출 (ffmpeg) |
| `run_extract_features_longvale.sh` | LongVALE 비디오 → CLIP+CLAP feature 추출 (Avicuna 입력용) |

재현 시: 새 환경에서 데이터 처음 셋업 할 때 사용.

---

## test_inputs/ — 변환된 테스트 JSON (재현용)

| 파일 | 내용 |
|---|---|
| `unav100_test_avicuna_hybrid.json` | 팀 UnAV 3455 + Avicuna LongVALE tail |
| `unav100_test_avicuna_json_hybrid.json` | 팀 UnAV 3455 + Avicuna JSON native tail |
| `unav100_test_chronusomni_hybrid.json` | 팀 UnAV 3455 + ChronusOmni native tail |
| `unav100_test_arc_hunyuan.json` | 팀 UnAV 3455 (build_prompt wrap 으로 ARC-Hunyuan 형식 변환) |
| `unav100_test_qwen_omni_raw.json` | 팀 UnAV 3455 + tail 없음 (foundation cdh 정책) |

원본 팀 JSON: `Team4-cdh/data/unav100_test_multiseg_salmonn2plus.json` (cdh 브랜치).

---

## 의존성 / 환경 (RunPod A100 80GB 기준)

### conda envs
| env | 용도 | 핵심 패키지 |
|---|---|---|
| `avicuna2` | Avicuna 추론 | torch, decord, CLIP/CLAP encoders |
| `chronusomni` | ChronusOmni 추론 | torch 2.3.0+cu118, flash_attn 2.6.0, transformers 4.49.0 |
| `crab` | Crab+ / Qwen-Omni raw 추론 | torch 2.5.1+cu121, transformers 4.51.3, flash-attn 2.7.4.post1 (prebuilt wheel) |
| `archunyuan` | ARC-Hunyuan 추론 | torch 2.5.1+cu121, **transformers fork** `liyz15/transformers@arc_hunyuan_video`, flash-attn 2.7.4.post1 |

### 모델 체크포인트 (Git ignore — `/workspace/models/` 또는 별도 위치)
| 모델 | 위치 | 크기 |
|---|---|---|
| Vicuna-7B-v1.5 (Avicuna backbone) | `/workspace/models/vicuna-7b-v1.5/` | 13GB |
| Qwen2.5-VL-7B (SALMONN backbone) | `/workspace/models/qwen25vl_7b/` | 16GB |
| Qwen2.5-Omni-7B (Crab+, Qwen-Omni raw) | `/workspace/models/Qwen2.5-Omni-7B/` | 21GB |
| video-SALMONN-2_plus_7B_full (SALMONN base) | `/workspace/models/vs2plus_7b_audio/` | 18GB |
| ARC-Hunyuan-Video-7B | `/workspace/jsy/hf_cache/archunyuan/` | 17GB |
| ChronusOmni | `/workspace/jsy/hf_cache/chronusomni_ckpt/` (HF gated 미공개 ckpt) | 20GB |
| Crab+ I-LoRA | `/workspace/jsy/Crab_Plus/weight/finetune_weights.bin` | 1.74GB |

각 모델 다운로드 출처 / 자세한 셋업 절차는 `notes/memory/*.md` 의 `arc_hunyuan_prep_*.md`, `crab_plus_feasibility_*.md` 등 참조.

---

## 작업 히스토리 / 의사결정 히스토리

`Team4/notes/CLAUDE.md` (마스터 노트, §1–§9 전부) 와 `Team4/notes/memory/*.md` (11개) 에 모든 의사결정 / 측정 결과 / limitation 명시 기록.

핵심 메모:
- `union_iou_protocol_2026_04_23.md` — Union-IoU + FP/FN 통일 결정
- `fair_parser_policy_2026_04_23.md` — 공식 포맷 모델 / foundation 별 파서 분리 정책
- `feedback_headtohead_prompts.md` — Hybrid prompt 전략 (팀 body + 모델 native tail)
- `chronusomni_single_seg_2026_04_22.md` — ChronusOmni 구조적 single-seg 한계
- `arc_hunyuan_prep_2026_04_22.md` — ARC-Hunyuan 셋업
- `crab_plus_limitation_2026_04_22.md` — Crab+ 10s training cap 한계
- `avicuna_unav_eval_file.md` — UnAV-100 공용 eval 파일 위치 변경

---

## Reproducibility — 재현 가이드

### 1단계: 데이터 셋업
```bash
# UnAV-100 듀레이션 캐시
python utils/cache_unav_durations.py
# (durations.json 이 Avicuna inference 시 필요)
```

### 2단계: test JSON 변환
```bash
# 예: ChronusOmni 용 변환
python convert/convert_unav_to_chronusomni.py \
    --src /path/to/unav100_test_multiseg_salmonn2plus.json \
    --dst /tmp/test_chronusomni.json
```

### 3단계: 추론
```bash
cd /workspace/jsy/Chronus  # 모델 위치
python /path/to/inference/inference_chronusomni.py \
    --json_file /tmp/test_chronusomni.json \
    --output_dir /tmp/output \
    --model_path ./checkpoints
```

### 4단계: 평가
```bash
python eval/eval_miou_multiseg_chronusomni.py \
    --results /tmp/output/test_results_rank0.json \
    --test_json /tmp/test_chronusomni.json \
    --max_time 60 \
    --out_dir /tmp/output
# → /tmp/output/eval_miou_summary.json
```

또는 watcher 사용 (자동화):
```bash
nohup python /path/to/inference_chronusomni.py ... > /tmp/output/inference.log 2>&1 &
INFER_PID=$!
nohup bash watchers/wait_and_eval_chronusomni.sh "$INFER_PID" > /tmp/output/watcher.out 2>&1 &
# infer 종료 시 자동으로 eval 실행됨
```

---

## 주요 한계 / Limitation

각 모델별 측정 시 한계가 본문 footnote 로 명시되어야 함:

1. **ChronusOmni / ARC-Hunyuan**: single-seg output by design (훈련 타깃 구조적). multi-seg GT 대비 recall 상한.
2. **Avicuna LongVALE tail**: tail 이 single 형식 ("From xx to xx") 이라 multi-seg capability 억제 — 0/3455 multi-seg 출력.
3. **Avicuna JSON tail**: multi-seg unleashed (71.7% multi-event) 이나 30% narrative fallback 으로 FSR 70%.
4. **SALMONN base**: hint 없으면 FSR 1.7% (timestamp 거의 안 뱉음). minimal "seconds" hint 필수.
5. **Crab+**: 저자가 UnAV-100 을 10s 클립으로 훈련 → predictions 전부 [0, 10] 범위 bounded. 우리 full-length test (avg 42s) 와 distribution mismatch. 본문 reference 만, 비교표 제외.
6. **LongVALE-LLM, AVST-Zero**: 모델 체크포인트 미공개 → 재학습 비현실적, skip.
7. **TriSense**: 측정 안 함 (시간 부족).
8. **Qwen-Omni raw**: foundation model 한계 — FN 65.47% (다수 GT 놓침). multi-event 인식 약함.

---

## 작업 환경 / 도구
- 작업 위치: RunPod A100 80GB
- HuggingFace dataset 백업: `ewha-team404/team4_tavg_baselines_0423` (private, 1.3MB tarball)
- 마스터 노트: `Team4/notes/CLAUDE.md`
- 작업자: jsy (serena2140@ewha.ac.kr) + Claude (협업)

---

## 추가 작업 예정 (TODO)

- [ ] Crab+ full 3455 측정 (현재 sanity 만, 10s cap 한계로 우선순위 낮음)
- [ ] Avicuna labelset prompt 측정 (저자 inference_labelset_full.py, UnAV 100-class 주입)
- [ ] LongVALE OOD 비교 (현재 1k subset, full 13867 으로 확장 가능)
- [ ] TimeChat baseline (시간 있으면)
- [ ] 모델별 multi-seg output 분석 (현재 SALMONN V2 만 9.6% multi, 나머지 < 1%)
