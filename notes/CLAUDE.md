# Project Handoff — Team4 TAVG Eval (2026-04-20 → 04-21 Handoff)

> 다른 Claude 세션이 이어받을 수 있도록 정리된 문서. 먼저 이 전체를 읽고 작업 시작할 것.

---

## 1. 프로젝트 전체 맥락

**연구 목표**: Audio-Visual Temporal Grounding (TAVG) — 비디오+오디오 입력 + 자연어 쿼리 → 해당 이벤트의 (start, end) 타임스탬프 예측.

**팀 중점 모델**: **video-SALMONN-2+ 7B** (Qwen2.5-VL-7B 백본) — SFT + **GDPO** 기반 align으로 TAVG에서 개선. 팀 레포: `AI-Living-Lab/Team4`, 브랜치 `cdh` 가 SALMONN-2+ 파이프라인 (`_tools/SFT/`, `eval/`).

**이 세션의 역할**: **Avicuna (Vicuna-7B, Audio-Visual VTG 선행 연구, AAAI 2025)** 를 reference baseline 으로 평가. SALMONN-2+와 **동일 eval 셋**에서 head-to-head 비교 준비.

**벤치마크**:
1. **LongVALE** — `/workspace/datasets/LongVALE`, 1171 videos / 13867 queries, 긴 서술형 캡션, median duration 199s
2. **UnAV-100 QA** — 팀이 만든 multiseg 포맷, cdh `data/unav100_test_full.json`, 3455 samples / 5900 GT segments, UnAV-100 비디오(평균 30–60s)

**작업 환경**:
- 작업 루트: `/workspace/jsy`
- GPU: A100 80GB (RunPod)
- conda envs:
  - `avicuna2` — Avicuna 트랙용
  - `jsy-vs2plus` — **video-SALMONN-2+ 트랙 전용** (사용자가 직접 만든 env; `cdh-salmonn2`는 다른 용도라 별도 생성)
  - cdh 브랜치 worktree: `/workspace/jsy/Team4-cdh` (paths.env 이미 설정됨)
- HF 캐시: `/workspace/jsy/hf_cache/` (루트 `/` 디스크 5GB 밖에 없어 workspace로 리다이렉트)

---

## 2. 완료된 평가 결과 (Avicuna)

### 2-1. Avicuna × LongVALE TVG — 1k subset 완료

**출력 디렉토리**: `/workspace/jsy/output/avicuna_longvale/`

| 메트릭 | 값 |
|---|---|
| Samples | 1000 (전체 13867의 subset) |
| **FSR** (format success rate, aux) | **99.5%** |
| **mIoU** (format fail = 0 IoU in mean) | **16.56%** |
| R@1 @ IoU=0.3 | 21.1% |
| R@1 @ IoU=0.5 | 14.3% |
| R@1 @ IoU=0.7 | 9.8% |

**진행 맥락 (의사결정 히스토리)**:
- 초기에 Avicuna stage4 훈련 분포 프롬프트 사용 → sanity check에서 공식 LongVALE 프롬프트와 pct 60% 달라짐 확인 → **공식 프롬프트로 재시작**
- 재시작 중 nohup 2개 동시 실행 사고로 predictions.jsonl 8458 dup 발생 → 프로세스 정리 후 subset 전략으로 전환
- 1k subset sequential(첫 1000) ~42분 소요, GPU 1장에서 완료
- 전체 13867은 ETA ~4h (아직 안 돌림) — 필요 시 next step 참조

**파일**:
- `predictions_subset1k.jsonl` — 실제 추론 결과
- `eval_subset1k.json` — 메트릭 요약
- `features/{video_clip,audio_clap}/` — 1171 비디오 CLIP+CLAP features (전체 13867 재실행 시 그대로 재사용 가능)
- `longvale_annotations.json` — Avicuna 포맷 변환 결과 (1171 videos)
- `sanity_official_prompt.json`, `verify_maxtokens.json` — 실험 기록 (프롬프트/토큰 수 영향 검증)
- `predictions_stage4prompt_dup.jsonl` — 아카이브 (stage4 프롬프트 + 중복, 삭제 가능)

### 2-2. Avicuna × UnAV-100 QA — Full 3455 완료

**출력 디렉토리**: `/workspace/jsy/output/avicuna_unavqa/`

| 메트릭 | 값 |
|---|---|
| Samples | 3455 |
| GT segments | 5900 |
| **FSR** (aux) | **100.0%** |
| **mIoU** (format fail = 0 IoU in mean) | **30.42%** |
| R@1 @ IoU=0.1 | 48.36% |
| R@1 @ IoU=0.3 | 35.51% |
| R@1 @ IoU=0.5 | 28.20% |
| R@1 @ IoU=0.7 | 22.92% |
| R@1 @ IoU=0.9 | 16.78% |

**진행 맥락**:
- Eval 셋은 cdh 브랜치 `data/unav100_test_full.json` — SALMONN-2+ 평가셋과 **동일**, head-to-head 비교 가능
- `<video>` 토큰 호환 확인 (Avicuna `DEFAULT_IMAGE_TOKEN = "<video>"`)
- UnAV-100은 Avicuna 학습 분포 내 → LongVALE보다 훨씬 좋음
- Features 재사용: `/workspace/jsy/Team4/AVicuna-main/data/unav100/features/{video_clip,audio_clap}/` (2167 unique videos)
- Duration 캐시: decord로 추출 (2167 → `durations.json`). pkgs ffprobe는 libtesseract5 누락으로 포기, apt 경로도 libtesseract5 Ubuntu 저장소 부재로 막힘. 10개 샘플 ffprobe(cdh env)와 교차검증 → 8/10 완벽 일치, 2/10 <35ms 차이 (VFR 추정, 무시 가능).

**파일**:
- `predictions.jsonl` — 3455 추론 결과 (`row_idx`, `vid`, `duration`, `gt_label`, `gt_segments`, `question`, `raw`)
- `eval_miou_summary.json` — 메트릭 요약
- `unav100_test_full.json` — cdh 브랜치 복사본
- `durations.json` — decord duration 캐시

### 2-3. 두 벤치마크 비교 요약 (Avicuna)

| | FSR | mIoU | R@1@0.3 | R@1@0.5 | R@1@0.7 |
|---|---|---|---|---|---|
| LongVALE (1k subset) | 99.5% | 16.6% | 21.1% | 14.3% | 9.8% |
| UnAV-100 QA (full) | 100.0% | **30.4%** | **35.5%** | **28.2%** | **22.9%** |

### 2-4. SALMONN-2+ base × UnAV-100 QA (hint ablation) — 완료

**V1 (no hint, 기존 native 프롬프트)**:
| 메트릭 | Strict (time-token `<t0><tdot>`) | Relaxed (seconds NL) |
|---|---:|---:|
| FSR | 0% | 1.65% |
| mIoU | 0% | 0.38% |

**V2 (+ hint: "Answer with the start and end timestamps of the event in seconds.")**:
| 메트릭 | Strict | Relaxed |
|---|---:|---:|
| FSR | 0% | **89.81%** |
| mIoU | 0% | **19.81%** |
| R@1@0.3 | 0% | 24.39% |
| R@1@0.5 | 0% | 14.53% |
| R@1@0.7 | 0% | 9.03% |

**Bucket 분포 변화 (V1 → V2)**:
- A no_numbers: 97.0% → **1.9%** (format 거의 완전 해결)
- B seconds_explicit: 1.7% → **98.0%**
- duration_exceeded flag: 0.8% → **51.7%** (V4 duration-aware 필요성 입증)

**V4 (duration-aware hint) 스킵 확정** — V2 결과로 motivation 방어 충분.

**출력 디렉토리**:
- `/workspace/jsy/Team4-cdh/outputs/base/vs2plus_7b_audio/unav100_test_full/` (V1)
- `/workspace/jsy/Team4-cdh/outputs/base/vs2plus_7b_audio/unav100_test_full_v2/` (V2)

### 2-5. SALMONN-2+ base × LongVALE 1k subset — 완료 (공식 프롬프트)

Avicuna 1k subset과 동일 샘플(첫 1000). LongVALE 공식 프롬프트 (`"From xx to xx"` format hint 이미 포함).

| 파서 | FSR | mIoU | R@1@0.3 | R@1@0.5 | R@1@0.7 |
|---|---:|---:|---:|---:|---:|
| **A: LongVALE official strict** (`\d{2}\s+(to|and)\s+\d{2}`) | **3.6%** | **0.21%** | 0.2% | 0% | 0% |
| **C: Relaxed** (seconds NL parser) | **91.7%** | **8.25%** | 8.2% | 5.1% | 2.9% |
| ~~B: Lenient~~ (폐기, 아래 파서 주의사항 §3-3 참조) | 99.7% | 16.81% | 22.6% | 12.2% | 5.4% |

**해석**:
- **A strict 0.21%는 localization 능력 부족이 아니라 format mismatch** (§7 "SALMONN strict mIoU 0은 format mismatch" 참조). SALMONN base는 공식 프롬프트의 `"From xx to xx"` 지시를 대부분 무시하고 자연어 "X seconds to Y seconds" 출력 (bucket B 96.5%). → C relaxed 8.25%가 실제 localization 능력을 보여줌.
- duration_exceeded flag 9.8% (UnAV V2의 51.7%보다 훨씬 낮음) — LongVALE 공식 프롬프트의 `"xx to xx"` 표현이 비디오 내부 좌표를 연상시켜 hallucination 억제 추정.

**출력 디렉토리**: `/workspace/jsy/Team4-cdh/outputs/base/vs2plus_7b_audio/longvale_test_subset1k/`

### 2-6. Head-to-head 비교 요약 (Union-IoU + FP/FN, 2026-04-23 재측정)

**지표 변경 (2026-04-23)**: 기존 Best-IoU → **Union-IoU** 통일 (cdh f7920fd 표준 채택) + **FP_rate / FN_rate** 병기. 저장된 pred 로 재 eval. 상세 `memory/union_iou_protocol_2026_04_23.md`.

**UnAV-100 full (3455 samples, 5900 GT segments)** — in-distribution:
| 모델 | 파서 | FSR | mIoU(union) | R@0.3 | R@0.5 | R@0.7 | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **ChronusOmni hybrid**§ | native `second{X}` | 100.00% | **36.78%** ★ | 43.61% | 33.49% | 25.58% | 5.46% | 22.00% |
| ARC-Hunyuan hybrid§ | `<answer>HH:MM:SS-HH:MM:SS</answer>` | 100.00% | 35.46% | 41.88% | 31.64% | 24.59% | **4.78%** ★ | **18.59%** ★ |
| Avicuna stage4 | native pct | 100.00% | 30.42% | 35.51% | 28.20% | 22.92% | 22.92% | 40.93% |
| SALMONN base V2 (+ hint) | NL parser | 97.2% | 16.89% | 21.31% | 12.08% | 6.22% | 28.34% | 38.17% |
| SALMONN base V1 (no hint) | NL parser | 1.68% | 0.28% | 0.32% | 0.15% | 0.12% | 18.64% | 98.93% |

§ ChronusOmni / ARC-Hunyuan 모두 **single-seg output by design** (훈련 타깃 구조적 제약) — multi-seg GT 대비 recall 상한. 상세: `memory/chronusomni_single_seg_2026_04_22.md`, `memory/arc_hunyuan_prep_2026_04_22.md`.

**측정 prompt (2026-04-22 hybrid 전략)**: 팀 UnAV wording 본문 + 모델 native format-tail. ChronusOmni tail `Output in the format of 'From second{start_time} to second{end_time}'.`, Crab+ tail `Please describe the events and time range...`, ARC-Hunyuan 은 build_prompt wrap 이 `<answer>` 포맷 강제. 상세: `memory/feedback_headtohead_prompts.md`.

**출력 위치**:
- ChronusOmni: `/workspace/jsy/outputs/base/ChronusOmni/Unav100QA/`
- ARC-Hunyuan: `/workspace/jsy/outputs/base/ArcHunyuan/Unav100QA/`
- Avicuna: `/workspace/jsy/output/avicuna_unavqa/eval_miou_summary.json`
- SALMONN V1/V2: `/workspace/jsy/Team4-cdh/outputs/base/vs2plus_7b_audio/unav100_test_full(_v2)/eval_miou_nl_summary.json`

**해석**:
- Union-IoU 재측정에서 **single-seg 모델은 Best-IoU 와 거의 동일** (multi-seg 이득 없음) — ChronusOmni/ARC-Hunyuan 숫자 불변, Avicuna 도 대부분 single-seg 출력이라 30.42% 유지
- **FP/FN 로 과장/놓침 분리 가능**: ARC-Hunyuan 이 FP 4.78% + FN 18.59% 로 가장 균형. Avicuna 는 FP/FN 둘 다 높음 (과장+놓침)
- Avicuna stage4(SFT'd) vs SALMONN base V2(hint only) gap ~**13.5%p (mIoU)** 가 SFT 의 localization 기여 상한. ChronusOmni / ARC-Hunyuan 의 RL 기반 학습이 Avicuna SFT-only 를 +5-6%p 초과

**LongVALE 1k subset** — OOD:
| 모델 | 파서 | FSR | mIoU(union) | FP | FN |
|---|---|---:|---:|---:|---:|
| Avicuna stage4 | LongVALE strict | 99.50% | **16.56%** | 54.47% | 54.70% |
| SALMONN base | NL parser | 96.6% | 7.76% | 62.53% | 63.80% |

→ OOD 에서 두 모델 모두 FP/FN 50%+ 로 고전. Avicuna 가 SALMONN 대비 약 2배 mIoU.

### 2-6-B. Head-to-head 2차 업데이트 (2026-04-23 오후, fair parser policy)

**원칙**: "공식 포맷 모델 → 각자 native parser 전용 / 공식 포맷 없는 foundation model → boosted NL parser" — 공정성 확보.

| 모델 | 포맷 | 파서 | 정책 |
|---|---|---|---|
| ChronusOmni | ✅ `second{X} to second{Y}` (training-baked) | native regex 만 | 그대로 |
| ARC-Hunyuan | ✅ `<answer><span>HH:MM:SS</span></answer>` (build_prompt wrap) | native regex 만 | 그대로 |
| Avicuna | ✅ `from XX to YY` pct | native pct parser | 그대로 |
| **SALMONN V1/V2** | ❌ 공식 포맷 없음 (base) | **boosted NL parser** | 재eval |
| **Qwen-Omni raw** | ❌ 공식 포맷 없음 (foundation) | **boosted NL parser** | V1 no-tail (진행 중) |
| Crab+ | ✅ `<event>{label}, (s e)</event>` | native | **skip** (10s cap) |

**재측정 결과 (boosted NL parser 사용, 기존 pred 재활용)**:
| 모델 | 파서 | FSR | mIoU(union) | R@0.3 | R@0.5 | R@0.7 | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SALMONN base V2 (+hint) | boosted | **97.16%** | **16.97%** | 21.90% | 11.85% | 5.85% | 27.31% | 36.15% |
| SALMONN base V1 (no hint) | boosted | 1.74% | 0.35% | 0.41% | 0.25% | 0.17% | 18.33% | **98.90%** |
| **Qwen-Omni raw V1 (no tail)** | **boosted** | **75.31%** | **11.24%** | **14.63%** | **9.05%** | **4.98%** | **28.21%** | **65.47%** |

**V1 vs V2 가르침 (SALMONN base)**:
- V1 FSR 1.74% — hint 없이는 **timestamp 거의 안 뱉음**. "no-hint 정책" 이 base SALMONN 엔 작동 안 함.
- V2 hint 한 문장만 줘도 FSR 97%+. base model 이라도 minimal format hint 필요함을 시사.
- 본문 footnote 필요: "SALMONN base evaluated with a minimal 'seconds' format hint to avoid trivial 0% FSR; all other baselines use their native prompt formats."

**Qwen-Omni raw 관찰 (2026-04-23 13:34 KST 완료)**:
- FSR 75.31%, mIoU 11.24% — 같은 Qwen2.5-VL backbone 으로 SALMONN aligner 덧붙인 V2 (16.97%) 대비 **-5.73%p**. audio-visual aligner 훈련이 TVG capability 에 실제 기여함을 확인.
- FN 65.47% — 전 모델 중 가장 높음. Foundation model 은 **"여러 event 나열" 구조적 약점**.

**UnAV-100 최종 순위 (mIoU, 2026-04-23 확정)**:
| 순위 | 모델 | mIoU | FSR | FP | FN |
|---:|---|---:|---:|---:|---:|
| 🥇 1 | ChronusOmni hybrid | **36.78%** | 100.0% | 5.46% | 22.00% |
| 🥈 2 | ARC-Hunyuan hybrid | 35.46% | 100.0% | 4.78% | 18.59% |
| 🥉 3 | Avicuna stage4 | 30.42% | 100.0% | 22.92% | 40.93% |
| 4 | SALMONN base V2 (+hint) | 16.97% | 97.2% | 27.31% | 36.15% |
| 5 | Qwen-Omni raw V1 (no tail) | 11.24% | 75.3% | 28.21% | 65.47% |
| 6 | SALMONN base V1 (no hint) | 0.35% | 1.7% | 18.33% | 98.90% |
| skip | Crab+ | — | — | — | — | 10s cap, `memory/crab_plus_limitation_2026_04_22.md` |
| skip | LongVALE-LLM / AVST-Zero | — | — | — | — | ckpt 미공개 |

**Family gap 분해 (backbone → aligner → SFT → RL)**:
- Raw foundation (Qwen-Omni) → **11.24%**
- + Audio-visual aligner (SALMONN V2 +hint) → **16.97%**  (Δ +5.73%p from aligner)
- + TVG SFT (Avicuna stage4) → **30.42%**  (Δ +13.45%p from SFT)
- + TVG SFT + RL (ChronusOmni / ARC-Hunyuan) → **35-37%**  (Δ +5-7%p from RL)
→ 단계별 기여도 가시화. 우리 팀 SFT+GDPO 목표는 ChronusOmni/ARC-Hunyuan 수준 (35%+).

**출력**:
- SALMONN V1 boosted: `/workspace/jsy/outputs/base/SALMONN_V1_boosted/eval_miou_summary.json`
- SALMONN V2 boosted: `/workspace/jsy/outputs/base/SALMONN_V2_boosted/eval_miou_summary.json`
- Qwen-Omni raw V1: `/workspace/jsy/outputs/base/QwenOmniRaw/Unav100QA/eval_miou_summary.json`
- Parser: `/workspace/jsy/scripts/eval_miou_nl_boosted.py`

### 2-7. Prior Work 3개 발견 — "첫 LLM-based TAVG" 주장 폐기

셋 다 TAVG 3조건 (A+V input, TVG-specific training, [t_start, t_end] output) 완전 만족:

| 모델 | 출처 | 핵심 구조 | 겹치는 축 |
|---|---|---|---|
| **ChronusOmni** | arXiv 2512.09841 (2025.12) | Interleaved timestamp + SFT + GRPO | RL 기반 AV-TVG, "closest structural peer" |
| **AVST-Zero** | arXiv 2511.16901 (2025.11) | Full GRPO on R-AVST, multi-dim rewards | GRPO + multi-dim reward (모달리티 분해인지 불명) |
| **TriSense** | NeurIPS 2025 (arXiv 2505.18110) | V+A+S, Query-Based Connector | modality priority 동적 선택 |

**살릴 수 있는 차별화 축 (조사 후 확정 필요)**:
1. Modality-decomposed reward의 구체 설계 (audio-only/visual-only/AV boundary 분해) — 3 prior work 중 동일 구조 있는지 확인 필수
2. Compositional time token (VTG-LLM style `<t0>~<t9>`, `<tdot>`) — TriSense `<sync><time>`, ChronusOmni "explicit textual timestamps"와 다름
3. Benchmark focus (UnAV-100 중심)

---

## 3. 확정된 세팅 (재실행/재현 필수)

### 3-1. Generation 세팅 (greedy, 1024 tokens)
```python
# scripts/inference_longvale.py :: generate_one
model.generate(
    input_ids,
    images=features,           # [(1,75,768), (1,25,512)]
    do_sample=False,           # GREEDY (확정)
    num_beams=1,
    max_new_tokens=1024,       # (확정) — 128은 LongVALE 긴 캡션에서 잘림
    use_cache=True,
    stopping_criteria=[stopping],
)
```
- `do_sample=True, temperature=0.05` (저자 데모 기본값)는 near-greedy이나 determinism 보장 안 됨 → greedy 채택
- 128 vs 1024 결과 bit-identical 검증 완료 (`verify_maxtokens.json`) — greedy이므로 stopping 기준만 달라지고 생성 내용 동일

### 3-2. 프롬프트 (확정)

**LongVALE TVG (공식 프로토콜)**:
```python
# scripts/inference_longvale.py :: build_prompt_query
f"<video>\nAt which time interval can we find {event_label} taking place in the video? Give the timestamps in the fromat: From xx to xx."
```
- 공식 리포 `longvalellm/eval/eval.py` 원문 그대로 (`fromat` 오타도 보존)
- `<video>\n` 직후 **공백 없음** (Avicuna stage4 훈련 데이터 분포 일치)
- 저자 데모 `"<video>\n " + query` (공백 1개)와 다르지만, 훈련 데이터를 따름

**UnAV-100 QA**:
- `unav100_test_full.json`의 `conversations[0].value` 를 **verbatim** 사용
- 이미 `<video>\n...` prefix 있음, 호환
- 질문이 row마다 다양 (템플릿 variation) — 그대로 사용

### 3-3. 파서 — **3종 구분 필수 (SALMONN 이후)**

| 파서 | regex / 동작 | 단위 가정 | 적용 대상 |
|---|---|---|---|
| **A (strict, LongVALE official)** | `r"(\d{2})\s+(to|and)\s+(\d{2})"` | pct 0–99 | **공식 비교, protocol 준수 여부** |
| **B (lenient)** | `r"(\d{1,3})\s*(?:to|-|and|~)\s*(\d{1,3})"` | pct 0–99 (가정) | **폐기 권장** — Avicuna 출력 특성상 A와 결과 동등하지만 SALMONN native "X seconds" 출력에서 **단위 오해석** (X초를 pct로 읽음) |
| **C (relaxed)** | `scripts/relaxed_parser.py` — seconds / mm:ss / approximate / single-ts 인식 | 원래 단위 그대로 | **diagnostic** — base zero-shot에서 모델이 실제 localize한 능력 측정 |

```python
# A (strict) 사용 예
import re
STRICT_RE = re.compile(r"(\d{2})\s+(to|and)\s+(\d{2})")
m = STRICT_RE.search(raw)
if m:
    start_pct = int(m.group(1))  # 0-99
    end_pct   = int(m.group(3))
    # 초 단위 변환: sec = pct * duration / 100
```

**Avicuna 경우**:
- 훈련 시 `convert()`가 항상 2자리 zero-pad로 포맷 → 출력이 항상 "XX to YY" 정수 pct → A/B 결과 동등 (0.1%p 이내)
- LongVALE 1k: strict 16.57% / lenient 16.55%
- UnAV full: strict 30.42% / lenient 30.42%

**SALMONN base 경우 — B가 위험** (2026-04-20 발견):
- native 출력: `"from 5 seconds to 45 seconds"` — 초 단위 의도
- B 파서가 `(5, 45)`를 pct로 해석 → `[5×dur/100, 45×dur/100]` = `[3s, 27s]` 등 단위 왜곡
- 결과적으로 LongVALE에서 B로 측정 시 mIoU 16.81% (우연히 Avicuna와 비슷) — **이 숫자 사용 금지**
- 올바른 비교:
  - A (protocol 준수): SALMONN base 0.21% (3.6% FSR) — pct 포맷 거의 생성 못함
  - C (native 단위 인식): SALMONN base 8.25% (91.7% FSR)
- **본문 리포트 기준**: A(strict) + C(relaxed) **병기**, B는 사용하지 않음

> **⚠️ 함정 사례 (future reference)**: B 파서는 단위 무시하고 숫자만 잡음. 모델이 seconds 출력한 걸 pct로 오해석해 **우연히 높은 mIoU** 나올 수 있음. 숫자가 비정상적으로 잘 나오면 **항상 파서 단위 가정 먼저 의심.** (오늘 LongVALE에서 B로 16.81% 찍혀 한참 혼선됨.)

**TODO (§5-2)**: `eval_longvale_tvg.py`/`eval_unavqa_tvg.py`의 PRED_RE/SEG_RE를 A로 교체 (재추론 불필요).

### 3-4. 평가 정책 (모든 모델/벤치마크 공통)

**Avicuna, video-SALMONN-2+, GDPO 등 모든 트랙에 동일 적용.**

1. **Format 실패 = 0 IoU로 평균에 포함** (표준 TVG 프로토콜)
   - parse 실패 / gen error / feature missing 전부 해당 row의 IoU를 0으로 기록하고 mIoU·R@1 분모에 포함
   - Avicuna: `eval_longvale_tvg.py`, `eval_unavqa_tvg.py` 양쪽 구현됨
   - SALMONN-2+: cdh `eval/eval_miou_multiseg.py`도 `if not pred_segments: all_ious.append(0.0)` 로 이미 구현됨 ✅

2. **Format Success Rate (FSR) = parse_ok / n_total 을 보조(aux) 메트릭으로 병기**
   - mIoU가 모델 성능 + 포맷 준수 혼합이라 FSR로 분리 가시화
   - Avicuna eval 스크립트: 완료 — summary JSON에 `FSR` / `FSR_%` 필드
   - SALMONN-2+ cdh 스크립트: **현재 `parse_ok`/`parse_fail` 카운트만 있음**. 다음 작업자가 `FSR_%` 필드 추가 필요 (한 줄 수정):
     ```python
     summary["FSR_%"] = round(parse_ok / max(len(raw_results), 1) * 100, 4)
     ```

### 3-5. Train-time resampling 재현 (inference_longvale.py::resample_video/audio)
- `av_ratio=0.25` 고정 (stage4.sh 훈련 설정): `n_image_feats=75`, `n_audio_feats=25`
- Video `(N, 768)`:
  - N≥75: `v[(j*N)//75]` for j in [0,75) (stride 샘플)
  - N<75: zero-pad
- Audio `(M, 512)`:
  - M==25: identity
  - 일반: `repeat_factor = 25 // M`, `remainder = 25 % M`, per-feature repeat
  - M>25: 첫 25개만 사용 (train 로직 그대로, information loss 있지만 의도적으로 유지)

---

## 4. 10개 Sanity Check 결과 요약

### `verify_maxtokens.py` — greedy determinism 검증
- 같은 10개 샘플 (길이 분포 다양), max_new_tokens 128 vs 1024 비교
- **10/10 모두 pct 값 bit-identical, raw 출력 prefix도 identical, 토큰 길이도 동일**
- 결론: greedy + 짧은 출력이라 token limit은 성공 케이스에 무영향. 실패 케이스(=긴 라벨 echo)만 1024로 살아남

### `sanity_official_prompt.py` — LongVALE 공식 vs Avicuna stage4 프롬프트
- 같은 10개 샘플에 두 프롬프트 각각 돌림
- **포맷은 10/10 동일** ("We can observe ... from XX to YY"), 파싱 10/10 성공
- **pct 값은 4/10만 일치 (6/10 달라짐)** — 프롬프트 wording이 모델 예측에 실제 영향
- IoU는 둘 다 낮음 (Avicuna의 LongVALE 분포 외 예측 한계, 프롬프트 문제 아님)
- 결론: **공식 프롬프트 채택 확정** (CLAUDE.md 방침 "프로토콜 차용" 준수)

### LongVALE 프롬프트 영향 상세 예시
| vid[:12] | len | stage4 pct | official pct |
|---|---|---|---|
| pjpObnTgOXo | 84 | (9,13) | (0,7) |
| cRIYcS6awZQ | 158 | (0,9) | (0,9) ✓ |
| Ji2vdGSxb6Y | 189 | (0,2) | (0,2) ✓ |
| JJPTYTswPCI | 201 | (11,13) | (24,31) |
| sMlvY8GLKi4 | 258 | (0,3) | (51,55) |

---

## 5. 다음 해야 할 일

### 5-0. 🚨 P0 — Prior Work 조사 + 팀 공유 (2026-04-21 우선)
1. ChronusOmni / AVST-Zero / TriSense PDF 정독 — 각 모델의 reward 설계 구체 확인
   - ChronusOmni "specially designed reward functions" 구체 내용 (모달리티별 분해?)
   - AVST-Zero "multi-dimensional rewards" = 모달리티 분해인지
   - TriSense Query-Based Connector가 reward에도 관여하는지 / 입력만인지
2. 각 모델이 UnAV-100 수치 리포트하는지 체크 — 없으면 직접 측정 필요
3. 네 contribution 차별화 축 확정 (§2-7 참조)
4. **은지/교수님께 prior work 발견 공유** — positioning 재설계는 팀 논의 필수

### 5-1. (선택) Avicuna × LongVALE 전체 13867 실행
- 현재는 1k subset만 완료. 전체 돌리려면:
  ```bash
  HF_HOME=/workspace/jsy/hf_cache HF_HUB_CACHE=/workspace/jsy/hf_cache/hub \
  TRANSFORMERS_CACHE=/workspace/jsy/hf_cache HF_HUB_OFFLINE=1 \
  nohup /workspace/miniconda3/envs/avicuna2/bin/python /workspace/jsy/scripts/inference_longvale.py \
      --annotation /workspace/jsy/output/avicuna_longvale/longvale_annotations.json \
      --video_feat_dir /workspace/jsy/output/avicuna_longvale/features/video_clip \
      --audio_feat_dir /workspace/jsy/output/avicuna_longvale/features/audio_clap \
      --output /workspace/jsy/output/avicuna_longvale/predictions_full.jsonl \
      --resume > /workspace/jsy/output/avicuna_longvale/inference_full.log 2>&1 &
  ```
- ETA ~4시간, GPU 1장 단독. **주의**: `--resume` 쓰기 전에 `pgrep -f inference_longvale.py` 로 다른 프로세스 없는지 꼭 확인 (이전 세션에 duplicate 사고 있었음)

### 5-2. Eval 스크립트를 strict parser로 업데이트
- `scripts/eval_longvale_tvg.py`, `scripts/eval_unavqa_tvg.py` 의 PRED_RE/SEG_RE를 `r"(\d{2})\s+(to|and)\s+(\d{2})"` 로 교체
- 재추론 필요 없음 (predictions.jsonl의 raw 필드만 재파싱)
- 결과 변화는 미미 (0.1%p 이내) 이지만 프로토콜 준수 차원에서 필요

### 5-3. **[SALMONN-2+ 트랙]** video-SALMONN-2+ 평가 (head-to-head)
- cdh 브랜치 파이프라인 사용: `Team4/eval/eval_salmonn2plus.sh`
- env: `cdh-salmonn2`
- 동일 `unav100_test_full.json` (3455) 으로 돌린 결과와 Avicuna 결과 직접 비교
- LongVALE 트랙은 SALMONN-2+ 공식 평가 스크립트 추가 구현 필요 (시간 토큰 `<t0>..<tdot>` 파서 조정)
- 결과는 cdh 규칙 따라 `Team4/outputs/base/...` 저장
- **평가 정책 (§3-4) 준수**: format fail = 0 IoU in mean (이미 구현됨), `FSR_%` 필드 추가 필요 (한 줄)

### 5-4. GDPO finetune 결과 평가
- SFT/GDPO 체크포인트 준비되면 동일 파이프라인 재사용 (CKPT_MODEL_ID, CKPT_STEP 인자만 변경)
- `bash _tools/SFT/train_salmonn2plus.sh MODEL_ID=... TRAINSET_FILE=unav100_train.json GPUS=...`
- `bash eval/eval_salmonn2plus.sh CKPT_MODEL_ID=... CKPT_STEP=5000 TESTSET_FILE=unav100_test_full.json`

### 5-5. 추가 baseline 측정 (positioning과 무관하게 진행)
- **Qwen2.5-Omni raw** (UnAV) — ✅ **완료 (2026-04-23): mIoU 11.24%, FSR 75.31%**
- **ChronusOmni** (UnAV) — ✅ **완료: mIoU 36.78%**
- **ARC-Hunyuan-Video-7B** (UnAV) — ✅ **완료: mIoU 35.46%**
- **Crab+** — ❌ skip (10s cap 구조적 한계, `memory/crab_plus_limitation_2026_04_22.md`)
- **LongVALE-LLM / AVST-Zero** — ❌ skip (ckpt 미공개)
- TimeChat (UnAV + LongVALE) — (선택, 우선순위 낮음)
- Avicuna 신 JSON 재측정 (신 wording + no tail) — 사용자 2026-04-23 "skip" 확정 (현 30.42% 유지)
- Prior work 논문 본문 reward 설계 분석 — P0 (다음 세션)

### 5-6. 자투리 정리
- `outputs/base/vs2plus_7b_audio/longvale_test_subset1k/eval_miou_summary.json` (UnAV strict 잘못 적용됨, **삭제**)
- `outputs/base/.../longvale_test_subset1k/predictions_salmonn_longvale_style.jsonl` (B 파서용 어댑터 출력, **삭제 또는 `_WRONG` 접미사**)
- `outputs/base/.../longvale_test_subset1k/eval_salmonn_subset1k.json` (B 파서 결과, **삭제 또는 `_WRONG` 접미사**)
- 본문 draft용 핵심 숫자 표 한 장 정리 (§2-6 기반)

---

## 5-X. 미해결 / 의심 포인트 (체크 필요)

아직 확정 안 됐지만 놓치면 안 되는 것들 — 내일 작업 시작 전에 훑어볼 것.

1. **LongVALE relaxed 8.25% sanity check 미완**
   - 논의했던 "짧은 비디오만 필터" 등 부분집합 분석 안 돌림. 긴 비디오에서 relaxed 파서가 잘못 잡는 케이스 있는지 확인 필요.
   - TODO: duration bucket별 mIoU 분포, outlier 샘플 수동 검토.

2. **Prior work 3개 reward 설계 상세 불명**
   - ChronusOmni "specially designed reward", AVST-Zero "multi-dim rewards", TriSense Query-Based Connector — 셋 다 PDF 정독 전엔 modality-decomposed인지 단순 composite인지 **결론 유예 상태**.
   - 이 조사 전까진 네 차별화 축 (§2-7) 중 어느 게 살아남을지 불확실.

3. **B 파서 혼선 재발 방지**
   - 오늘 SALMONN LongVALE에서 B 파서로 16.81% 찍혀 한참 "이상하다" 헷갈렸음. 근본 원인: lenient regex가 단위 가정을 무시.
   - 새 모델/새 벤치마크 결과 리포트 전 **단위 가정 first check** 체크리스트화 필요.

4. **SALMONN V2 duration hallucination (51.7%) 미조사**
   - V4 스킵 결정했지만 hallucination이 어느 길이 구간에서 나오는지 (짧은 비디오만? 긴 비디오도?) 분포 분석 안 함.
   - 본문에 hallucination rate 언급 시 기초 통계 있으면 좋음.

5. **FSR_% 필드 cdh 스크립트 추가 미수행**
   - §3-4에서 언급한 한 줄 패치 아직 적용 안 함. SALMONN SFT/GDPO 결과 리포트 시작 전 반드시 반영.

---

## 6. 주요 파일 레이아웃

```
/workspace/jsy/
├── CLAUDE.md                           # 이 문서
├── data_0418/                          # 원본 grounding.json 등 (현재 미사용)
├── output/
│   ├── avicuna_longvale/
│   │   ├── predictions_subset1k.jsonl  # 1k subset 결과
│   │   ├── eval_subset1k.json
│   │   ├── features/                   # CLIP+CLAP 1171
│   │   ├── longvale_annotations.json   # Avicuna 포맷
│   │   ├── sanity_official_prompt.json
│   │   └── verify_maxtokens.json
│   └── avicuna_unavqa/
│       ├── predictions.jsonl           # 3455 결과
│       ├── eval_miou_summary.json
│       ├── unav100_test_full.json      # cdh 복사본
│       └── durations.json              # decord 2167
├── scripts/
│   ├── convert_longvale_to_avicuna.py  # LongVALE → Avicuna 포맷
│   ├── convert_longvale_to_salmonn.py  # LongVALE → SALMONN-2+ 포맷 (2026-04-20)
│   ├── cache_unav_durations.py         # decord duration 캐시
│   ├── inference_longvale.py           # ★ 메인 LongVALE 추론 (Avicuna)
│   ├── inference_unavqa.py             # ★ 메인 UnAV-100 QA 추론 (Avicuna)
│   ├── eval_longvale_tvg.py            # LongVALE eval (단일 seg, R@1+mIoU)
│   ├── eval_unavqa_tvg.py              # UnAV eval (multiseg best-IoU, cdh 스타일)
│   ├── verify_maxtokens.py             # 128 vs 1024 검증
│   ├── sanity_official_prompt.py       # 프롬프트 영향 sanity
│   ├── make_hint_variant.py            # V2/V3/V4 hint JSON 생성기 (2026-04-20)
│   ├── classify_outputs.py             # bucket 분류 (py3.10+ type hints, 원본)
│   ├── classify_outputs_run.py         # bucket 분류 (py3.9 실행판)
│   ├── relaxed_parser.py               # ★ C 파서 (자연어 + 단위 인식)
│   ├── apply_relaxed_to_v1.py          # 기존 결과에 C 파서 적용
│   └── salmonn_to_longvale_jsonl.py    # SALMONN → Avicuna 스타일 jsonl 어댑터
├── Team4/                              # 팀 레포 (브랜치: jsy_gpu_avicuna)
│   └── AVicuna-main/                   # Avicuna 모델 코드
│       ├── checkpoints/                # stage1~4 + clip/clap
│       └── data/unav100/features/      # UnAV-100 기존 feature (재사용)
└── hf_cache/                           # Vicuna-7B 등 HF 모델 캐시
```

---

## 7. 주요 구현 결정 (근거 정리)

- **Avicuna 학습 분포 vs 공식 프로토콜 충돌**: stage4 훈련 프롬프트는 Avicuna에 유리하지만 LongVALE 벤치마크 비교에는 부적절. 공식 프로토콜 채택 (CLAUDE.md 원칙).
- **Greedy 선택**: 저자 데모 `do_sample=True, temp=0.05` 대신 `do_sample=False` — determinism 필요 (재현/디버깅). temp=0.05는 near-greedy라 실제 차이 미미.
- **max_new_tokens 1024**: 저자 데모 기본값과 동일. 128은 긴 LongVALE 캡션에서 "from X to Y" 생성 전에 잘림 (~1.3% 실패). 검증: greedy이므로 1024가 128의 strict superset.
- **UnAV feature 재사용**: 이미 추출된 기존 UnAV 100 features (video/audio) 그대로 사용. 재추출 불필요.
- **Duration 도구**: decord 채택. Ubuntu ffmpeg apt 저장소에 없음, pkgs ffprobe는 libtesseract5(Ubuntu libtesseract4만) 누락으로 막힘. decord는 avicuna2 env 기본 의존성, ffprobe와 교차검증 통과.
- **Multi-seg eval (UnAV-100 QA)**: cdh `eval_miou_multiseg.py`의 best-IoU-per-GT-segment 로직 차용. Avicuna는 보통 1 segment만 내놓지만 "From X to Y. From X to Y." 로 멀티세그도 가능. 팀 표준과 호환.
- **중복 프로세스 사고**: nohup `--resume` 만으로 동시 실행 방지 불가 (파일 lock 없음). **재시작 전 `pgrep -f ...` 필수 체크.**
- **V4 (duration-aware hint) 스킵 (2026-04-20)**: V2 결과(FSR 89.81%, relaxed mIoU 19.81%)로 "base는 SFT 없이 못 한다" motivation 방어 충분. V4 추가 ablation은 cost 대비 이득 미미.
- **"첫 LLM-based TAVG" 주장 폐기 (2026-04-20)**: ChronusOmni / AVST-Zero / TriSense 3개 prior work 발견. 그러나 **base zero-shot + hint ablation 실험은 계속 유효** — "base는 SFT+GDPO 없이 못 한다"가 방어 대상이고 prior work와 무관.
- **LongVALE = OOD 표 용도 확정 (2026-04-20)**: Avicuna/SALMONN 모두 학습 분포 외라 성능 낮음. 본문에서 "OOD generalization 제한" 서사로 활용.
- **SALMONN strict mIoU 0은 format mismatch (2026-04-20)**: base 모델의 localization 능력 부족이 아니라 예상 format (pct XX to XX) 출력 불가 때문. 본문에 이 구분 명시 필수.

---

## 8. 환경 / 재현 주의사항

- `HF_HUB_OFFLINE=1` 꼭 설정 (network 에러 방지, 이미 캐시된 Vicuna-7B 재다운로드 방지)
- `HF_HOME=/workspace/jsy/hf_cache` 등 `/workspace` 로 리다이렉트 (루트 디스크 5GB 불충분)
- Python은 반드시 `/workspace/miniconda3/envs/avicuna2/bin/python` 절대 경로 사용 (conda activate 이슈 방지)
- 재시작 시 `pgrep -f inference_*.py` 로 기존 프로세스 확인 후 `pkill` 또는 wait

**참고 리포**:
- Avicuna: `/workspace/jsy/Team4/AVicuna-main/` + https://github.com/yunlong10/AVicuna
- LongVALE: https://github.com/ttgeng233/LongVALE (`longvalellm/eval/eval.py`, `eval/metric.py`)
- video-SALMONN-2+ 팀 파이프라인: origin/cdh 브랜치 (`_tools/SFT/`, `eval/eval_miou_multiseg.py`)
- Prior work (2026-04-20 발견):
  - ChronusOmni arXiv 2512.09841
  - AVST-Zero arXiv 2511.16901
  - TriSense arXiv 2505.18110 (NeurIPS 2025)

---

## 9. 논문 본문 문구 초안 (2026-04-20 추가)

**UnAV-100 V2 hint ablation**:
> "We test a format hint ('Answer with start/end timestamps in seconds.') applied to base video-SALMONN 2+. FSR rises from 1.65% to 89.81% and relaxed mIoU from 0.38% to 19.81%. However, 51.7% of predictions exceed the video duration, and residual mIoU remains substantially below AVicuna (30.42%). This demonstrates that format is not the sole bottleneck — localization capability itself requires training."

**LongVALE OOD**:
> "On LongVALE, base video-SALMONN 2+ achieves 0.21% mIoU under the official strict parser due to format mismatch (native seconds output vs. expected pct format). Under a relaxed parser interpreting native-unit outputs, mIoU is 8.25% — still substantially below AVicuna's 16.56%, indicating format resolution alone does not close the OOD temporal localization gap."
