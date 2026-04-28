---
name: Crab+ UnAV-100 구조적 한계 — 10s cap (저자 의도, our fault 아님)
description: Crab+ 는 10s clip 분포로 훈련됨. 원본 UnAV-100 full-length 측정 시 [0,10] bounded 출력 → 구조적 low mIoU. 논문 limitation 명시 확정
type: project
originSessionId: 7cf60908-ffa2-425c-9445-c628fd4bc0af
---
**결정 (2026-04-22)**: Crab+ × UnAV-100 측정은 V3 hybrid prompt + **as-is** (chunking 적용 안 함). 낮은 mIoU 는 structural limitation 으로 본문에 명시.

**Why (저자 의도 증거):**
1. Crab+ repo `AVUIE_2/unav/train.json` 4925 라벨 **전수 분석**: max_end ≤ 10s, 69% 가 정확히 10 (전혀 10 초과 없음)
2. 원본 UnAV-100 해당 비디오 실측: 평균 30–60s (e.g. `bHfesK3rBE4` 38s, `TOEwQXeyNe8` 44s, `6Ehj3Egq2VI` 60s) — 저자가 clip/normalize 한 것
3. Crab+ paper (arXiv 2603.04128) Sec 5.1: 저자 본인이 "we train on a sampled subset of clips with stable temporal dynamics" 라고 명시
4. Crab+ paper Table 2: UnAV-100 test 성능 **리포트 안 함** (AVE 10s + AVVP 10s 만 리포트)
5. `OmniTestDataset` 에 UnAV 용 `add_unav_samples` 메서드 없음 (저자가 공식 test set 구성 안 함)
6. `inference_omni.sh` supported_tasks 에 'unav' 없음 → 공식 측정 skip

**V3 sanity 10 관측 (us):**
- FSR 10/10 — Crab+ native `<event>{label}, (s e)(s e)</event>` 100% 준수
- 하지만 모든 예측 [0, 10] bounded:
  - male singing GT max 40s → pred (0, 10)
  - people cheering GT [23.5, 25.9] → pred (5, 6)  (4x scale mismatch)
  - driving motorcycle GT [0, 29] → pred (0, 10)
  - people crowd GT [0, 10][22, 37][41, 44] → pred (4, 5)

**Chunking (옵션 B) 폐기 이유:**
- 저자가 UnAV 를 chunked 로 쓰지 않음 → chunking 은 우리가 모델을 "저자 의도 너머" 로 쥐어짜는 것
- Unfair comparison 논쟁 여지
- Hallucination 리스크 (AVE/AVVP 훈련 분포는 "event 항상 존재하는 10s clip" 전제)

**논문 limitation 문구 초안:**
> Crab+ [Cai et al., 2026] is trained on AVE (4,143 clips of 10s each, 1-second grid annotation) and AVVP (10-second clips with segment-level annotations) for temporal localization. For UnAV-100, the authors explicitly state that they "train on a sampled subset of clips with stable temporal dynamics" (Sec. 5.1). We verified this in their released training data: all 4,925 UnAV-100 instruction samples have max_end ≤ 10s, with 69% exactly at 10. Notably, the original paper does not report Crab+ performance on the UnAV-100 benchmark (Table 2, which covers AVE and AVVP only).
>
> When we evaluate Crab+ on the full-length UnAV-100 test set (average 42s), predictions are structurally confined to the [0, 10] range regardless of actual video duration, leading to near-zero IoU for events beyond the 10-second mark. This reflects a training-time design choice by the Crab+ authors, not a prompt engineering issue. We include Crab+ in our comparison for completeness but note its results as reference values given this fundamental distribution mismatch with long-video TVG evaluation.

**How to apply:** 비교표에 Crab+ mIoU 숫자 옆 footnote "structural [0,10] cap — trained on clipped distribution". 추가로 GT max_end ≤10s vs >10s split breakdown 병기 (`eval_miou_multiseg_crab_plus.py` 에 이미 구현). ≤10 split 에서 Crab+ 가 정상 성능 나오면 limitation 주장 수치 근거.

**저자 인용:** arXiv 2603.04128, "Crab+: A Scalable and Unified Audio-Visual Scene Understanding Model with Explicit Cooperation", Cai et al., 2026. Gaoling School of AI, Renmin University of China. Code: https://github.com/GeWu-Lab/Crab_Plus.

**측정 세팅 (고정):**
- Prompt: hybrid = 팀 wording body + Crab+ native AVE/UnAV tail ("Please describe the events and time range...")
- 추론: `scripts/inference_crab_plus.py` (cdh 표준 output), nframes=10 유지
- 파서/Eval: `scripts/eval_miou_multiseg_crab_plus.py` (`<event>{L}, (s e)...</event>` + fuzzy label filter)
- 출력: `/workspace/jsy/outputs/base/CrabPlus/Unav100QA/` (test_results_rank0.json + eval_miou_summary.json + watcher.log)

**V3 vs V1 (사전 검증):** V3 (hybrid) 가 V1 (team only) 대비 포맷 안정성 압도 — V1 은 `<answer>`/`<range>`/`<audio_event>` 혼재 + "six" 같은 텍스트 답 섞임. V3 로 고정.
