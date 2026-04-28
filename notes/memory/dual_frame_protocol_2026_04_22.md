---
name: 이중 시간 해상도 프로토콜 — (A) 저자 권장 vs (B) 팀 기본
description: 모든 모델 측정 시 프레임/fps 셋업을 두 가지로 수행. (A) 각 모델 저자 권장 세팅, (B) 팀 기본 (MAX 128 / MIN 64 / fps=2) 통일 세팅
type: feedback
originSessionId: 7cf60908-ffa2-425c-9445-c628fd4bc0af
---
TAVG 벤치마크 측정 시 추론 time-resolution 을 **두 가지 프로토콜로 병기**:

### (A) 모델별 저자 권장 (author-recommended)
각 모델이 훈련/평가 시 실제로 쓴 프레임·fps 설정 그대로.

| 모델 | 권장 프레임 세팅 |
|---|---|
| Avicuna | 최대 100 프레임 (stage4 훈련 시 75 video + 25 audio features stride-sampling) |
| ChronusOmni | `frame_num=64` (저자 eval.py 기본값) |
| Crab+ | `nframes=10` (저자 `omni_dataset.py:654-694` hardcoded, UnAV train 레이블이 0-10s 분포인 이유) |
| SALMONN-2+ (우리 팀 모델) | 이미 팀 기본 = 128/64/2fps |
| **Qwen2.5-Omni backbone 기반** (ARC-Hunyuan, Crab+ 등) | **≤384초: 2fps / >384초: 768 프레임** (Qwen team guideline) |

### (B) 팀 기본 통일 (team-standard)
- `MAX_FRAMES = 128`
- `MIN_FRAMES = 64`
- `fps = 2`

UnAV-100 평균 42s → 84 frames @ 2fps (팀 기본 MIN/MAX 범위 내).

**Why:** (a) (A) 단독 리포트 시 "모델마다 다른 프레임 수로 측정했다" 공정성 지적 여지. (b) (B) 단독 리포트 시 "저자 권장 분포 밖이라 capability 저평가" 반박. (c) **두 개 병기** 가 표준 practice — capability 측정 + 통제된 비교 둘 다 방어 가능.

### 현재 상황 매핑 (2026-04-22):

**이미 (A) 로 측정 완료:**
- Avicuna stage4: 75/25 features stride-sample ≈ 저자 권장 ≈ (A)
- ChronusOmni: frame_num=64 ≈ 저자 기본 ≈ (A)
- SALMONN base V2 (+hint): 팀 세팅이라 이건 사실상 (B) — 다른 모델에도 동일 적용하는 standard

**(A) 로 진행 예정:**
- Crab+: nframes=10 (저자 권장) — 현재 runner 그대로. **10s cap limitation 논문 명시 예정**
- ARC-Hunyuan: sample_fps-based (최대 150f) — 저자 default

**(B) 재측정 여부 결정 대상:**
- Avicuna, SALMONN, ChronusOmni 까지 포함할지 — 사용자 확인 필요
- Crab+ (B) 재측정 — nframes 증가가 [0,10] cap 해소하는지 empirical check 할 가치 있음

**How to apply:**
- 새 모델 추론 스크립트 작성 시 **프레임/fps 셋업을 argparse flag 로 노출** (예: `--frame_policy author|team`)
- 논문 본문: (A) 값을 primary 로, (B) 값을 ablation 으로 or 병기 표로
- ARC-Hunyuan 시작 시 우리 runner 에서 `nframes` 파라미터화 필수

**기록일:** 2026-04-22, Crab+ full run 준비 중 사용자 제공
