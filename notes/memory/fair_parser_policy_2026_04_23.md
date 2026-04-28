---
name: Fair parser policy — 공식 포맷 유무로 파서 선택 구분
description: 공식 출력 포맷 있는 모델 → 각자 native regex 만, 공식 포맷 없는 foundation/base → boosted NL parser. 혼용 금지
type: feedback
originSessionId: 7cf60908-ffa2-425c-9445-c628fd4bc0af
---
TAVG baseline 측정 시 파서 선택 원칙 (2026-04-23 사용자 피드백 반영):

### 원칙
- **공식 출력 포맷 있는 모델** (훈련 시 고정된 format 학습) → **각자 native regex 만** 사용
- **공식 포맷 없는 foundation/base 모델** → **boosted NL parser** (`scripts/eval_miou_nl_boosted.py`) 사용

### 이유
공식 포맷 모델에 boosted parser 를 쓰면:
1. 여러 regex 중복 매칭 → 같은 구간을 여러 번 카운트 (FP 부풀림)
2. "여러 규칙 중 하나라도 걸리면 성공" 식 FSR 과장
3. 훈련 분포 따르도록 학습된 모델에게 관대한 조건 부여 → **unfair**

### 모델별 적용
| 모델 | 포맷 | 파서 |
|---|---|---|
| ChronusOmni | `second{X} to second{Y}` (SFT baked) | native regex (`eval_miou_multiseg_chronusomni.py`) |
| ARC-Hunyuan | `<answer><span>HH:MM:SS</span></answer>` (build_prompt wrap) | native regex (`eval_miou_multiseg_arc_hunyuan.py`) |
| Avicuna | `from XX to YY` pct | native pct (`eval_miou_multiseg_avicuna.py`) |
| Crab+ | `<event>{label}, (s e)</event>` | native (`eval_miou_multiseg_crab_plus.py`) |
| SALMONN base V1/V2 | 없음 (base model) | **boosted NL** (`eval_miou_nl_boosted.py`) |
| Qwen-Omni raw | 없음 (foundation) | **boosted NL** (`eval_miou_nl_boosted.py`) |

### Hint 정책 (보완)
cdh 팀 "no hint" 정책은 "공식 포맷 없는 foundation" 에만 엄격 적용. SALMONN base 는 hint 없으면 FSR 1.74% (거의 0) 이라 **minimal seconds hint 필요** — 본문에 명시:
> "SALMONN base evaluated with a minimal 'seconds' format hint to avoid trivial 0% FSR; all other baselines use their native prompt formats."

### How to apply
- 새 모델 측정 시 먼저 "훈련 시 고정 format 있나?" 체크
- 있으면 저자 repo 에서 native regex 가져와 파서 작성
- 없으면 `eval_miou_nl_boosted.py` 사용
- 혼용 금지 — 같은 모델에 두 파서 동시 적용 X

### 재측정 필요 시 (기존 pred 재활용)
추론 재실행 없이 파서만 바꿔 재eval 가능. 5분 이내.

**기록일**: 2026-04-23, Qwen-Omni raw full run 병행 중 사용자 피드백.
