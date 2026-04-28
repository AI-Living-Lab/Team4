---
name: ChronusOmni 는 구조적 single-segment 모델 (multi-seg 불가)
description: ChronusOmni 훈련 데이터/템플릿 구조상 한 쿼리에 1 segment 만 출력. UnAV-100 multi-seg GT 대비 recall 구조적 제약
type: project
originSessionId: 7cf60908-ffa2-425c-9445-c628fd4bc0af
---
**결정 (2026-04-22)**: ChronusOmni × UnAV-100 측정 시 **single-segment 출력으로 그대로 측정**하고, multi-seg recall 제약을 본문 limitation 으로 명시. 2-4h 추가 투자해서 multi-seg prompt variant 돌리는 대신 1번 측정으로 정리.

**근거 — ChronusOmni 는 구조적으로 single-seg 모델:**
ChronusOmni repo `data/test/*.json` 의 grounding 타깃 9종 중 **timestamp-only multi-seg 타깃은 0개**:
- `v2t_openqa` / `a2t_openqa` target = `second{X}-second{Y}` (single)
- `longvale_test_grounding` target = `From second{X} to second{Y}.` (single)
- `longvale_test_captioning` 만 multi-listing 이지만 event 설명 동반 (pure timestamp 아님)

→ Single-seg 학습만 받은 모델. Prompt tail 수정해도 multi-seg 출력 기대 어려움.

**우리 측정 prompt (팀 wording + ChronusOmni single-seg tail):**
```
At what point in the video does {event} occur in terms of both video and audio? Output in the format of 'From second{start_time} to second{end_time}'.
```

**Sanity 10 관측:**
- FSR 10/10 = 100%
- GT 1-seg: pred 정확히 겹침 (e.g. `[0,11.67]` → `From second{0.0} to second{11.6}`)
- GT multi-seg: union/전체/첫-seg 로 뭉쳐 출력 (e.g. 3-seg GT → `From second{0.0} to second{47.0}` 전체)
- mIoU 31.5% (표본 10, noise 큼)

**논문 본문 문구 (초안):**
> "ChronusOmni outputs a single (start, end) pair per query, a structural constraint inherited from its training targets which contain no pure-timestamp multi-segment examples. On multi-segment UnAV-100 queries it merges segments into a single coverage span, yielding a lower bound on multi-segment recall. This distinguishes it from AVicuna (which can emit multiple 'from X to Y' fragments) and from our team model (which emits N segments via `<t>` token listing)."

**How to apply:** head-to-head 표에 mIoU 숫자 옆 footnote: "single-seg output by design". Multi-seg 측정 variant 돌리는 건 향후 ablation 에만 (현재 우선순위 아님).
