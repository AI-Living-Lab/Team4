---
name: Prompt 전략 — 팀 wording + 모델 native format-tail
description: prior-work/base-model baseline = 팀 UnAV wording 유지 + 맨 뒤에 각 모델 native output-format hint 만 append
type: feedback
originSessionId: 8bfff642-2202-4b4d-89cf-726ca869ecfd
---
TAVG 벤치마크 (UnAV-100 등) 에서 prior work / base model baseline 은 **팀 질문 wording 을 그대로 유지하고**, **맨 끝에 각 모델의 native output-format hint 만 append** 해서 측정한다. 완전 verbatim(no hint) 은 FSR 0% 로 측정 무의미. 완전 native template 은 task framing 자체가 달라 UnAV "both" 의미 왜곡.

**Why:** (a) 완전 verbatim (hint 제거) — ChronusOmni 에서 sanity 30/30 FSR 0% 관측, 측정 무의미. (b) 완전 native template (예: ChronusOmni LongVALE grounding `"At which time interval..."`) — task framing 이 "in the video" 단독 시각 중심이라 UnAV 의 "both video and audio" 의도를 지움. (c) **하이브리드 — 질문 본문은 팀 wording 유지(task 의미 보존), tail 만 모델 출력 포맷 맞춤** — 가장 방어 가능: reviewer 에게 "task 조건은 동일, output 계약만 맞춰 줬다" 라고 설명 가능.

**How to apply:**
- **ChronusOmni** (UnAV): 팀 질문 본문 `"At what point in the video does {event} occur in terms of both video and audio?"` + tail `" Output in the format of 'From second{start_time} to second{end_time}'."`
  - `<video>\n` 프리픽스는 제거 (ChronusOmni eval.py 가 `<speech><image>\n` 자체 주입)
- **SALMONN base V2** (이미 완료): 팀 wording + `"Answer with the start and end timestamps of the event in seconds."` — 2026-04-20 측정 (relaxed mIoU 19.81%)
- **Crab+** (추후 측정 예정): 팀 wording + native tail `"Please describe the events and time range that occurred in the video."` 또는 `<event>{label}, (s e)</event>` 형태 tail 조정
- **Avicuna** (이미 완료): 현재 30.4% 는 (구) 팀 wording 으로 측정 — 신 wording (`"At what point in the video does ..."`) + 하이브리드 tail 로 재측정 필요 (Avicuna stage4 포맷 `XX to YY` pct tail 매칭)
- **팀 SFT+GDPO 모델**: 팀 wording **그대로** (hint 없이) — train/test 분포 일치

**이전 정책에서 변경점 (2026-04-22):**
- 구 정책 = "prior work 는 각 모델의 native training template 전체 사용"
- 신 정책 = "팀 wording + native format-tail 만" (hybrid)
- 이유: native template 전체 사용은 task framing 자체 변경 — "visual segment" 또는 "video only" 로 의미 왜곡. 팀 "both video and audio" 의미 보존하려면 본문은 팀 것 유지하고 tail 만 바꿔야 fair.

**논문 본문 기재 문구 초안:**
"Each prior-work baseline is evaluated with the team UnAV-100 question wording extended by one format-hint sentence matching that model's native output grammar. The task framing is held constant; only the output-format contract is adapted per model."
