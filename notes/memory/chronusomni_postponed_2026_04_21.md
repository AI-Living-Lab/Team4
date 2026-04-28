---
name: ChronusOmni UnAV-100 측정 일시 보류 (2026-04-21)
description: ChronusOmni 의 grounding eval template 3종 모두 UnAV-100 의 audio-visual "both" framing 과 구조적 불일치. AVST-Zero 선행 후 재검토
type: project
originSessionId: 7cf60908-ffa2-425c-9445-c628fd4bc0af
---
ChronusOmni × UnAV-100 측정을 **2026-04-21 일시 보류**. 재개 조건은 AVST-Zero 결과 확정 후 재검토.

**보류 이유 — task framing 구조적 불일치:**
ChronusOmni 공식 repo (`data/test/`) 의 grounding 관련 eval template 3종 모두 UnAV-100 과 부정합:

| template | modality framing | UnAV 적합성 |
|---|---|---|
| `longvale_test_grounding.json` | "in the video" (암묵적 multimodal) | 의미 근접하나 **LongVALE adapter layer** — UnAV 에 쓰면 adapter-of-adapter |
| `v2t_openqa.json` | "video segment ... visual information" | **visual-only** — audio-dominant event (wind noise 등) 부당 |
| `a2t_openqa.json` | "audio segment ... audio information" | **audio-only** — visual-dominant event (skateboarding 등) 부당 |

→ ChronusOmni 저자가 **audio-visual 공동 이벤트 grounding 을 위한 native template 을 설계 안 함**. 어느 선택도 fair measurement 아님.

**Why:** (a) 사용자가 longvale 기반 3455 full run 중단 지시 (pid 12287, 128/3455 진행 시점). (b) v2t/a2t 둘 다 돌리기 (~5h40) + limitation 명시 방안도 있었으나 AVST-Zero 먼저 보고 결정하기로. (c) ChronusOmni 완전 skip (prior work 표에서 "not applicable") 옵션도 AVST-Zero 결과 후 재검토 가능.

**보존된 자산 (재사용 가능):**
- `/workspace/jsy/scripts/convert_unav_to_chronusomni.py` — 템플릿 교체 쉬운 구조
- `/workspace/jsy/scripts/eval_chronusomni_tvg.py` — `From second{X.X} to second{Y.Y}` 파서 구현 완료 (v2t/a2t 용 `second{X}-second{Y}` 패턴은 추가 필요)
- `/workspace/jsy/output/chronusomni_unavqa/` 안:
  - `sanity_10.json`, `sanity_10_results.json` — 팀 질문 verbatim (V1, FSR 0%)
  - `sanity_20.json`, `sanity_20_results.json` — 팀 질문 verbatim 추가 20 (총 30/30 range 0건)
  - `sanity_10_native.json`, `sanity_10_native_results.json` — longvale-style native (FSR 10/10 = 100%)
  - `unav100_test_chronusomni.json` — 팀 질문 verbatim (3455)
  - `unav100_test_chronusomni_native.json` — longvale-style (3455)
  - `predictions_native.log` — 중단된 full run 587 log lines (128/3455 진행 후 kill)
- `/workspace/jsy/Chronus/checkpoints/` 심볼릭링크 → `/workspace/jsy/hf_cache/chronusomni_ckpt/` (필수 — 없으면 whisper/BEATs 로드 실패)
- conda env `chronusomni` 완전 구성 (torch 2.3.0+cu118, flash_attn 2.6.0, transformers 4.49.0)

**핵심 발견 (미래 세션 참고):**
- ChronusOmni 의 grounding 출력 포맷 trigger 는 **format hint** (`"Give the timestamps in the fromat: From second{} to second{}."`) 에 강하게 의존. hint 없으면 자유 자연어 설명 (sanity 30/30 range 0개).
- ChronusOmni 는 **항상 single segment** 출력 — multiseg GT 에 대해 recall 구조적 제약.
- 타이밍: 2.9 s/sample (A100 80GB), 3455 샘플 약 2h50.
- 파일 경로 주의: repo config 가 `./checkpoints/` 상대경로라 CWD=`/workspace/jsy/Chronus/` 에서만 실행.

**How to apply:** AVST-Zero 측정 끝난 뒤 ChronusOmni 재검토. 그때 옵션:
(4) v2t + a2t 둘 다 돌리고 각각 별도 리포트 (max 같은 합성 지표 금지), limitation 명시
(5) "not applicable" 로 prior work 표에서 제외
결정은 AVST-Zero 가 UnAV 에 얼마나 잘 적용되는지 본 뒤 상대적으로 판단.
