---
name: Avicuna UnAV-100 QA eval file
description: UnAV-100 QA 평가 공용 파일은 main 브랜치 `unav100_test_multiseg_salmonn2plus.json` (3455 rows) — 이전 `unav100_test_full.json` / `unav100_test_grounding.json` 은 모두 폐기됨
type: project
originSessionId: 8bfff642-2202-4b4d-89cf-726ca869ecfd
---

UnAV-100 QA TVG 공용 평가셋은 **`data/unav100_test_multiseg_salmonn2plus.json`** (3455 rows, multiseg, origin/main 기준). 2026-04-21 은지가 재푸시 (commit `a9b133f`): 질문 템플릿을 모든 3455 row에서 재작성, `/data0/aix23102/unav_100/` 경로로 기록. 이전 `unav100_test_full.json` / `unav100_test_grounding.json` / `unav100_test_dense.json` 전부 main에서 삭제됨 (cdh 브랜치에만 구버전 남아있을 수 있음).

로컬 사본:
- 팀 파이프라인용(pristine): `/workspace/jsy/Team4-cdh/data/unav100_test_multiseg_salmonn2plus.json`
- 자체 추론용(경로 remap): `/workspace/jsy/output/avicuna_unavqa/unav100_test_multiseg_salmonn2plus.json` — video/audio 경로가 `/workspace/datasets/unav_100/...` 로 치환됨
- 구버전 보존: `*.OLD_APR20.json` (Apr 20 cdh 본, 질문 wording만 다르고 gt_segments 동일)

포맷:
- `video`, `audio`: 런팟은 remap본 사용 (`/workspace/datasets/unav_100/...`); 팀 공유는 `/data0/aix23102/unav_100/...`
- `conversations[0].value`: `<video>\n...` 자연어 TVG 질문 (Avicuna `DEFAULT_IMAGE_TOKEN`도 `<video>`라 호환)
- `conversations[1].value`: `<t0>..<t9><tdot>` 시간토큰 GT (SALMONN2+ 학습용, Avicuna 추론에서는 무시)
- `gt_segments`: list of [start_sec, end_sec], per-row 1+개 segment
- `gt_label`: 이벤트 라벨
- `event`: 신규 필드 (= gt_label, 중복 정보)

**Why:** (a) main에서 legacy dense/grounding JSON이 전부 삭제돼 `unav100_test_grounding.json` 참조는 이제 깨짐. (b) 새 multiseg 파일이 팀 공통 평가셋이라 Avicuna/SALMONN2+/ChronusOmni 간 head-to-head 비교 가능. (c) 질문 wording 변경으로 인해 CLAUDE.md §2-2, §2-4 의 구버전 숫자는 새 평가 숫자와 **직접 비교 불가** — 재추론 필요.

**How to apply:** Avicuna / ChronusOmni / SALMONN base 재추론 시 avicuna_unavqa 경로의 remap본을 입력으로 사용. gt_segments·video set·sample count는 구버전과 동일하므로 `eval_miou_multiseg.py`·`eval_unavqa_tvg.py` 등 평가 스크립트는 수정 불필요. Avicuna는 pct 출력이라 duration 캐시(`durations.json`)로 초 변환 그대로 유지.
