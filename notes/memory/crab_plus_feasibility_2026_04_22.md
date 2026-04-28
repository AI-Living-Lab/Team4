---
name: Crab+ (GeWu-Lab) UnAV-100 측정 가능 — ChronusOmni 이후 진행
description: Crab+ (Qwen2.5-Omni-7B + I-LoRA) 는 ckpt 공개 + UnAV 를 temporal grounding 으로 학습 — 측정 가능. ChronusOmni 결과 본 후 착수
type: project
originSessionId: 7cf60908-ffa2-425c-9445-c628fd4bc0af
---
**2026-04-22 조사 결과**: Crab+ (https://github.com/GeWu-Lab/Crab_Plus) 는 AVST-Zero 와 달리 **즉시 측정 가능**.

**가용 자산:**
- Fine-tuned LoRA weights: HF dataset `Jayson236/Crab_Plus/finetune_weights.bin` (1.74 GB, public, gated=False)
- Backbone: Qwen/Qwen2.5-Omni-7B (15 GB, public)
- Annotation zip `AVUIE_2.zip` (22 MB) — 이미 `/workspace/jsy/output/crab_plus_inspect/` 에 다운 완료
- Inference 스크립트: `scripts/finetune/inference_omni.{py,sh}` (`--ckpt_dir weight/` 로 LoRA load)

**UnAV task framing (Crab+ 공식):**
- Prompt: `"This is a video and an audio:" + <video> + <audio> + "Please describe the events and time range that occurred in the video."`
- Target 포맷 (AVUIE_2/unav/train.json 실측):
  - `original_label`: `<event>people eating, (0 3)(7 10)</event>` — multi-tuple = multi-segment, multi `<event>` 블록 = multi-event
  - `label_content`: 자연어 서술 + 마지막 XML 요약 `"Therefore, the events and time range are <event>...</event>."`
- 정수 초 단위 (float 아님)

**주의점:**
- UnAV **test split 공식 미공개** — `AVUIE_2/unav/` 에 train.json (4925) 만 존재, test.json 없음. Crab+ 저자가 UnAV 수치 리포트 안 함 → **우리가 first-reporter**
- 우리 3455 `unav100_test_multiseg_salmonn2plus.json` 을 Crab+ 포맷 `{vid, video_path, audio_path, label_content}` 로 변환 필요
- Video/audio 파일명이 Crab+ 예상 경로 (`AVUIE_2/unav/video/{vid}.mp4`) 와 로컬 (`/workspace/datasets/unav_100/...`) 매핑 확인 필요

**How to apply:** ChronusOmni 결과 고정 후 시작. 소요 시간 estimate:
- 다운로드 17 GB (~20 min)
- conda env `crab` 빌드 (~30 min, `requirements.txt` + flash-attn)
- UnAV 어댑터 JSON 변환 (~1 h, 팀 질문 wording + Crab+ format tail hybrid)
- 추론 3455 × 3–5 s/sample ≈ **3–5 h**
- 총 **4–6 h**

**Prompt 하이브리드 (신 정책 per feedback_headtohead_prompts):**
- 본문: 팀 wording `"At what point in the video does {event} occur in terms of both video and audio?"`
- tail: Crab+ native `"Please describe the events and time range that occurred in the video."` 또는 XML tail 직접 `"Output in the format of '<event>{label}, (start end)</event>'."`
