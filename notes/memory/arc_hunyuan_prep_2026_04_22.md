---
name: ARC-Hunyuan-Video-7B UnAV-100 측정 준비 (Crab+ 이후 착수)
description: 17GB public ckpt. audio 입력 지원. Grounding task 내장. UnAV-100 측정 가능. prompt + 어댑터 design 완료, full 실행은 Crab+ 끝난 뒤
type: project
originSessionId: 7cf60908-ffa2-425c-9445-c628fd4bc0af
---
**상태 (2026-04-22)**: ARC-Hunyuan-Video-7B repo clone 완료 (`/workspace/jsy/ARC-Hunyuan-Video-7B/`). 측정 가능성 확정, **Crab+ full 끝난 후 착수**.

**모델 자산:**
- HF: `TencentARC/ARC-Hunyuan-Video-7B` (public, gated=False, **17.27 GB**)
- Repo: https://github.com/TencentARC/ARC-Hunyuan-Video-7B
- Paper: arXiv 2507.20939 "Structured Video Comprehension of Real-World Shorts"
- Backbone: Hunyuan-7B VL + Whisper-large-v3 audio encoder + timestamp overlay

**Prompt 구조 (video_inference.py:106-114):**
```python
def build_prompt(question, num_frames, task="Grounding"):
    video_prefix = "<image>" * num_frames
    if task == "Grounding":
        return f"<|startoftext|>{video_prefix}\n{question}\nOutput the thinking process in <think> </think> and final answer (only time range) in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.<sep>"
```
→ task="Grounding" 필수. wrapping 은 고정, question 부분만 우리가 제어.

**Native grounding 예시 (저자 데모 영어):**
```
"When will we be able to see the man in the video eat the pork cutlet in the restaurant?"
```
→ 자연어 query-based TVG. UnAV query wording 과 궁합 좋음 (ChronusOmni 와 유사한 task framing).

**Timestamp 포맷 (output):**
- `<answer>{time_range}</answer>` — 정확한 문법 미공개 (README 예시 없음)
- **sanity 첫 실행에서 관측 필요** — HH:MM:SS? seconds? from–to?
- max_new_tokens=1024, do_sample=False (greedy)

**UnAV-100 적용 (hybrid 전략):**
- Body: 팀 wording `"At what point in the video does {event} occur in terms of both video and audio?"`
- Task: `"Grounding"` (build_prompt 에 고정)
- 출력 파서는 sanity 후 결정 (mm:ss vs seconds 판정)

**기술적 차별점 (vs Crab+):**
- Crab+ 는 nframes=10 고정 → [0,10] cap. ARC-Hunyuan 은 `sample_fps` 기반 frame sampling (video_duration 까지 최대 150 초) → **장시간 video 네이티브 지원**, UnAV 42s 평균 문제없음
- 오디오 입력 지원 (Whisper-large-v3 → audio encoder)
- 5분 초과 시 segment split, 5분 내는 one-shot

**준비물 (Crab+ 이후):**
1. `conda create -n archunyuan python=3.10`
2. `pip install -r /workspace/jsy/ARC-Hunyuan-Video-7B/requirements.txt` + flash-attn  
3. `scripts/convert_unav_to_arc_hunyuan.py` — 팀 JSON → question list (`{id,video,audio,question,task:"Grounding"}`)
4. `scripts/inference_arc_hunyuan.py` — cdh 표준 runner, model=ARCHunyuanVideoForConditionalGeneration.from_pretrained
5. sanity 10 → timestamp 포맷 확인 → parser 작성 → full 3455
6. `scripts/eval_miou_multiseg_arc_hunyuan.py` — cdh 인터페이스

**예상 리소스:**
- 모델 다운로드 17 GB (/workspace)
- env 빌드 ~30 min
- sanity + format 확인: 10 min
- full 3455 추론 (GPU 단일): 추정 3–5 h (ChronusOmni 3h / Crab+ 6.7h 사이)

**예상 결과 (가설):**
- UnAV 42s 평균을 네이티브로 처리 → Crab+ 의 [0,10] cap 문제 없음
- Grounding task 로 query-based TVG 지원 → ChronusOmni 급 또는 상회 가능
- Audio input 활용해 AV TVG 에 fit → 팀 SALMONN base 와 직접 비교 가능

**Head-to-head 표 (현재 + 예상):**
| 모델 | FSR | mIoU | 상태 |
|---|---:|---:|---|
| ChronusOmni hybrid | 100% | 36.78% | ✅ 완료 |
| Avicuna stage4 | 100% | 30.42% | ✅ 완료 |
| SALMONN V2 | 89.81% | 19.81% | ✅ 완료 |
| Crab+ hybrid | — | 예상 낮음 (10s cap) | 🟡 진행 중 |
| **ARC-Hunyuan hybrid** | — | TBD | ⏳ 다음 차례 |
| LongVALE-LLM | — | — | ❌ ckpt 미공개 |
| AVST-Zero | — | — | ❌ ckpt 미공개 |

**How to apply:** Crab+ watcher completion 후 이 memo 참조해서 ARC-Hunyuan 착수. sanity 에서 `<answer>` 내부 timestamp 문법 먼저 확인 → parser 짜고 → full.
