"""UnAV-100 QA (team multiseg JSON) → Qwen2.5-Omni raw eval input.

Qwen2.5-Omni-7B 자체는 UnAV / TVG 전용 SFT 안 됨 (foundation model). 그래서
SALMONN base V2 와 동일하게 "seconds 형식 출력" hint 를 tail 에 붙이는 중립 hybrid 사용.

Hybrid:
  body  = 팀 wording  "At what point in the video does {event} occur in terms of both video and audio?"
  tail  = " Answer with the start and end timestamps of the event in seconds."
           (SALMONN V2 와 동일)

Crab+ 와 구별: Crab+ 는 I-LoRA 가 `<event>{label}, (s e)</event>` 출력 학습 → 우리가 Crab+ native tail 사용.
Qwen-Omni raw 는 trained format 없음 → 포괄적 seconds hint.
"""

import argparse
import json
from pathlib import Path

SECONDS_TAIL = " Answer with the start and end timestamps of the event in seconds."


def vid_of(path: str) -> str:
    return Path(path).stem


def strip_video_token(q: str) -> str:
    if q.startswith("<video>\n"):
        return q[len("<video>\n"):]
    if q.startswith("<video>"):
        return q[len("<video>"):].lstrip("\n")
    return q


def build(src_path: str, dst_path: str) -> None:
    with open(src_path) as f:
        src = json.load(f)
    out = []
    for idx, r in enumerate(src):
        vid = vid_of(r["video"])
        video_fn = Path(r["video"]).name
        audio_fn = Path(r["audio"]).name
        body = strip_video_token(r["conversations"][0]["value"]).rstrip()
        q = body + SECONDS_TAIL
        out.append({
            "id": f"{vid}_{idx:04d}",
            "vid": vid,
            "video_path": video_fn,
            "audio_path": audio_fn,
            "gt_label": r["gt_label"],
            "gt_segments": r["gt_segments"],
            "question": q,
        })
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"wrote {dst_path}  n={len(out)}")
    print(f"sample[0].question: {out[0]['question']!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/jsy/output/avicuna_unavqa/unav100_test_multiseg_salmonn2plus.json")
    ap.add_argument("--dst", default="/workspace/jsy/outputs/base/QwenOmniRaw/test.json")
    args = ap.parse_args()
    build(args.src, args.dst)
