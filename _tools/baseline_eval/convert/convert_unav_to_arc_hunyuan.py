"""UnAV-100 QA (team multiseg JSON) → ARC-Hunyuan eval input.

ARC-Hunyuan build_prompt (video_inference.py:106-114) Grounding task:
  f"<|startoftext|>{video_prefix}\n{question}\n
   Output the thinking process in <think> </think> and final answer
   (only time range) in <answer> </answer> tags, i.e.,
   <think> reasoning process here </think><answer> answer here </answer>.<sep>"

Hybrid 전략:
  body (question slot) = 팀 wording 본문
    "At what point in the video does {event} occur in terms of both video and audio?"
  tail = **없음** — ARC-Hunyuan build_prompt wrap 이 이미 format hint 제공 (`only time range`)
        => query wording 뒤에 별도 native-tail 추가 불필요

출력 JSON 스키마 (우리 runner 가 읽을):
  {
    "id": "{vid}_{row_idx:04d}",
    "vid": "...",
    "video": "/workspace/datasets/unav_100/videos/{vid}.mp4",
    "audio": "/workspace/datasets/unav_100/audio/{vid}.wav",
    "question": "<our hybrid body>",
    "task": "Grounding",
    "gt_label": ...,
    "gt_segments": [[s,e],...],
  }
"""

import argparse
import json
from pathlib import Path


TEAM_PREFIX = "<video>\n"  # 팀 JSON 이 붙여놓은 Avicuna 토큰, 제거 대상


def vid_of(path: str) -> str:
    return Path(path).stem


def strip_video_token(q: str) -> str:
    if q.startswith(TEAM_PREFIX):
        return q[len(TEAM_PREFIX):]
    if q.startswith("<video>"):
        return q[len("<video>"):].lstrip("\n")
    return q


def build(src_path: str, dst_path: str) -> None:
    with open(src_path) as f:
        src = json.load(f)

    out = []
    for idx, r in enumerate(src):
        vid = vid_of(r["video"])
        body = strip_video_token(r["conversations"][0]["value"]).rstrip()
        out.append(
            {
                "id": f"{vid}_{idx:04d}",
                "vid": vid,
                "video": r["video"],
                "audio": r["audio"],
                "question": body,
                "task": "Grounding",
                "gt_label": r["gt_label"],
                "gt_segments": r["gt_segments"],
            }
        )

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"wrote {dst_path}  n={len(out)}")
    print(f"sample[0].question:\n  {out[0]['question']!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/jsy/output/avicuna_unavqa/unav100_test_multiseg_salmonn2plus.json")
    ap.add_argument("--dst", default="/workspace/jsy/outputs/base/ArcHunyuan/test.json")
    args = ap.parse_args()
    build(args.src, args.dst)
