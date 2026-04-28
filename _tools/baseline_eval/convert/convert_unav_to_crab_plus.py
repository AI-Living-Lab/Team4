"""UnAV-100 QA (team multiseg JSON) → Crab+ AVUIE_2/unav/test.json 포맷.

Crab+ 훈련 시 사용한 `add_unav_samples` 로직 (omni_dataset.py:654-694) 에 맞는 스키마:
  {
    "vid":            "bHfesK3rBE4",         # 모델 내부에서 쓰이진 않으나 meta
    "audio_path":     "bHfesK3rBE4.wav",     # AVUIE_2/unav/audio/ 하위 상대경로
    "video_path":     "bHfesK3rBE4.mp4",     # AVUIE_2/unav/video/ 하위 상대경로
    "label_content":  "(dummy, used by train-target; inference 에선 무시)",
    "original_label": "<event>{label}, (gt_s gt_e)(gt_s gt_e)</event>",  # GT (eval 용)
    "gt_segments":    [[s,e],...],
    "gt_label":       "...",
    "question":       "(hybrid team wording + crab-native tail, 실제 주입될 질문)",
  }

Crab+ 원래 user prompt (omni_dataset.py:673-676):
  "This is a video and an audio:" + <video> + <audio> + "Please describe the events and time range that occurred in the video."

Hybrid 전략 (feedback_headtohead_prompts):
  본문 = 팀 wording `"At what point in the video does {event} occur in terms of both video and audio?"`
  tail = Crab+ native `"Please describe the events and time range that occurred in the video."`
  → 우리 runner 가 저 질문을 Crab+ 메시지 템플릿에 주입

Path 처리:
  팀 JSON: /workspace/datasets/unav_100/videos/{vid}.mp4, /audio/{vid}.wav
  Crab+:   AVUIE_2/unav/video/{vid}.mp4, audio/{vid}.wav
  우리가 만든 symlink: data_local/AVUIE_2/unav/{video,audio} → /workspace/datasets/unav_100/{videos,audio}
"""

import argparse
import json
from pathlib import Path

TEAM_BODY_PREFIX = "<video>\n"   # 팀 JSON 이 붙여놓은 Avicuna 토큰, 제거 대상
CRAB_TAIL = " Please describe the events and time range that occurred in the video."


def vid_of(path: str) -> str:
    return Path(path).stem


def strip_video_token(q: str) -> str:
    if q.startswith(TEAM_BODY_PREFIX):
        return q[len(TEAM_BODY_PREFIX):]
    if q.startswith("<video>"):
        return q[len("<video>"):].lstrip("\n")
    return q


def original_label_from_gt(gt_label: str, gt_segments) -> str:
    """GT 를 Crab+ training label 문법으로: <event>{label}, (s1 e1)(s2 e2)</event>"""
    tuples = "".join(f"({int(round(float(s)))} {int(round(float(e)))})" for s, e in gt_segments)
    return f"<event>{gt_label}, {tuples}</event>"


def build(src_path: str, dst_path: str) -> None:
    with open(src_path) as f:
        src = json.load(f)

    out = []
    for idx, r in enumerate(src):
        vid = vid_of(r["video"])
        audio_fn = Path(r["audio"]).name   # e.g. "--Bu2xe4OSo.wav"
        video_fn = Path(r["video"]).name   # e.g. "--Bu2xe4OSo.mp4"
        body = strip_video_token(r["conversations"][0]["value"]).rstrip()
        question = body + CRAB_TAIL
        out.append(
            {
                "id": f"{vid}_{idx:04d}",
                "vid": vid,
                "audio_path": audio_fn,
                "video_path": video_fn,
                "label_content": "(inference only, no loss target)",
                "original_label": original_label_from_gt(r["gt_label"], r["gt_segments"]),
                "gt_label": r["gt_label"],
                "gt_segments": r["gt_segments"],
                "question": question,
            }
        )

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"wrote {dst_path}  n={len(out)}")
    print(f"sample[0].question:\n  {out[0]['question']!r}")
    print(f"sample[0].original_label:\n  {out[0]['original_label']!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/jsy/output/avicuna_unavqa/unav100_test_multiseg_salmonn2plus.json")
    ap.add_argument("--dst", default="/workspace/jsy/Crab_Plus/data_local/AVUIE_2/unav/test.json")
    args = ap.parse_args()
    build(args.src, args.dst)
