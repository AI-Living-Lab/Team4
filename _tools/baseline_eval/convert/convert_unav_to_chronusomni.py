"""UnAV-100 QA (multiseg salmonn2plus) → ChronusOmni eval input.

팀 신버전 JSON (3455 rows, multiseg) 을 읽어 ChronusOmni inference/eval.py가 요구하는
`[{id, video, audio, question}]` 포맷으로 변환한다.

Prompt 전략 (2026-04-22 신 정책, feedback_headtohead_prompts):
  본문 = 팀 wording 유지 (task framing 보존):
    "At what point in the video does {event} occur in terms of both video and audio?"
  tail = ChronusOmni native output-format hint append:
    " Output in the format of 'From second{start_time} to second{end_time}'."

  - 팀 JSON conversations[0].value 앞의 `<video>\n` 토큰은 제거
    (ChronusOmni eval.py 가 `<speech><image>\n` 을 자체 주입)

ID = f"{vid}_{row_idx:04d}" — multiseg에서 같은 vid + 다른 label 구분용
"""

import argparse
import json
from pathlib import Path

FORMAT_TAIL = " Output in the format of 'From second{start_time} to second{end_time}'."


def vid_of(video_path: str) -> str:
    return Path(video_path).stem


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
        team_body = strip_video_token(r["conversations"][0]["value"]).rstrip()
        q = team_body + FORMAT_TAIL
        out.append(
            {
                "id": f"{vid_of(r['video'])}_{idx:04d}",
                "video": r["video"],
                "audio": r["audio"],
                "question": q,
                "gt_label": r["gt_label"],
                "gt_segments": r["gt_segments"],
            }
        )

    with open(dst_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"wrote {dst_path}  n={len(out)}")
    print(f"sample[0].question:\n  {out[0]['question']!r}")
    print(f"sample[3454].question:\n  {out[-1]['question']!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/jsy/output/avicuna_unavqa/unav100_test_multiseg_salmonn2plus.json")
    ap.add_argument("--dst", default="/workspace/jsy/output/chronusomni_unavqa/unav100_test_chronusomni_hybrid.json")
    args = ap.parse_args()
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    build(args.src, args.dst)
