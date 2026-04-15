#!/usr/bin/env python3
"""
UnAV-100 multi-segment QA 데이터 생성 (SALMONN2+ 형식).

같은 비디오+같은 event의 모든 구간을 하나의 응답으로:
  Q: "<video>\n{query}"
  A: "From <t...> to <t...>. From <t...> to <t...>."

- train: 10358 samples (unique video+event pairs)
- test: 3455 samples
- Time token 형식: VTG-LLM (<t0>~<t9>, <tdot>)
- max_time 기반으로 초→time token 변환
"""
import json
import random
import os
import re
from collections import defaultdict

random.seed(42)

TEMPLATES = [
    "At what point in the video does {EVENT} occur in terms of both video and audio?",
    "Can you identify the timestamps where {EVENT} happens in the video?",
    "In which segments of the video do we find {EVENT} visually or audibly represented?",
    "Where in the video can we detect {EVENT}, both in the visuals and in the audio track?",
    "Which parts of the video showcase {EVENT} in both imagery and sound?",
]

VIDEO_DIR = "/data0/aix23102/unav_100/videos"
AUDIO_DIR = "/data0/aix23102/unav_100/audio"
UNAV_JSON = "/home/aix23102/audiolm/CCNet/data/unav_100/annotations/unav100_annotations.json"
OUT_DIR = "/home/aix23102/audiolm/vS2_eunji/data"

MAX_TIME = 60.0  # UnAV-100 max duration


def seconds_to_time_tokens(t, max_time=60.0):
    """초를 VTG-LLM time token 문자열로 변환."""
    t = min(t, max_time)
    t = max(t, 0.0)
    # 4자리 정수 + 1자리 소수
    integer = int(t)
    decimal = int(round((t - integer) * 10)) % 10
    digits = f"{integer:04d}"
    tokens = "".join(f"<t{d}>" for d in digits)
    tokens += f"<tdot><t{decimal}>"
    return tokens


def make_response(segments, max_time=60.0):
    """여러 구간을 'From X to Y. From X to Y.' 형식으로."""
    parts = []
    for seg in segments:
        start_tok = seconds_to_time_tokens(seg[0], max_time)
        end_tok = seconds_to_time_tokens(seg[1], max_time)
        parts.append(f"From {start_tok} to {end_tok}.")
    return " ".join(parts)


def main():
    with open(UNAV_JSON, "r") as f:
        db = json.load(f)["database"]

    for split in ["train", "test"]:
        vids = {k: v for k, v in db.items() if v.get("subset", "").lower() == split}

        data = []
        for vid, item in sorted(vids.items()):
            video_path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
            audio_path = os.path.join(AUDIO_DIR, f"{vid}.wav")

            # 같은 event의 구간 그룹핑
            label_segs = defaultdict(list)
            for ann in item.get("annotations", []):
                label_segs[ann["label"]].append(ann["segment"])

            for label, segments in label_segs.items():
                # 시간순 정렬
                segments = sorted(segments, key=lambda x: x[0])

                template = random.choice(TEMPLATES)
                question = template.replace("{EVENT}", label)
                response = make_response(segments, MAX_TIME)

                sample = {
                    "video": video_path,
                    "audio": audio_path,
                    "use_audio": True,
                    "conversations": [
                        {"from": "human", "value": f"<video>\n{question}"},
                        {"from": "gpt", "value": response},
                    ],
                }

                # test용 metadata
                if split == "test":
                    sample["gt_label"] = label
                    sample["gt_segments"] = segments

                data.append(sample)

        out_path = os.path.join(OUT_DIR, f"unav100_{split}_multiseg_salmonn2plus.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # 통계
        seg_counts = []
        for item in data:
            n = item["conversations"][1]["value"].count("From")
            seg_counts.append(n)
        multi = sum(1 for c in seg_counts if c > 1)

        print(f"[{split}] {len(data)} samples -> {out_path}")
        print(f"  single-seg: {len(data) - multi}, multi-seg: {multi}")

        # 샘플
        for item in data[:2]:
            print(f"  Q: {item['conversations'][0]['value'][8:80]}...")
            print(f"  A: {item['conversations'][1]['value'][:120]}")
            print()


if __name__ == "__main__":
    main()
