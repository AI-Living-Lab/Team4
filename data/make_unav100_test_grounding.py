#!/usr/bin/env python3
"""
UnAV-100 test set을 학습 데이터(unav100_train_grounding.json)와
동일한 single QA 형식으로 생성.

각 GT segment마다 하나의 QA 쌍:
  Human: "<image>\n{질문 템플릿 with event label}"
  GPT:   "From <t...> to <t...>."

+ GT 정보를 metadata로 포함 (eval용)
"""
import json
import random
import os

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
OUT_PATH = "/home/aix23102/audiolm/vS2_eunji/data/unav100_test_grounding.json"

with open(UNAV_JSON, "r") as f:
    db = json.load(f)["database"]

test_data = []
for vid, item in sorted(db.items()):
    if item.get("subset", "").lower() != "test":
        continue

    video_path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
    audio_path = os.path.join(AUDIO_DIR, f"{vid}.wav")

    for ann in item.get("annotations", []):
        label = ann["label"]
        seg = ann["segment"]

        template = random.choice(TEMPLATES)
        question = template.replace("{EVENT}", label)

        sample = {
            "video": video_path,
            "audio": audio_path,
            "conversations": [
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": "PLACEHOLDER"},
            ],
            # eval용 metadata
            "gt_label": label,
            "gt_segment": seg,
        }
        test_data.append(sample)

with open(OUT_PATH, "w") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(test_data)} test samples to {OUT_PATH}")
print(f"  videos: {len(set(d['video'] for d in test_data))}")
print(f"  labels: {len(set(d['gt_label'] for d in test_data))}")
