#!/usr/bin/env python3
"""
UnAV-100 raw annotation → stage3-style JSON (T1 DAVC + T2 TSG).

Input:  /home/aix23102/audiolm/CCNet/data/unav_100/annotations/unav100_annotations.json
Output: unav_train.json  (T1 + T2 samples, stage3-style)

stage3-style convention:
  - GPT response 에 <s0>/<e0>/... placeholder 사용
  - meta.token 에 placeholder → 실제 초 매핑
  - tokenization (4+1, 3+2 등) 은 별도 후처리 단계에서 처리

T1 (DAVC):  1 sample per video, 모든 segment 를 multi-event narrative 로
T2 (TSG):   1 sample per (video, unique_label), 그 label 의 모든 instance 를 multi-segment 응답으로
            (UnAV 의 label 은 짧고 반복 출현 → unique label 단위가 자연스러움)
"""
import argparse
import json
import os
import random
from collections import defaultdict

ANN_PATH = "/home/aix23102/audiolm/CCNet/data/unav_100/annotations/unav100_annotations.json"
VIDEO_DIR = "/data0/aix23102/unav_100/videos"
AUDIO_DIR = "/data0/aix23102/unav_100/audio"

# PU-VALOR T1 (DAVC) 스타일 — open Q. 비디오 전체 narrative 요청.
T1_PROMPTS = [
    "Can you give a brief overview of significant moments in the video, including both imagery and sound?",
    "Tell me about the visual and audio events in the video.",
    "What notable events, both visual and audio, occur at different points in the video?",
    "Could you describe the main events in the video, including both visual and audio cues?",
    "Provide details about the visual scenes and audio events in the video.",
    "What events occur at different points in the video, in terms of both visuals and audio?",
    "Could you briefly detail the events in the video, including both imagery and sound?",
    "Can you outline what happens at different moments in the video, both visually and audibly?",
    "What are the key visual and audio events in the video?",
    "Can you summarize the key events in the video, both visually and audibly?",
]

# PU-VALOR T2 (TSG) 스타일 — caption-based Q. {event} 가 언제 일어났는지 묻기.
T2_PROMPTS = [
    "Where in the video can we detect {event}, both in the visuals and in the audio track?",
    "At what point in the video does {event} occur in terms of both video and audio?",
    "In which segments of the video does {event} happen, both visually and audibly?",
    "At which intervals in the video can we observe {event}, in terms of both visuals and audio?",
    "During which periods of the video can we hear and see {event}?",
    "Which parts of the video showcase {event} in both imagery and sound?",
    "Can you identify the timestamps where {event} occurs in the video?",
    "At what timestamps can we observe {event} in the video, either through sight or sound?",
    "In which sections of the video does {event} occur, both visually and audibly?",
    "Between which visual scenes and audio events does {event} happen in the video?",
]

# T2 의 turn-3 (description-asking) prompts.
T2_DESC_PROMPTS = [
    "Provide details about the visual scenes and audio events from <s0> to <e0> in the video.",
    "Tell me about the visual and audio events from <s0> to <e0> in the video.",
    "What was going on visually and audibly from <s0> to <e0> in the video?",
    "Explain what happened, considering both video and audio, from <s0> to <e0>.",
    "Could you tell me what happened, in terms of both imagery and sound, from <s0> to <e0> in the video?",
]


def build_t1_sample(vid, info, rng):
    """1 video → 1 T1 sample. 모든 annotation 을 시간순 narrative."""
    annotations = sorted(info["annotations"], key=lambda a: a["segment"][0])
    if not annotations:
        return None

    token_map = {}
    narrative_parts = []
    for i, ann in enumerate(annotations):
        s_ph = f"<s{i}>"
        e_ph = f"<e{i}>"
        token_map[s_ph] = float(ann["segment"][0])
        token_map[e_ph] = float(ann["segment"][1])
        label = ann["label"].strip()
        narrative_parts.append(f"From {s_ph} to {e_ph}, the sound of {label} can be heard.")

    narrative = " ".join(narrative_parts)
    prompt = rng.choice(T1_PROMPTS)

    return {
        "id": f"unav_{vid}_t1",
        "video": os.path.join(VIDEO_DIR, f"{vid}.mp4"),
        "audio": os.path.join(AUDIO_DIR, f"{vid}.wav"),
        "use_audio": True,
        "conversations": [
            {"from": "human", "value": f"<video>\n{prompt}"},
            {"from": "gpt", "value": narrative},
        ],
        "meta": {
            "duration": float(info["duration"]),
            "token": token_map,
        },
        "source": "unav",
        "source_id": "unav",
        "task_type": "T1",
    }


def build_t2_samples(vid, info, rng):
    """1 (video, unique_label) → 1 T2 sample. 그 label 의 모든 instance 를 multi-segment 응답."""
    # group by label, sort each group by start time
    by_label = defaultdict(list)
    for ann in info["annotations"]:
        by_label[ann["label"].strip()].append(ann["segment"])
    for lbl in by_label:
        by_label[lbl] = sorted(by_label[lbl], key=lambda s: s[0])

    out = []
    for lbl_idx, (label, segs) in enumerate(sorted(by_label.items())):
        token_map = {}
        time_parts = []
        for i, (s, e) in enumerate(segs):
            s_ph = f"<s{i}>"
            e_ph = f"<e{i}>"
            token_map[s_ph] = float(s)
            token_map[e_ph] = float(e)
            time_parts.append(f"From {s_ph} to {e_ph}.")

        time_response = " ".join(time_parts)
        prompt_q = rng.choice(T2_PROMPTS).format(event=label)
        desc_q = rng.choice(T2_DESC_PROMPTS)
        desc_a = f"The sound of {label} can be heard."

        out.append({
            "id": f"unav_{vid}_t2_{lbl_idx}",
            "video": os.path.join(VIDEO_DIR, f"{vid}.mp4"),
            "audio": os.path.join(AUDIO_DIR, f"{vid}.wav"),
            "use_audio": True,
            "conversations": [
                {"from": "human", "value": f"<video>\n{prompt_q}"},
                {"from": "gpt", "value": time_response},
                {"from": "human", "value": desc_q},
                {"from": "gpt", "value": desc_a},
            ],
            "meta": {
                "duration": float(info["duration"]),
                "token": token_map,
            },
            "source": "unav",
            "source_id": "unav",
            "task_type": "T2",
            "event": label,
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["train", "validation", "test", "all"], default="train")
    parser.add_argument("--output", default="/home/aix23102/audiolm/vS2_eunji/data/unav_train.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check_paths", action="store_true",
                        help="Skip samples whose video/audio files are missing")
    args = parser.parse_args()

    print(f"Loading {ANN_PATH} ...", flush=True)
    with open(ANN_PATH) as f:
        ann = json.load(f)
    db = ann["database"]

    target_subsets = {"train", "validation", "test"} if args.subset == "all" else {args.subset}

    rng = random.Random(args.seed)

    all_samples = []
    n_skip_no_anno = 0
    n_skip_missing_file = 0
    n_skip_wrong_subset = 0

    vid_list = sorted(db.keys())
    for vid in vid_list:
        info = db[vid]
        if info.get("subset") not in target_subsets:
            n_skip_wrong_subset += 1
            continue
        if not info.get("annotations"):
            n_skip_no_anno += 1
            continue
        if args.check_paths:
            v = os.path.join(VIDEO_DIR, f"{vid}.mp4")
            a = os.path.join(AUDIO_DIR, f"{vid}.wav")
            if not (os.path.exists(v) and os.path.exists(a)):
                n_skip_missing_file += 1
                continue

        t1 = build_t1_sample(vid, info, rng)
        if t1 is not None:
            all_samples.append(t1)
        all_samples.extend(build_t2_samples(vid, info, rng))

    # task type 분포
    from collections import Counter
    cnt = Counter(s["task_type"] for s in all_samples)
    print(f"Total samples generated: {len(all_samples)}")
    print(f"  T1 (DAVC): {cnt.get('T1', 0)}")
    print(f"  T2 (TSG):  {cnt.get('T2', 0)}")
    print(f"Skipped: wrong_subset={n_skip_wrong_subset}, no_anno={n_skip_no_anno}, missing_file={n_skip_missing_file}")

    # shuffle then save (deterministic with seed)
    rng_shuf = random.Random(args.seed + 1)
    rng_shuf.shuffle(all_samples)

    with open(args.output, "w") as f:
        json.dump(all_samples, f)
    sz = os.path.getsize(args.output) / 1024 / 1024
    print(f"Saved → {args.output} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
