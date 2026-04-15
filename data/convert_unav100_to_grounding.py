#!/usr/bin/env python3
"""
UnAV-100 데이터를 PU-VALOR grounding 포맷으로 변환.

원본 JSON 포맷 (dense):
  events: [{"label": "...", "timestamps": [start, end]}, ...]
  conversations: JSON list 출력 포맷

변환 후 (PU-VALOR grounding 스타일):
  각 event마다 별도 QA pair 생성
  Q: "In which segments of the video do we find {event}?"
  A: "From <t0><t0>...<tdot>... to <t0><t0>...<tdot>..."
"""
import argparse, json, os, random

GROUNDING_TEMPLATES = [
    "In which segments of the video do we find {event} visually or audibly represented?",
    "Which parts of the video showcase {event} in both imagery and sound?",
    "Where in the video can we detect {event}, both in the visuals and in the audio track?",
    "At what point in the video does {event} occur in terms of both video and audio?",
    "Can you identify the timestamps where {event} happens in the video?",
]


def seconds_to_time_tokens(sec):
    """Convert seconds to PU-VALOR time token string: <t0><t0><t2><t4><tdot><t3>"""
    sec = max(0.0, sec)
    integer_part = int(sec)
    decimal_part = int(round((sec - integer_part) * 10)) % 10
    digits = f"{integer_part:04d}"
    tokens = "".join(f"<t{d}>" for d in digits)
    tokens += f"<tdot><t{decimal_part}>"
    return tokens


def convert_dense(data, seed=42):
    """dense 포맷 변환: 각 event를 개별 grounding QA로"""
    rng = random.Random(seed)
    out = []

    for item in data:
        video = item["video"]
        audio = item.get("audio", "")
        events = item.get("events", [])

        if not events:
            continue

        # 각 event마다 개별 sample 생성
        for ev in events:
            label = ev["label"]
            start, end = ev["timestamps"]
            start_tok = seconds_to_time_tokens(start)
            end_tok = seconds_to_time_tokens(end)

            template = rng.choice(GROUNDING_TEMPLATES)
            question = template.format(event=label)
            answer = f"From {start_tok} to {end_tok}."

            entry = {
                "video": video,
                "audio": audio,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
            }
            out.append(entry)

    return out


def convert_single(data, seed=42):
    """single 포맷 변환"""
    rng = random.Random(seed)
    out = []

    for item in data:
        video = item["video"]
        audio = item.get("audio", "")
        timestamps = item.get("timestamps", [])

        if len(timestamps) < 2:
            continue

        start, end = timestamps[0], timestamps[1]
        start_tok = seconds_to_time_tokens(start)
        end_tok = seconds_to_time_tokens(end)

        # conversations에서 event label 추출
        human_text = item["conversations"][0]["value"]
        # "Event: {label}" 패턴에서 추출
        label = ""
        for line in human_text.split("\n"):
            if line.strip().startswith("Event:"):
                label = line.strip().replace("Event:", "").strip()
                break

        if not label:
            continue

        template = rng.choice(GROUNDING_TEMPLATES)
        question = template.format(event=label)
        answer = f"From {start_tok} to {end_tok}."

        entry = {
            "video": video,
            "audio": audio,
            "conversations": [
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": answer},
            ],
        }
        out.append(entry)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", default="/home/aix23102/audiolm/vS2_eunji/data/unav100_train_dense.json")
    parser.add_argument("--single", default="/home/aix23102/audiolm/vS2_eunji/data/unav100_train_single.json")
    parser.add_argument("--output", default="/home/aix23102/audiolm/vS2_eunji/data/unav100_train_grounding.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Dense 변환
    with open(args.dense) as f:
        dense_data = json.load(f)
    dense_out = convert_dense(dense_data, args.seed)
    print(f"Dense: {len(dense_data)} videos → {len(dense_out)} grounding QA pairs")

    # Single 변환
    with open(args.single) as f:
        single_data = json.load(f)
    single_out = convert_single(single_data, args.seed)
    print(f"Single: {len(single_data)} samples → {len(single_out)} grounding QA pairs")

    # 합치기 & 셔플
    combined = dense_out + single_out
    random.Random(args.seed).shuffle(combined)
    print(f"Total: {len(combined)} grounding QA pairs")

    with open(args.output, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
