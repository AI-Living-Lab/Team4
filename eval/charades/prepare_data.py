#!/usr/bin/env python3
"""
Charades-STA test set → vS2 inference JSON (PU-VALOR grounding format)

Input:  charades_sta_test.txt  (format: "{vid} {start} {end}##{query}")
Output: charades_sta_test.json (vS2 conversation format)
"""
import argparse, json, os, random

GROUNDING_TEMPLATES = [
    "In which segments of the video do we find {query}?",
    "Which parts of the video showcase {query}?",
    "Where in the video can we detect {query}?",
    "At what point in the video does {query} occur?",
    "Can you identify the timestamps where {query} happens in the video?",
]

def parse_charades_sta(txt_path):
    samples = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta, query = line.split("##", 1)
            parts = meta.split()
            vid = parts[0]
            start = float(parts[1])
            end = float(parts[2])
            samples.append({
                "video_id": vid,
                "query": query.strip(),
                "start": start,
                "end": end,
            })
    return samples


def seconds_to_time_tokens(sec):
    """Convert seconds (e.g. 24.3) to PU-VALOR time token string like <t0><t0><t2><t4><tdot><t3>"""
    sec = max(0.0, sec)
    # 4-digit integer part + 1 decimal
    integer_part = int(sec)
    decimal_part = int(round((sec - integer_part) * 10)) % 10
    digits = f"{integer_part:04d}"
    tokens = "".join(f"<t{d}>" for d in digits)
    tokens += f"<tdot><t{decimal_part}>"
    return tokens


def build_conversation(query):
    template = random.choice(GROUNDING_TEMPLATES)
    prompt = template.format(query=query)
    return [
        {"from": "human", "value": f"<image>\n{prompt}"},
        {"from": "gpt", "value": ""},  # placeholder for inference
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_path", default="/data0/aix23102/charades_sta/annotations/charades_sta_test.txt")
    parser.add_argument("--video_dir", default="/data0/aix23102/charades_sta/Charades_v1")
    parser.add_argument("--output", default="/home/aix23102/audiolm/vS2_eunji/eval/charades/charades_sta_test.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    samples = parse_charades_sta(args.ann_path)

    out = []
    for s in samples:
        vid_path = os.path.join(args.video_dir, f"{s['video_id']}.mp4")
        gt_start = seconds_to_time_tokens(s["start"])
        gt_end = seconds_to_time_tokens(s["end"])

        entry = {
            "video": vid_path,
            "conversations": build_conversation(s["query"]),
            "gt_start": s["start"],
            "gt_end": s["end"],
            "gt_start_tokens": gt_start,
            "gt_end_tokens": gt_end,
            "video_id": s["video_id"],
            "query": s["query"],
        }
        out.append(entry)

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(out)} samples → {args.output}")


if __name__ == "__main__":
    main()
