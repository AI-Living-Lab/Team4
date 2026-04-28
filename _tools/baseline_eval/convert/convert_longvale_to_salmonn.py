"""
Convert LongVALE annotations → SALMONN-2+ flat test JSON.

Input: Avicuna-format database JSON with schema
  {database: {vid: {duration, annotations: [{segment: [s,e], label: str}]}}}

Output: flat list of dicts, each row:
  {
    "video":  "/workspace/datasets/LongVALE/videos/{vid}.mp4",
    "audio":  "/workspace/datasets/LongVALE/audios/{vid}.wav",
    "use_audio": true,
    "conversations": [
      {"from": "human", "value": "<video>\\n<OFFICIAL_PROMPT>"},
      {"from": "gpt",   "value": "<placeholder>"},
    ],
    "gt_label": <label>,
    "gt_segments": [[s, e]],
    "duration": <float>,
  }

Prompt: LongVALE official TVG prompt verbatim (longvalellm/eval/eval.py).
This keeps the pipeline protocol-aligned with Avicuna's LongVALE 1k run.

Usage:
  python convert_longvale_to_salmonn.py \
      --input  /workspace/jsy/output/avicuna_longvale/longvale_annotations.json \
      --output /workspace/jsy/Team4-cdh/data/longvale_test_subset1k.json \
      --limit  1000
"""

import argparse
import json
import os
from pathlib import Path

OFFICIAL_PROMPT = (
    "At which time interval can we find {label} taking place in the video? "
    "Give the timestamps in the fromat: From xx to xx."
)
VIDEO_DIR = "/workspace/datasets/LongVALE/videos"
AUDIO_DIR = "/workspace/datasets/LongVALE/audios"
# Placeholder gpt value — not used during greedy inference; must be parseable
GPT_PLACEHOLDER = "From <t0><t0><t0><t0><tdot><t0> to <t0><t0><t0><t0><tdot><t0>."


def flatten(db_json):
    """Same iteration order as inference_longvale.py::build_queries."""
    queries = []
    for vid, info in db_json["database"].items():
        if info.get("subset") and info["subset"] != "test":
            continue
        duration = float(info["duration"])
        for ann_idx, ann in enumerate(info["annotations"]):
            queries.append({
                "vid": vid,
                "ann_idx": ann_idx,
                "duration": duration,
                "label": ann["label"],
                "segment": [float(ann["segment"][0]), float(ann["segment"][1])],
            })
    return queries


def to_salmonn_row(q):
    prompt = "<video>\n" + OFFICIAL_PROMPT.format(label=q["label"])
    return {
        "video": f"{VIDEO_DIR}/{q['vid']}.mp4",
        "audio": f"{AUDIO_DIR}/{q['vid']}.wav",
        "use_audio": True,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt",   "value": GPT_PLACEHOLDER},
        ],
        "gt_label": q["label"],
        "gt_segments": [q["segment"]],
        "duration": q["duration"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(args.input) as f:
        db = json.load(f)
    queries = flatten(db)
    print(f"Flattened {len(queries)} queries from {args.input}")

    end = len(queries) if args.limit is None else min(len(queries), args.start + args.limit)
    queries = queries[args.start:end]
    print(f"Using queries {args.start}..{end-1} (n={len(queries)})")

    # sanity: first-10 file existence
    missing_v = sum(1 for q in queries[:10]
                    if not os.path.exists(f"{VIDEO_DIR}/{q['vid']}.mp4"))
    missing_a = sum(1 for q in queries[:10]
                    if not os.path.exists(f"{AUDIO_DIR}/{q['vid']}.wav"))
    print(f"first-10 sanity: missing_video={missing_v}, missing_audio={missing_a}")

    rows = [to_salmonn_row(q) for q in queries]
    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(rows)} rows to {args.output}")

    # preview
    print("\n--- row 0 preview ---")
    print("  Q :", rows[0]["conversations"][0]["value"][:220])
    print("  GT:", rows[0]["gt_segments"], "dur=", rows[0]["duration"])


if __name__ == "__main__":
    main()
