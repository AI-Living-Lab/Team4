"""
Convert LongVALE evaluation annotations to Avicuna's `database` schema.

LongVALE schema:
  {vid: {duration, timestamps: [[s,e], ...], sentences: [str, ...]}}

Avicuna schema (used by extract_features.py / inference_unav_batch.py):
  {"database": {vid: {"subset": "test", "duration": float,
                       "annotations": [{"segment": [s,e], "label": str}, ...]}}}

Usage:
  python convert_longvale_to_avicuna.py \
      --input  /workspace/datasets/LongVALE/longvale-annotations-eval.json \
      --output /workspace/jsy/output/avicuna_longvale/longvale_annotations.json \
      --video_dir /workspace/datasets/LongVALE/videos
"""

import argparse
import json
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--video_dir", default="/workspace/datasets/LongVALE/videos",
                   help="Drop entries whose mp4 is missing (safety check)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        longvale = json.load(f)

    database = {}
    skipped_missing_video = 0
    skipped_bad_shape = 0

    for vid, info in longvale.items():
        mp4 = os.path.join(args.video_dir, f"{vid}.mp4")
        if not os.path.exists(mp4):
            skipped_missing_video += 1
            continue

        duration = float(info["duration"])
        tss = info.get("timestamps", [])
        sents = info.get("sentences", [])
        if len(tss) != len(sents) or len(tss) == 0:
            skipped_bad_shape += 1
            continue

        anns = []
        for (s, e), sent in zip(tss, sents):
            s = max(0.0, float(s))
            e = min(duration, float(e))
            if e <= s:
                continue
            anns.append({"segment": [s, e], "label": sent.strip()})

        if not anns:
            skipped_bad_shape += 1
            continue

        database[vid] = {
            "subset": "test",
            "duration": duration,
            "annotations": anns,
        }

    out = {"database": database}
    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    n_videos = len(database)
    n_queries = sum(len(v["annotations"]) for v in database.values())
    print(f"Input videos  : {len(longvale)}")
    print(f"Missing mp4   : {skipped_missing_video}")
    print(f"Bad shape     : {skipped_bad_shape}")
    print(f"Output videos : {n_videos}")
    print(f"Output queries: {n_queries}")
    print(f"Saved to      : {args.output}")


if __name__ == "__main__":
    main()
