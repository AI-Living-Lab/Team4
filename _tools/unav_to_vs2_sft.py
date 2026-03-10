#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os


def load_unav_db(unav_json: str):
    with open(unav_json, "r", encoding="utf-8") as f:
        return json.load(f)["database"]


def build_single_prompt(event: str) -> str:
    return (
        "<image>\n"
        "You are an audio-visual event localization model.\n"
        "Given the video and audio, identify the start and end time of the following event.\n\n"
        f"Event: {event}\n\n"
        "Answer with exactly one line using ONLY special time tokens from <t0> to <t299>:\n"
        "start: <t_start> end: <t_end>"
    )

def build_single_gt_answer(start: float, end: float) -> str:
    return f"start: {float(start):.1f} end: {float(end):.1f}"


def build_single_samples_from_unav(db, split, video_dir, audio_dir, skip_missing_files=False):
    samples = []
    skipped_missing = 0

    for vid, item in db.items():
        if item.get("subset", "").lower() != split.lower():
            continue

        video_path = os.path.join(video_dir, f"{vid}.mp4")
        audio_path = os.path.join(audio_dir, f"{vid}.wav")

        if skip_missing_files and (not os.path.exists(video_path) or not os.path.exists(audio_path)):
            skipped_missing += 1
            continue

        for ann in item.get("annotations", []):
            label = ann["label"]
            start, end = ann["segment"]

            samples.append({
                "video": video_path,
                "audio": audio_path,
                "timestamps": [float(start), float(end)],
                "conversations": [
                    {"from": "human", "value": build_single_prompt(label)},
                    {"from": "gpt", "value": build_single_gt_answer(start, end)}
                ]
            })

    return samples, skipped_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unav_json", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip_missing_files", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    db = load_unav_db(args.unav_json)
    samples, skipped_missing = build_single_samples_from_unav(
        db=db,
        split=args.split,
        video_dir=args.video_dir,
        audio_dir=args.audio_dir,
        skip_missing_files=args.skip_missing_files,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {args.out}")
    print(f"[INFO] split={args.split}, samples={len(samples)}")
    if args.skip_missing_files:
        print(f"[INFO] skipped missing files: {skipped_missing}")


if __name__ == "__main__":
    main()