#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os


def load_unav_db(unav_json: str):
    with open(unav_json, "r", encoding="utf-8") as f:
        return json.load(f)["database"]


# ──────────────────────────────────────────────
# SINGLE mode
# ──────────────────────────────────────────────

def build_single_prompt(event: str) -> str:
    return (
        "<image>\n"
        "You are an audio-visual event localization model.\n"
        "Given the video and audio, identify the start and end time of the following event.\n\n"
        f"Event: {event}\n\n"
        "Answer with exactly one line in the following format,\n"
        "where each time is expressed as six time tokens "
        "(4-digit integer part, <tdot>, 1-digit decimal):\n"
        "start: <tX><tX><tX><tX><tdot><tX> end: <tX><tX><tX><tX><tdot><tX>"
    )


def build_single_gt_answer(start: float, end: float) -> str:
    # av_dataset.py의 _get_item에서 "timestamps" 필드를 읽어
    # VTG-LLM 토큰으로 변환해 덮어쓰므로, 여기선 placeholder만 둔다.
    return "PLACEHOLDER"


def build_single_samples_from_unav(db, split, video_dir, audio_dir, skip_missing_files=False):
    samples = []
    skipped_missing = 0

    for vid, item in db.items():
        if item.get("subset", "").lower() != split.lower():
            continue

        video_path = os.path.join(video_dir, f"{vid}.mp4")
        audio_path = os.path.join(audio_dir, f"{vid}.wav")

        if skip_missing_files and (
            not os.path.exists(video_path) or not os.path.exists(audio_path)
        ):
            skipped_missing += 1
            continue

        for ann in item.get("annotations", []):
            label = ann["label"]
            start, end = ann["segment"]

            samples.append({
                "video": video_path,
                "audio": audio_path,
                "mode": "single",                          # av_dataset 분기 키
                "timestamps": [float(start), float(end)],  # av_dataset이 VTG-LLM 토큰으로 변환
                "conversations": [
                    {"from": "human", "value": build_single_prompt(label)},
                    {"from": "gpt",   "value": build_single_gt_answer(start, end)},
                ],
            })

    return samples, skipped_missing


# ──────────────────────────────────────────────
# DENSE mode
# ──────────────────────────────────────────────

def build_dense_prompt() -> str:
    return (
        "<image>\n"
        "You are an audio-visual event localization model.\n"
        "Given the video and audio, localize all audio-visual events.\n\n"
        "Output a JSON list. Each element must have:\n"
        "  \"event\": event label (string)\n"
        "  \"start\": six time tokens (4-digit integer part, <tdot>, 1-digit decimal)\n"
        "  \"end\":   six time tokens (same format)\n"
        "Example: [{\"event\": \"dog barking\", "
        "\"start\": \"<t0><t0><t1><t2><tdot><t3>\", "
        "\"end\": \"<t0><t0><t4><t5><tdot><t6>\"}]"
    )


def build_dense_gt_answer_placeholder() -> str:
    # av_dataset.py의 _get_item에서 "events" 필드를 읽어
    # VTG-LLM 토큰으로 변환해 덮어쓰므로, 여기선 placeholder만 둔다.
    return "PLACEHOLDER"


def build_dense_samples_from_unav(db, split, video_dir, audio_dir, skip_missing_files=False):
    samples = []
    skipped_missing = 0

    for vid, item in db.items():
        if item.get("subset", "").lower() != split.lower():
            continue

        video_path = os.path.join(video_dir, f"{vid}.mp4")
        audio_path = os.path.join(audio_dir, f"{vid}.wav")

        if skip_missing_files and (
            not os.path.exists(video_path) or not os.path.exists(audio_path)
        ):
            skipped_missing += 1
            continue

        annotations = item.get("annotations", [])
        if not annotations:
            continue

        # av_dataset.py가 VTG-LLM 토큰 변환에 사용할 이벤트 리스트
        events = [
            {
                "label": ann["label"],
                "timestamps": [float(ann["segment"][0]), float(ann["segment"][1])],
            }
            for ann in annotations
        ]

        samples.append({
            "video": video_path,
            "audio": audio_path,
            "mode": "dense",    # av_dataset 분기 키
            "events": events,   # 모든 이벤트 (av_dataset이 GT 생성)
            "conversations": [
                {"from": "human", "value": build_dense_prompt()},
                {"from": "gpt",   "value": build_dense_gt_answer_placeholder()},
            ],
        })

    return samples, skipped_missing


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unav_json", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip_missing_files", action="store_true")
    ap.add_argument(
        "--mode",
        choices=["dense", "single"],
        default="dense",
        help=(
            "dense:  one sample per video (all events at once) [default]\n"
            "single: one sample per annotation (original behaviour)"
        ),
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    db = load_unav_db(args.unav_json)

    if args.mode == "dense":
        samples, skipped_missing = build_dense_samples_from_unav(
            db=db,
            split=args.split,
            video_dir=args.video_dir,
            audio_dir=args.audio_dir,
            skip_missing_files=args.skip_missing_files,
        )
    else:
        samples, skipped_missing = build_single_samples_from_unav(
            db=db,
            split=args.split,
            video_dir=args.video_dir,
            audio_dir=args.audio_dir,
            skip_missing_files=args.skip_missing_files,
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[OK]  wrote: {args.out}")
    print(f"[INFO] mode={args.mode}, split={args.split}, samples={len(samples)}")
    if args.skip_missing_files:
        print(f"[INFO] skipped missing files: {skipped_missing}")


if __name__ == "__main__":
    main()
