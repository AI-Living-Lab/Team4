#!/usr/bin/env python3
"""
PU-VALOR 데이터를 SALMONN2+ 학습 형식으로 변환.

변경사항:
  - <image> → <video>
  - audio 필드 → use_audio: true
  - conversations 구조 유지 (human/gpt)
"""
import json
import argparse
import os


def convert_sample(item):
    new_item = {
        "video": item["video"],
        "use_audio": True,
        "conversations": [],
    }

    for conv in item.get("conversations", []):
        new_conv = {
            "from": conv["from"],
            "value": conv["value"].replace("<image>", "<video>"),
        }
        new_item["conversations"].append(new_conv)

    # ce_only 필드 보존 (있으면)
    if "ce_only" in item:
        new_item["ce_only"] = item["ce_only"]

    return new_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="PU-VALOR json path")
    ap.add_argument("--output", required=True, help="Output json path")
    args = ap.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    converted = [convert_sample(item) for item in data]

    with open(args.output, "w") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(converted)} samples")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")

    # 샘플 확인
    print("\n=== Sample ===")
    for conv in converted[0]["conversations"][:2]:
        print(f'  [{conv["from"]}]: {conv["value"][:150]}')


if __name__ == "__main__":
    main()
