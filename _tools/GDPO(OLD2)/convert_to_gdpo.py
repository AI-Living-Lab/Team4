#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_to_gdpo.py
  기존 VS2 SFT/DPO 학습 데이터를 GDPO용 프롬프트+GT 형식으로 변환.

  GDPO는 모델이 직접 답변을 생성하므로 chosen/rejected 응답이 필요 없고,
  프롬프트 + GT(정답)만 있으면 됩니다.

Usage:
  python _tools/GDPO/convert_to_gdpo.py \
    --input  data/unav100_train_dense.json \
    --output data/unav100_train_gdpo.json
"""

import argparse
import json
import os


def convert_sample(sample: dict) -> dict:
    """SFT/DPO 샘플 하나를 GDPO 형식으로 변환.

    입력 (SFT dense 형식):
      {
        "video": "...", "audio": "...",
        "mode": "dense",
        "events": [{"label": "...", "timestamps": [s, e]}, ...],
        "conversations": [{"from": "human", ...}, {"from": "gpt", ...}]
      }

    출력 (GDPO 형식):
      {
        "video": "...", "audio": "...",
        "prompt": "...",
        "events": [{"label": "...", "timestamps": [s, e]}, ...]
      }
    """
    # 프롬프트 추출 (human turn)
    prompt = ""
    for turn in sample.get("conversations", []):
        if turn["from"] == "human":
            prompt = turn["value"]
            break

    gdpo_sample = {
        "video": sample["video"],
        "audio": sample["audio"],
        "prompt": prompt,
    }

    # dense mode: events 리스트
    if sample.get("mode") == "dense" and "events" in sample:
        gdpo_sample["events"] = sample["events"]

    # single mode: timestamps
    elif "timestamps" in sample:
        gdpo_sample["events"] = [{
            "label": extract_label_from_prompt(prompt),
            "timestamps": sample["timestamps"],
        }]

    return gdpo_sample


def extract_label_from_prompt(prompt: str) -> str:
    """프롬프트에서 이벤트 라벨 추출 (single mode용).

    예: "... Identify the start and end time of the following event: dog barking"
    """
    markers = [
        "following event:",
        "the event:",
        "event:",
    ]
    lower = prompt.lower()
    for marker in markers:
        idx = lower.find(marker)
        if idx != -1:
            label = prompt[idx + len(marker):].strip()
            # 줄바꿈이나 특수 토큰 전까지만
            for stop in ["\n", "<", '"']:
                stop_idx = label.find(stop)
                if stop_idx != -1:
                    label = label[:stop_idx]
            return label.strip()
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="VS2 SFT/DPO 데이터를 GDPO 프롬프트+GT 형식으로 변환"
    )
    parser.add_argument("--input", required=True, help="입력 JSON 파일 경로")
    parser.add_argument("--output", required=True, help="출력 JSON 파일 경로")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted = []
    skipped = 0
    for sample in data:
        result = convert_sample(sample)
        if result.get("events"):
            converted.append(result)
        else:
            skipped += 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"[convert_to_gdpo] 변환 완료: {len(converted)}개 / 스킵: {skipped}개")
    print(f"[convert_to_gdpo] 출력: {args.output}")


if __name__ == "__main__":
    main()
