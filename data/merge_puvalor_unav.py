#!/usr/bin/env python3
"""
PU-VALOR train (stage3-style, with task_type/source_id) + UnAV train (stage3-style)
→ merged training JSON.

Schema 통일:
  - id, video, audio, use_audio, conversations, meta (duration, token), source, source_id, task_type
  - PU-VALOR 의 meta 추가 필드 (sub_videos, scale) 는 보존
  - UnAV 의 event (T2만) 보존
"""
import json
import os
import random
import argparse


DATA_DIR = "/home/aix23102/audiolm/vS2_eunji/data"

PUVALOR_VIDEO_ROOT = "/data0/aix23102/PU-VALOR/videos"
PUVALOR_AUDIO_ROOT = "/data0/aix23102/PU-VALOR/audios"


def normalize_puvalor(s):
    """PU-VALOR sample 에 video/audio path, use_audio 추가 (full 추출 시 빠졌던 필드)."""
    vid = s.get("id", "")
    out = dict(s)
    if "video" not in out:
        out["video"] = os.path.join(PUVALOR_VIDEO_ROOT, f"{vid}.mp4")
    if "audio" not in out:
        out["audio"] = os.path.join(PUVALOR_AUDIO_ROOT, f"{vid}.wav")
    if "use_audio" not in out:
        out["use_audio"] = True
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--puvalor", default=os.path.join(DATA_DIR, "puvalor_train_v5.json"))
    parser.add_argument("--unav", default=os.path.join(DATA_DIR, "unav_train.json"))
    parser.add_argument("--output", default=os.path.join(DATA_DIR, "puvalor_unav_train.json"))
    parser.add_argument("--shuffle_seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading PU-VALOR train: {args.puvalor}")
    with open(args.puvalor) as f:
        pu = json.load(f)
    print(f"  {len(pu)} samples")

    print(f"Loading UnAV train: {args.unav}")
    with open(args.unav) as f:
        un = json.load(f)
    print(f"  {len(un)} samples")

    # Normalize PU-VALOR (add video/audio path)
    pu_norm = [normalize_puvalor(s) for s in pu]

    # Merge
    merged = pu_norm + un
    print(f"\nBefore shuffle: {len(merged)} samples")

    # Source/task 분포
    from collections import Counter
    c_src = Counter(s["source_id"] for s in merged)
    c_task = Counter((s["source_id"], s["task_type"]) for s in merged)
    print(f"source_id: {dict(c_src)}")
    print("source_id x task_type:")
    for k, v in sorted(c_task.items()):
        print(f"  {k}: {v}")

    # SAOC 대상 (T3 제외) 비율 재확인
    saoc_pu = sum(v for k, v in c_task.items() if k[0] == "puvalor" and k[1] in ("T1", "T2"))
    saoc_un = sum(v for k, v in c_task.items() if k[0] == "unav")
    print(f"\nSAOC-eligible samples: PU-VALOR(T1+T2)={saoc_pu}, UnAV={saoc_un}, ratio={saoc_pu/saoc_un:.2f}:1")

    # Shuffle for training
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(merged)

    # Schema check — 모든 sample 에 필수 필드 있는지
    REQUIRED = ["video", "audio", "use_audio", "conversations", "meta", "source_id", "task_type"]
    missing = []
    for i, s in enumerate(merged):
        for k in REQUIRED:
            if k not in s:
                missing.append((i, k))
                break
        if i < 5:
            assert "duration" in s["meta"] and "token" in s["meta"], f"sample {i} missing meta.duration/token"
    if missing:
        print(f"\nWARNING: {len(missing)} samples missing required fields (first 5): {missing[:5]}")
    else:
        print("\nSchema check: all samples have required fields.")

    # Save
    with open(args.output, "w") as f:
        json.dump(merged, f)
    sz = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nSaved → {args.output} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
