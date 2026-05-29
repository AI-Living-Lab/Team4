#!/usr/bin/env python3
"""
puvalor_full.json (101,080 samples, with task_type/source_id) 을
기존 puvalor_train/val/test.json 의 video-level split 그대로 재현해서
새 split JSON 으로 분리.

v4 baseline 과 비교 가능성을 보존하기 위해 새 random split 하지 않고,
기존 video → split 매핑을 그대로 사용함.
"""
import json
import os


def get_video_id(s):
    """sample 에서 video basename 추출.
    - 새 full sample: s["id"] 가 video basename (e.g., "6.rNsFSVhZi9w")
    - 구 train/val/test sample: s["video"] path 의 basename
    """
    if s.get("id"):
        return s["id"]
    v = s.get("video", "")
    return os.path.basename(v).replace(".mp4", "")


DATA_DIR = "/home/aix23102/audiolm/vS2_eunji/data"


def main():
    # 1. 기존 split 의 video 별 assignment 추출
    print("Loading existing splits to extract video-level assignment...")
    splits = {}
    for name in ["train", "val", "test"]:
        with open(os.path.join(DATA_DIR, f"puvalor_{name}.json")) as f:
            old = json.load(f)
        splits[name] = {get_video_id(s) for s in old}
        print(f"  {name}: {len(old)} samples, {len(splits[name])} unique videos")

    # disjoint check
    assert len(splits["train"] & splits["val"]) == 0
    assert len(splits["train"] & splits["test"]) == 0
    assert len(splits["val"] & splits["test"]) == 0
    all_vids = splits["train"] | splits["val"] | splits["test"]
    print(f"  union: {len(all_vids)} videos (disjoint)")

    # 2. puvalor_full.json 로드 후 video 별 split 적용
    print("\nLoading puvalor_full.json...")
    with open(os.path.join(DATA_DIR, "puvalor_full.json")) as f:
        full = json.load(f)
    print(f"  {len(full)} samples")

    out = {"train": [], "val": [], "test": []}
    n_unknown = 0
    for s in full:
        vid = get_video_id(s)
        assigned = None
        for name in ["train", "val", "test"]:
            if vid in splits[name]:
                assigned = name
                break
        if assigned is None:
            n_unknown += 1
            continue
        out[assigned].append(s)

    print(f"\nSplit results (sample count):")
    for name in ["train", "val", "test"]:
        print(f"  {name}: {len(out[name])}")
    print(f"  unassigned: {n_unknown}")

    # task_type 분포
    from collections import Counter
    for name in ["train", "val", "test"]:
        c = Counter(s["task_type"] for s in out[name])
        print(f"  {name} task distribution: {dict(c)}")

    # 3. 저장
    print("\nSaving split files...")
    for name in ["train", "val", "test"]:
        path = os.path.join(DATA_DIR, f"puvalor_{name}_v5.json")
        with open(path, "w") as f:
            json.dump(out[name], f)
        sz = os.path.getsize(path) / 1024 / 1024
        print(f"  {path} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
