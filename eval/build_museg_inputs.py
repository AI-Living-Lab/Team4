#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_museg_inputs.py — MUSEG 추론용 입력 청크 생성.

data/test/unav100_museg.json (옛 서버 경로 + format 블록 포함) →
outputs/base/MUSEG/unav100_multiseg/inputs/chunk_XXXX.json 로:
  1) video 경로 rewrite: /workspace/... → 이 서버 실제 루트
  2) question 에서 "Answer format" 블록 제거(stem만 남김).
     MUSEG 네이티브 TEMPLATE(eval_grounding.py)이 <think>/<answer> 형식을
     이미 강제하므로 우리 format 블록은 중복 → 제거.
  3) eval_grounding.py 가 요구하는 groundtruth 필드 보존.
경로/청크크기는 인자로 조정. 멱등(재실행 안전).
"""
import argparse, json, os

OLD = "/workspace/datasets/unav_100/"
NEW = "/home/aix23102/audiolm/workspace/datasets/unav_100/"
MARK = "\n\nAnswer format:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/test/unav100_museg.json")
    ap.add_argument("--out_dir", default="outputs/base/MUSEG/unav100_multiseg/inputs")
    ap.add_argument("--chunk_size", type=int, default=500)
    ap.add_argument("--old", default=OLD)
    ap.add_argument("--new", default=NEW)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = json.load(open(args.src))
    print(f"source samples: {len(data)}")

    miss, built = 0, []
    for x in data:
        v = x["video"].replace(args.old, args.new)
        if not os.path.exists(v):
            miss += 1
        q = x["question"].split(MARK)[0].strip()
        built.append({
            "id": x["id"],
            "vid": x.get("vid", ""),
            "video": v,
            "question": q,
            "gt_label": x.get("gt_label", ""),
            "gt_segments": x.get("gt_segments", []),
            "groundtruth": x.get("groundtruth", x.get("gt_segments", [])),
        })
    print(f"[warn] missing video files after rewrite: {miss}" if miss else "all videos exist")

    n = (len(built) + args.chunk_size - 1) // args.chunk_size
    for i in range(n):
        seg = built[i*args.chunk_size:(i+1)*args.chunk_size]
        p = os.path.join(args.out_dir, f"chunk_{i:04d}.json")
        json.dump(seg, open(p, "w"), ensure_ascii=False, indent=1)
        print(f"  wrote {p} ({len(seg)})")


if __name__ == "__main__":
    main()
