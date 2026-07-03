#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_arc.py — ARC-Hunyuan-Video-7B 결과 채점 래퍼 (chronus 방식과 동형).

ARC 러너(arc_infer_unav.py)는 결과를 이미 {video, gt_label, gt_segments, pred} 로 저장한다
(pred = "<think>...</think><answer><span>HH:MM:SS - HH:MM:SS</span></answer>").
따라서 gt embed·pred 변환이 이미 되어 있어, chunk merge → eval_miou --natural → maketable 만 하면 된다.
  - eval_miou 는 <answer>..</answer> 안만 자동 추출, --natural 로 HH:MM:SS 파싱.

부가: --save_inputs 로 경로보정된 입력 청크를 <out_dir>/../inputs/ 에 저장(chronus 구조와 통일).

사용:
  python3 eval_arc.py                     # 기본 경로로 현재까지 결과 채점 → table.txt
  python3 eval_arc.py --save_inputs       # + 경로보정 입력 저장
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = "/home/aix23102/audiolm/workspace"
DEF_RESULTS = f"{WS}/outputs/base/ARC-Hunyuan-Video-7B/unav100_multiseg/results"
DEF_ARC_CHUNKS = f"{WS}/data/test/unav100_arc"


def _load(p):
    with open(p) as f:
        return json.load(f)


def fix_path(p):
    if p.startswith("/data0/aix23102/unav_100/"):
        return p.replace("/data0/aix23102/unav_100/", f"{WS}/datasets/unav_100/", 1)
    if p.startswith("/workspace/"):
        return p.replace("/workspace/", f"{WS}/", 1)
    return p


def save_inputs(chunks_dir, inputs_dir):
    os.makedirs(inputs_dir, exist_ok=True)
    n = 0
    for cf in sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.json"))):
        d = _load(cf)
        for x in d:
            x["video"] = fix_path(x["video"])
            if "audio" in x:
                x["audio"] = fix_path(x["audio"])
        out = os.path.join(inputs_dir, os.path.basename(cf))
        json.dump(d, open(out, "w"), ensure_ascii=False, indent=2)
        n += 1
    print(f"[eval_arc] saved {n} corrected input chunks → {inputs_dir}")


def merge_results(results_dir):
    chunks = sorted(glob.glob(os.path.join(results_dir, "chunk_*.json")))
    if not chunks:
        raise SystemExit(f"[eval_arc] no result chunks in {results_dir}")
    out = []
    for c in chunks:
        out.extend(_load(c))
    print(f"[eval_arc] merged {len(chunks)} result chunks → {len(out)} samples")
    return out


def main():
    ap = argparse.ArgumentParser(description="ARC-Hunyuan 결과 채점 → table.txt")
    ap.add_argument("--results_dir", default=DEF_RESULTS)
    ap.add_argument("--arc_chunks", default=DEF_ARC_CHUNKS, help="경로보정 입력 저장 소스")
    ap.add_argument("--eval_dir", default=None, help="기본 = results_dir 형제 'eval'")
    ap.add_argument("--label", default="ARC-Hunyuan-Video-7B")
    ap.add_argument("--testset", default="unav100_multiseg")
    ap.add_argument("--save_inputs", action="store_true")
    args = ap.parse_args()

    base = os.path.dirname(args.results_dir)  # .../unav100_multiseg
    eval_dir = args.eval_dir or os.path.join(base, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    if args.save_inputs:
        save_inputs(args.arc_chunks, os.path.join(base, "inputs"))

    # merge → test_results_rank0.json (이미 pred/gt_segments embed 됨)
    results = merge_results(args.results_dir)
    rank0 = os.path.join(eval_dir, "test_results_rank0.json")
    with open(rank0, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[eval_arc] wrote {rank0}")

    # eval_miou.py --natural  (<answer> 자동추출 + HH:MM:SS 파싱)
    cmd = [sys.executable, os.path.join(HERE, "eval_miou.py"), rank0,
           "--natural", "--label", args.label, "--testset", args.testset]
    print("[eval_arc] $", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # maketable.py → table.txt
    cmd = [sys.executable, os.path.join(HERE, "maketable.py"), eval_dir]
    print("[eval_arc] $", " ".join(cmd))
    subprocess.run(cmd, check=True)

    table = os.path.join(eval_dir, "table.txt")
    print(f"\n[eval_arc] DONE → {table}\n")
    if os.path.exists(table):
        print(open(table).read())


if __name__ == "__main__":
    main()
