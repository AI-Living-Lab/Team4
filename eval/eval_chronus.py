#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_chronus.py — ChronusOmni 추론 결과 채점 래퍼.

Chronus inference/eval.py 는 결과를 [{"id","question","output"}, ...] 로 저장한다.
통합 평가기(eval_miou.py)는 예측을 `pred` 필드에서 읽고, GT 는 embedded `gt_segments`
를 우선 쓴다. 그런데 chronus 결과엔 `pred`도 `gt_segments`도 없다.
이 스크립트가 그 간극을 메운다:

  1) results 디렉토리의 chunk_*.json (또는 단일 파일) 을 모두 concat
  2) 각 항목을 id 로 test_json(unav100_chronus.json) 과 매칭 →
     {"id", "gt_label", "gt_segments"(embed), "pred"(=output)} 로 변환
  3) <eval_dir>/test_results_rank0.json 로 저장
  4) eval_miou.py --natural 실행 → 3종 summary JSON
  5) maketable.py 실행 → <eval_dir>/table.txt (최종 산출물)

사용:
  python3 eval_chronus.py \
     --results_dir /workspace/outputs/base/ChronusOmni/unav100_multiseg/results \
     --test_json   /workspace/data/test/unav100_chronus.json \
     [--eval_dir <출력폴더>] [--label ChronusOmni] [--testset unav100_multiseg]

기본값은 리포지토리 표준 경로를 쓰므로 인자 없이도 동작한다.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = "/home/aix23102/audiolm/workspace"
DEF_RESULTS = f"{WS}/outputs/base/ChronusOmni/unav100_multiseg/results"
DEF_TESTJSON = f"{WS}/data/test/unav100_chronus.json"


def _load(p):
    with open(p) as f:
        return json.load(f)


def merge_results(results_dir):
    """chunk_*.json (있으면) 전부 concat, 없으면 디렉토리 자체를 파일로 취급."""
    if os.path.isfile(results_dir):
        return _load(results_dir)
    chunks = sorted(glob.glob(os.path.join(results_dir, "chunk_*.json")))
    if not chunks:
        raise SystemExit(f"[eval_chronus] chunk_*.json 없음: {results_dir}")
    out = []
    for c in chunks:
        out.extend(_load(c))
    print(f"[eval_chronus] merged {len(chunks)} chunks → {len(out)} samples")
    return out


def transform(results, test_json):
    gt = {x["id"]: x for x in _load(test_json)}
    conv, miss = [], 0
    for r in results:
        g = gt.get(r["id"])
        if g is None:
            miss += 1
            continue
        conv.append({
            "id": r["id"],
            "gt_label": g.get("gt_label", ""),
            "gt_segments": g.get("gt_segments", []),
            "pred": r.get("output", ""),   # output → pred
        })
    if miss:
        print(f"[eval_chronus] [WARN] {miss} 샘플 id 매칭 실패 → 제외")
    print(f"[eval_chronus] transformed {len(conv)} samples (output→pred, gt embedded)")
    return conv


def main():
    ap = argparse.ArgumentParser(description="ChronusOmni 결과 채점 → table.txt")
    ap.add_argument("--results_dir", default=DEF_RESULTS,
                    help="chunk_*.json 들이 있는 폴더 또는 단일 결과 파일")
    ap.add_argument("--test_json", default=DEF_TESTJSON, help="GT json (id 매칭용)")
    ap.add_argument("--eval_dir", default=None,
                    help="채점 산출물 폴더 (기본 = results_dir 의 형제 'eval')")
    ap.add_argument("--label", default="ChronusOmni")
    ap.add_argument("--testset", default="unav100_multiseg")
    args = ap.parse_args()

    base = args.results_dir if os.path.isdir(args.results_dir) \
        else os.path.dirname(args.results_dir)
    eval_dir = args.eval_dir or os.path.join(os.path.dirname(base), "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # 1~3) merge + transform → test_results_rank0.json
    results = merge_results(args.results_dir)
    conv = transform(results, args.test_json)
    rank0 = os.path.join(eval_dir, "test_results_rank0.json")
    with open(rank0, "w") as f:
        json.dump(conv, f, ensure_ascii=False, indent=2)
    print(f"[eval_chronus] wrote {rank0}")

    # 4) eval_miou.py --natural  (embedded gt_segments 자동사용, second{} 파싱)
    cmd = [sys.executable, os.path.join(HERE, "eval_miou.py"), rank0,
           "--natural", "--label", args.label, "--testset", args.testset]
    print("[eval_chronus] $", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # 5) maketable.py → table.txt (eval_dir 하위 summary 1행)
    cmd = [sys.executable, os.path.join(HERE, "maketable.py"), eval_dir]
    print("[eval_chronus] $", " ".join(cmd))
    subprocess.run(cmd, check=True)

    table = os.path.join(eval_dir, "table.txt")
    print(f"\n[eval_chronus] DONE → {table}\n")
    if os.path.exists(table):
        with open(table) as f:
            print(f.read())


if __name__ == "__main__":
    main()
