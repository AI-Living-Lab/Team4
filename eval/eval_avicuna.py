#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_avicuna.py — AVicuna 추론 결과 채점 래퍼 (chronus 패턴 복제).

infer_thumos_avicuna.py 는 각 청크 결과를 이미
  {video, audio, gt_label, gt_segments, id, duration, pred(=초변환본), raw_pred}
로 저장한다(0~99 → 초 변환이 pred에 반영됨). 따라서 chronus처럼 output→pred 변환이
필요 없고, chunk merge 후 곧장 eval_miou.py --natural 로 채점하면 된다.

  1) results 디렉토리의 chunk_*.json 전부 concat
  2) <eval_dir>/test_results_rank0.json 저장 (pred·gt_segments 이미 들어있음)
  3) eval_miou.py --natural 실행 → 3종 summary
  4) maketable.py 실행 → <eval_dir>/table.txt

사용(기본 경로 내장, 인자 없이도 동작):
  python3 eval_avicuna.py \
    --results_dir /home/aix23102/audiolm/workspace/outputs/base/AVicuna/unav100_multiseg/results
"""
import argparse, glob, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = "/home/aix23102/audiolm/workspace"
DEF_RESULTS = f"{WS}/outputs/base/AVicuna/unav100_multiseg/results"


def _load(p):
    with open(p) as f:
        return json.load(f)


def merge_results(results_dir):
    if os.path.isfile(results_dir):
        return _load(results_dir)
    chunks = sorted(glob.glob(os.path.join(results_dir, "chunk_*.json")))
    if not chunks:
        raise SystemExit(f"[eval_avicuna] chunk_*.json 없음: {results_dir}")
    out = []
    for c in chunks:
        out.extend(_load(c))
    print(f"[eval_avicuna] merged {len(chunks)} chunks → {len(out)} samples")
    return out


def main():
    ap = argparse.ArgumentParser(description="AVicuna 결과 채점 → table.txt")
    ap.add_argument("--results_dir", default=DEF_RESULTS)
    ap.add_argument("--eval_dir", default=None,
                    help="기본 = results_dir 형제 'eval'")
    ap.add_argument("--label", default="AVicuna")
    ap.add_argument("--testset", default="unav100_multiseg")
    args = ap.parse_args()

    base = args.results_dir if os.path.isdir(args.results_dir) \
        else os.path.dirname(args.results_dir)
    eval_dir = args.eval_dir or os.path.join(os.path.dirname(base), "eval")
    os.makedirs(eval_dir, exist_ok=True)

    results = merge_results(args.results_dir)
    # pred가 비어있는(추론실패) 샘플 수 리포트 — 제외하진 않음(공정성)
    empty = sum(1 for r in results if not str(r.get("pred", "")).strip())
    if empty:
        print(f"[eval_avicuna] [WARN] pred 비어있는 샘플 {empty}개(추론실패) 포함")

    rank0 = os.path.join(eval_dir, "test_results_rank0.json")
    with open(rank0, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[eval_avicuna] wrote {rank0}")

    # pred는 이미 초 단위(rescale됨) → --natural 만. gt_segments embed 자동사용.
    cmd = [sys.executable, os.path.join(HERE, "eval_miou.py"), rank0,
           "--natural", "--label", args.label, "--testset", args.testset]
    print("[eval_avicuna] $", " ".join(cmd)); subprocess.run(cmd, check=True)

    cmd = [sys.executable, os.path.join(HERE, "maketable.py"), eval_dir]
    print("[eval_avicuna] $", " ".join(cmd)); subprocess.run(cmd, check=True)

    table = os.path.join(eval_dir, "table.txt")
    print(f"\n[eval_avicuna] DONE → {table}\n")
    if os.path.exists(table):
        with open(table) as f:
            print(f.read())


if __name__ == "__main__":
    main()
